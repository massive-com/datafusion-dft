//! The needles and scans as SQL through DataFusion, on Parquet and on Vortex (see [`Reader`]),
//! at one partition and at one per core. Files: `common::data`.
//!
//! Ids are `df needle/<needle>/<reader>/<partitions>` and
//! `df scan/<projection>/<reader>/<partitions>`; pick them with Criterion's filter:
//!
//! ```text
//! cargo bench --bench df -- 'high > 1000/vortex/1$'
//! cargo bench --bench df -- "$(cat crates/datafusion-vortex-ohlc/benches/quick.filter)"
//! ```
//!
//! Every reader that runs is first checked against Parquet. `VORTEX_OHLC_SOURCE` names the
//! source the files are written from. `DF_BENCH_VORTEX=<path>` benches another Vortex file; its
//! readers are named `<reader>@<file stem>`, and only it is read by `vortex v2`. Parquet's last
//! results are in `target/vortex-ohlc-bench/parquet-results.txt`.

mod common;

use std::sync::Arc;
use std::time::Duration;

use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Schema, SchemaRef};
use criterion::{criterion_group, criterion_main, Criterion, SamplingMode, Throughput};
use datafusion::common::hash_utils::{with_hashes, RandomState};
use datafusion::datasource::file_format::{FileFormat, FileFormatFactory};
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::datasource::physical_plan::FileScanConfig;
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::physical_plan::{execute_stream, ExecutionPlan};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_vortex_ohlc::vortex_format_factory;
use datafusion_vortex_prune::{reader_factory, PrunedTable, VortexExecFactory, ZonePruner};
use futures::StreamExt;
use tokio::runtime::Runtime;
use vortex::scan::selection::Selection as ScanSelection;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_datafusion::{VortexAccessPlan, VortexFormat, VortexTableOptions};

use common::needle::{needles, Needle, Op, Value};
use common::{data, Selection};

/// The table every query reads.
const TABLE: &str = "bars";

/// How a table reads its file.
#[derive(Clone, Copy, PartialEq)]
enum Reader {
    Parquet,
    /// Parquet with `pushdown_filters`: the filter decodes its columns first, and the others
    /// only for the rows that pass.
    ParquetPushdown,
    /// The Vortex file, its scans pruned by a `ZonePruner` before they start.
    Vortex,
    /// The Vortex file, pruned on Vortex's own zone maps and by the bars layout as the scan runs:
    /// vortex-ohlc alone.
    VortexUnpruned,
    /// The Vortex file read with Vortex's V2 scan, which prunes a scan's zones before it starts
    /// on its own, and no `ZonePruner`. Only for a `DF_BENCH_VORTEX` file that isn't a bars file:
    /// the bars layout has no V2 scan plan.
    VortexScanV2,
}

impl Reader {
    const ALL: [Reader; 5] = [
        Reader::Parquet,
        Reader::ParquetPushdown,
        Reader::Vortex,
        Reader::VortexUnpruned,
        Reader::VortexScanV2,
    ];

    fn name(self) -> &'static str {
        match self {
            Reader::Parquet => "parquet",
            Reader::ParquetPushdown => "parquet pushdown",
            Reader::Vortex => "vortex",
            Reader::VortexUnpruned => "vortex unpruned",
            Reader::VortexScanV2 => "vortex v2",
        }
    }

    fn is_parquet(self) -> bool {
        matches!(self, Reader::Parquet | Reader::ParquetPushdown)
    }
}

/// The files, and the Parquet table's schema, which the Vortex table takes too.
struct Tables {
    parquet: String,
    vortex: String,
    /// The Vortex file's name, when it isn't the bench's own (see `common::data`).
    vortex_name: Option<String>,
    schema: SchemaRef,
}

impl Tables {
    fn new(runtime: &Runtime) -> Self {
        let files = data::files();
        let parquet = files.parquet.to_str().unwrap().to_string();
        let vortex = files.vortex.to_str().unwrap().to_string();
        let ctx = SessionContext::new();
        runtime
            .block_on(ctx.register_parquet(TABLE, &parquet, ParquetReadOptions::default()))
            .unwrap();
        let schema = Arc::clone(runtime.block_on(ctx.table(TABLE)).unwrap().schema().inner());
        Self {
            parquet,
            vortex,
            vortex_name: files.vortex_name,
            schema,
        }
    }

    /// `reader`'s name in benchmark ids: a Vortex reader of a file other than the bench's own is
    /// `<reader>@<file>`.
    fn reader_name(&self, reader: Reader) -> String {
        match &self.vortex_name {
            Some(file) if !reader.is_parquet() => format!("{}@{file}", reader.name()),
            _ => reader.name().to_string(),
        }
    }

    /// Whether `reader` runs: Vortex's V2 scan can't read the bench's own file, whose bars layout
    /// has no V2 scan plan, so it reads only a `DF_BENCH_VORTEX` file.
    fn runs(&self, reader: Reader) -> bool {
        reader != Reader::VortexScanV2 || self.vortex_name.is_some()
    }

    /// A session on `partitions` partitions with [`TABLE`] read by `reader`.
    fn context(&self, runtime: &Runtime, reader: Reader, partitions: usize) -> SessionContext {
        let mut config = SessionConfig::new().with_target_partitions(partitions);
        if reader == Reader::ParquetPushdown {
            config.options_mut().execution.parquet.pushdown_filters = true;
        }
        let ctx = SessionContext::new_with_config(config);
        if reader.is_parquet() {
            let options = ParquetReadOptions::default();
            runtime
                .block_on(ctx.register_parquet(TABLE, &self.parquet, options))
                .unwrap();
            return ctx;
        }
        // Vortex does its I/O on the runtime it finds when the session is made.
        let _runtime = runtime.enter();
        let session = VortexSession::default();
        // The factory registers vortex-ohlc on the session, which the V2 scan's format reads with
        // too.
        let factory = vortex_format_factory(session.clone());
        let format: Arc<dyn FileFormat> = match reader {
            Reader::VortexScanV2 => {
                let mut options = VortexTableOptions::default();
                options.scan_v2 = true;
                Arc::new(VortexFormat::new_with_options(session.clone(), options))
            }
            _ => factory.default(),
        };
        let options = ListingOptions::new(format).with_file_extension(".vortex");
        let config = ListingTableConfig::new(ListingTableUrl::parse(&self.vortex).unwrap())
            .with_listing_options(options)
            .with_schema(Arc::clone(&self.schema));
        let listing = Arc::new(ListingTable::try_new(config).unwrap());
        // The scans read the file directly, as dft reads a local file.
        let url = ObjectStoreUrl::local_filesystem();
        let store = ctx.runtime_env().object_store(&url).unwrap();
        let mut factory = VortexExecFactory::new(reader_factory(&url, store));
        if reader == Reader::Vortex {
            factory = factory.with_pruner(Arc::new(ZonePruner::new(session)));
        }
        ctx.register_table(TABLE, Arc::new(PrunedTable::new(listing, factory)))
            .unwrap();
        ctx
    }
}

/// Run `sql`, draining its batches; returns the row count.
async fn run(ctx: &SessionContext, sql: &str) -> usize {
    let mut stream = ctx.sql(sql).await.unwrap().execute_stream().await.unwrap();
    let mut rows = 0;
    while let Some(batch) = stream.next().await {
        rows += batch.unwrap().num_rows();
    }
    rows
}

/// What a query returned, whatever the order of its rows: their count, the columns' types, and
/// a sum of each row's hash (DataFusion's, over all its columns, with a fixed seed).
#[derive(Debug, PartialEq)]
struct Fingerprint {
    rows: usize,
    types: Vec<DataType>,
    hash: u64,
}

impl Fingerprint {
    fn new(schema: &Schema) -> Self {
        let types = (schema.fields().iter())
            .map(|f| f.data_type().clone())
            .collect();
        Self {
            rows: 0,
            types,
            hash: 0,
        }
    }

    fn add(&mut self, batch: &RecordBatch) {
        let state = RandomState::with_seed(0);
        let sum = with_hashes(batch.columns(), &state, |hashes| {
            Ok(hashes.iter().copied().fold(0, u64::wrapping_add))
        });
        self.hash = self.hash.wrapping_add(sum.unwrap());
        self.rows += batch.num_rows();
    }
}

/// Run `sql` once to check it, and to see what its file scan counted: its fingerprint, and the
/// scan's DataFusion metrics, summed by name (all but the timings). The batches are hashed as
/// they come, not kept.
async fn check(ctx: &SessionContext, sql: &str) -> (Fingerprint, Vec<String>) {
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let mut fingerprint = Fingerprint::new(&plan.schema());
    let mut stream = execute_stream(Arc::clone(&plan), ctx.task_ctx()).unwrap();
    while let Some(batch) = stream.next().await {
        fingerprint.add(&batch.unwrap());
    }
    (fingerprint, scan_metrics(&plan))
}

fn scan_metrics(plan: &Arc<dyn ExecutionPlan>) -> Vec<String> {
    let mut out = Vec::new();
    if plan.name() == "DataSourceExec" {
        if let Some(metrics) = plan.metrics() {
            for metric in metrics.aggregate_by_name().sorted_for_display().iter() {
                let value = metric.value();
                let name = value.name();
                if !(name.contains("time") || name.contains("elapsed")) {
                    out.push(format!("{name}={value}"));
                }
            }
        }
        // The rows of each Vortex file a selection keeps (see `ZonePruner`).
        let files = plan
            .downcast_ref::<DataSourceExec>()
            .and_then(|exec| exec.data_source().downcast_ref::<FileScanConfig>())
            .map(|conf| {
                conf.file_groups
                    .iter()
                    .flat_map(|g| g.iter().cloned())
                    .collect()
            })
            .unwrap_or_else(Vec::new);
        for file in files {
            let selected = match file
                .extensions
                .get::<VortexAccessPlan>()
                .and_then(VortexAccessPlan::selection)
            {
                Some(ScanSelection::IncludeRoaring(rows)) => rows.len().to_string(),
                Some(_) => "?".to_string(),
                None => "all".to_string(),
            };
            out.push(format!("selected_rows={selected}"));
        }
    }
    for child in plan.children() {
        out.extend(scan_metrics(child));
    }
    out
}

/// `needle` as a query for every column of the rows it matches.
fn needle_sql(needle: &Needle, schema: &Schema) -> String {
    let terms: Vec<String> = needle
        .terms
        .iter()
        .map(|term| {
            let op = match term.op {
                Op::Eq => "=",
                Op::Gt => ">",
                Op::Ge => ">=",
                Op::Lt => "<",
            };
            let field = schema.field_with_name(term.column).unwrap();
            let literal = sql_literal(term.value, field.data_type());
            format!("{} {op} {literal}", term.column)
        })
        .collect();
    format!("SELECT * FROM {TABLE} WHERE {}", terms.join(" AND "))
}

/// `value` as a SQL literal of `data_type`, the type of the column it's compared with.
fn sql_literal(value: Value, data_type: &DataType) -> String {
    match value {
        Value::Str(s) => format!("'{s}'"),
        Value::TimestampNs(ns) => format!("arrow_cast({ns}, '{data_type}')"),
        Value::Decimal { unscaled, scale } => {
            let DataType::Decimal128(precision, column_scale) = data_type else {
                panic!("{data_type} is not a Decimal128");
            };
            let text = if scale <= 0 {
                (unscaled * 10i128.pow(scale.unsigned_abs() as u32)).to_string()
            } else {
                let factor = 10i128.pow(scale as u32);
                let sign = if unscaled < 0 { "-" } else { "" };
                let (whole, fraction) = (unscaled.abs() / factor, unscaled.abs() % factor);
                format!("{sign}{whole}.{fraction:0width$}", width = scale as usize)
            };
            format!("CAST('{text}' AS DECIMAL({precision}, {column_scale}))")
        }
    }
}

/// What the benches run, and how: each reader that runs, at each number of partitions.
struct Setup {
    /// The runtime for each number of `partitions`: as many threads as partitions.
    runtimes: [Runtime; 2],
    tables: Tables,
    partitions: [usize; 2],
    selection: Selection,
}

impl Setup {
    fn new() -> Self {
        let cores = std::thread::available_parallelism().unwrap().get();
        let partitions = [1, cores];
        let runtimes = partitions.map(|threads| {
            let mut builder = match threads {
                // Vortex spreads even one partition over every worker (and a one-worker runtime
                // still uses two threads), so one partition runs on the bench's own thread.
                1 => tokio::runtime::Builder::new_current_thread(),
                _ => tokio::runtime::Builder::new_multi_thread(),
            };
            builder
                .worker_threads(threads)
                .enable_all()
                .build()
                .unwrap()
        });
        let tables = Tables::new(&runtimes[1]);
        Self {
            runtimes,
            tables,
            partitions,
            selection: Selection::from_args(),
        }
    }

    /// The runtime to run `partitions` partitions on.
    fn runtime(&self, partitions: usize) -> &Runtime {
        let index = self.partitions.iter().position(|&p| p == partitions);
        &self.runtimes[index.expect("a number of partitions the benches run")]
    }

    /// Check every reader of `sql` that runs (under `id/<reader>/<partitions>`) against Parquet
    /// with DataFusion's defaults, print what each counted, and time them as one group; `label`
    /// names the query in what's printed.
    fn bench(&self, c: &mut Criterion, id: &str, label: &str, sql: &str) {
        let runs: Vec<(Reader, usize)> = Reader::ALL
            .into_iter()
            .flat_map(|reader| self.partitions.map(|partitions| (reader, partitions)))
            .filter(|(reader, partitions)| {
                if !self.tables.runs(*reader) {
                    return false;
                }
                let name = self.tables.reader_name(*reader);
                self.selection.runs(&format!("{id}/{name}/{partitions}"))
            })
            .collect();
        if runs.is_empty() {
            return;
        }
        let rt = self.runtime(1);
        let baseline = self.tables.context(rt, Reader::Parquet, 1);
        let (expected, metrics) = rt.block_on(check(&baseline, sql));
        println!("{label}: {} rows; {sql}", expected.rows);
        println!("  parquet/1: {}", metrics.join(" "));
        let mut contexts = Vec::new();
        for (reader, partitions) in runs {
            let name = format!("{}/{partitions}", self.tables.reader_name(reader));
            let rt = self.runtime(partitions);
            let ctx = self.tables.context(rt, reader, partitions);
            let (actual, metrics) = rt.block_on(check(&ctx, sql));
            assert_eq!(actual, expected, "{label}: {name}");
            if (reader, partitions) != (Reader::Parquet, 1) {
                println!("  {name}: {}", metrics.join(" "));
            }
            contexts.push((name, ctx, rt));
        }

        // Few, flat samples in short windows: criterion's 100 samples took 25 s on a 250 ms
        // needle. Changes are kept at 14% or more, so a rough median does.
        let mut group = c.benchmark_group(id);
        group
            .throughput(Throughput::Elements(expected.rows as u64))
            .sampling_mode(SamplingMode::Flat)
            .sample_size(20)
            .warm_up_time(Duration::from_millis(500))
            .measurement_time(Duration::from_secs(2));
        for (name, ctx, rt) in contexts {
            group.bench_function(name, |b| b.iter(|| rt.block_on(run(&ctx, sql))));
        }
        group.finish();
    }
}

fn df_needles(c: &mut Criterion) {
    let setup = Setup::new();
    for needle in &needles() {
        let sql = needle_sql(needle, &setup.tables.schema);
        let id = format!("df needle/{}", needle.name);
        setup.bench(c, &id, needle.name, &sql);
    }
}

fn df_scans(c: &mut Criterion) {
    let setup = Setup::new();
    let decimals: Vec<&str> = setup
        .tables
        .schema
        .fields()
        .iter()
        .filter(|f| matches!(f.data_type(), DataType::Decimal128(..)))
        .map(|f| f.name().as_str())
        .collect();
    let projections = [
        ("all columns", "*".to_string()),
        ("decimal columns", decimals.join(", ")),
        ("dollar_volume", "dollar_volume".to_string()),
    ];
    for (label, columns) in projections {
        let sql = format!("SELECT {columns} FROM {TABLE}");
        setup.bench(c, &format!("df scan/{label}"), label, &sql);
    }
}

criterion_group! {
    name = benches;
    // The logs are what an A/B compares (`site/ab_summary.py`), so no plots.
    config = Criterion::default().without_plots();
    targets = df_needles, df_scans
}
criterion_main!(benches);
