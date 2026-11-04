//! The needles and scans as SQL through DataFusion, the way dft runs them: against a table over
//! the Parquet v2 file (DataFusion's Parquet reader with its defaults, and with
//! `pushdown_filters`), and against one over the Vortex file with the Parquet table's schema,
//! its columns rebuilt and its price filters pushed down into Vortex (see
//! `datafusion_vortex_ohlc`), with and without a `ZonePruner`. Each runs on one partition and on
//! one per core.
//!
//! A needle selects every column of the rows it matches; each of its terms compares a column
//! with a literal of the column's own type, so no format's filter is held back by a cast. The
//! scans select every column, the decimal columns, and dollar_volume alone.
//!
//! Criterion's filter picks the benchmarks, ids `df needle/<needle>/<reader>/<partitions>` and
//! `df scan/<projection>/<reader>/<partitions>`, e.g. `cargo bench --bench df -- 'high > 1000/.*/1$'`.
//! Before timing, every reader that runs is checked against Parquet with DataFusion's defaults
//! (the same rows, in any order), and what each scan counted is printed: DataFusion's metrics
//! of the file scan, and for Vortex the rows a scan's selection kept. See `common::data` for the
//! files.
//!
//! For quicker iterations, `benches/quick.filter` holds a filter for one needle or scan of each
//! kind of work, every Vortex reader at one partition per core:
//! `cargo bench --bench df -- "$(cat crates/datafusion-vortex-ohlc/benches/quick.filter)"`. Run
//! them all before relying on a result. Parquet's results don't change with this crate: they are
//! kept in `target/vortex-ohlc-bench/parquet-results.txt`, to compare against without timing
//! Parquet again.

mod common;

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow::row::{RowConverter, SortField};
use criterion::{criterion_group, criterion_main, Criterion, SamplingMode, Throughput};
use datafusion::datasource::file_format::FileFormatFactory;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::datasource::physical_plan::FileScanConfig;
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::physical_plan::{collect, ExecutionPlan};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_vortex_ohlc::{vortex_format_factory, BarsAdapterFactory};
use datafusion_vortex_prune::{reader_factory, PrunedTable, VortexExecFactory, ZonePruner};
use futures::StreamExt;
use tokio::runtime::Runtime;
use vortex::scan::selection::Selection as ScanSelection;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_datafusion::VortexAccessPlan;

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
}

impl Reader {
    const ALL: [Reader; 4] = [
        Reader::Parquet,
        Reader::ParquetPushdown,
        Reader::Vortex,
        Reader::VortexUnpruned,
    ];

    fn name(self) -> &'static str {
        match self {
            Reader::Parquet => "parquet",
            Reader::ParquetPushdown => "parquet pushdown",
            Reader::Vortex => "vortex",
            Reader::VortexUnpruned => "vortex unpruned",
        }
    }
}

/// The files, and the Parquet table's schema, which the Vortex table takes too.
struct Tables {
    parquet: String,
    vortex: String,
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
            schema,
        }
    }

    /// A session on `partitions` partitions with [`TABLE`] read by `reader`.
    fn context(&self, runtime: &Runtime, reader: Reader, partitions: usize) -> SessionContext {
        let mut config = SessionConfig::new().with_target_partitions(partitions);
        if reader == Reader::ParquetPushdown {
            config.options_mut().execution.parquet.pushdown_filters = true;
        }
        let ctx = SessionContext::new_with_config(config);
        if matches!(reader, Reader::Parquet | Reader::ParquetPushdown) {
            let options = ParquetReadOptions::default();
            runtime
                .block_on(ctx.register_parquet(TABLE, &self.parquet, options))
                .unwrap();
            return ctx;
        }
        // Vortex does its I/O on the runtime it finds when the session is made.
        let _runtime = runtime.enter();
        let session = VortexSession::default();
        let options = ListingOptions::new(vortex_format_factory(session.clone()).default())
            .with_file_extension(".vortex");
        let config = ListingTableConfig::new(ListingTableUrl::parse(&self.vortex).unwrap())
            .with_listing_options(options)
            .with_schema(Arc::clone(&self.schema))
            .with_expr_adapter_factory(Arc::new(BarsAdapterFactory));
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
/// a sum of each row's hash.
#[derive(Debug, PartialEq)]
struct Fingerprint {
    rows: usize,
    types: Vec<DataType>,
    hash: u64,
}

fn fingerprint(schema: &Schema, batches: &[RecordBatch]) -> Fingerprint {
    let types: Vec<DataType> = schema
        .fields()
        .iter()
        .map(|f| f.data_type().clone())
        .collect();
    let fields = types.iter().cloned().map(SortField::new).collect();
    let converter = RowConverter::new(fields).unwrap();
    let mut hash = 0u64;
    let mut rows = 0;
    for batch in batches {
        let converted = converter.convert_columns(batch.columns()).unwrap();
        for row in converted.iter() {
            let mut hasher = DefaultHasher::new();
            row.as_ref().hash(&mut hasher);
            hash = hash.wrapping_add(hasher.finish());
        }
        rows += batch.num_rows();
    }
    Fingerprint { rows, types, hash }
}

/// Run `sql` once to check it, and to see what its file scan counted: its fingerprint, and the
/// scan's DataFusion metrics, summed by name (all but the timings).
async fn check(ctx: &SessionContext, sql: &str) -> (Fingerprint, Vec<String>) {
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
    (fingerprint(&plan.schema(), &batches), scan_metrics(&plan))
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
    runtime: Runtime,
    tables: Tables,
    partitions: [usize; 2],
    selection: Selection,
}

impl Setup {
    fn new() -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let tables = Tables::new(&runtime);
        let cores = std::thread::available_parallelism().unwrap().get();
        Self {
            runtime,
            tables,
            partitions: [1, cores],
            selection: Selection::from_args(),
        }
    }

    /// Check every reader of `sql` that runs (under `id/<reader>/<partitions>`) against Parquet
    /// with DataFusion's defaults, print what each counted, and time them as one group; `label`
    /// names the query in what's printed.
    fn bench(&self, c: &mut Criterion, id: &str, label: &str, sql: &str, scan: bool) {
        let runs: Vec<(Reader, usize)> = Reader::ALL
            .into_iter()
            .flat_map(|reader| self.partitions.map(|partitions| (reader, partitions)))
            .filter(|(reader, partitions)| {
                self.selection
                    .runs(&format!("{id}/{}/{partitions}", reader.name()))
            })
            .collect();
        if runs.is_empty() {
            return;
        }
        let rt = &self.runtime;
        let baseline = self.tables.context(rt, Reader::Parquet, 1);
        let (expected, metrics) = rt.block_on(check(&baseline, sql));
        println!("{label}: {} rows; {sql}", expected.rows);
        println!("  parquet/1: {}", metrics.join(" "));
        let mut contexts = Vec::new();
        for (reader, partitions) in runs {
            let name = format!("{}/{partitions}", reader.name());
            let ctx = self.tables.context(rt, reader, partitions);
            let (actual, metrics) = rt.block_on(check(&ctx, sql));
            assert_eq!(actual, expected, "{label}: {name}");
            if (reader, partitions) != (Reader::Parquet, 1) {
                println!("  {name}: {}", metrics.join(" "));
            }
            contexts.push((name, ctx));
        }

        let mut group = c.benchmark_group(id);
        group.throughput(Throughput::Elements(expected.rows as u64));
        if scan {
            group.sampling_mode(SamplingMode::Flat).sample_size(10);
        } else {
            group
                .warm_up_time(Duration::from_secs(1))
                .measurement_time(Duration::from_secs(3));
        }
        for (name, ctx) in contexts {
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
        setup.bench(c, &id, needle.name, &sql, false);
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
        setup.bench(c, &format!("df scan/{label}"), label, &sql, true);
    }
}

criterion_group!(benches, df_needles, df_scans);
criterion_main!(benches);
