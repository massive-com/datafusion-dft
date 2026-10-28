//! The needles and scans as SQL through DataFusion, the way dft runs them: against a table over
//! the Parquet v2 file (DataFusion's Parquet reader with its defaults, and with
//! `pushdown_filters`), and against one over the Vortex file with the Parquet table's schema,
//! its columns rebuilt and its price filters pushed down into Vortex (see
//! `vortex_ohlc::datafusion`). Each runs on one partition and on one per core.
//!
//! A needle selects every column of the rows it matches; each of its terms compares a column
//! with a literal of the column's own type, so no format's filter is held back by a cast. The
//! scans select every column, the decimal columns, and dollar_volume alone.
//!
//! Criterion's filter picks the benchmarks, ids `df needle/<needle>/<reader>/<partitions>` and
//! `df scan/<projection>/<reader>/<partitions>`, e.g. `cargo bench --bench df -- 'high > 1000/.*/1$'`.
//! Before timing, every reader that runs is checked against Parquet with DataFusion's defaults
//! (the same rows, in any order), and what each scan counted is printed: DataFusion's metrics
//! of the file scan, and for Vortex the zones the bars layout kept and the rows a scan's
//! selection kept. See `common::data` for the files.

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow::row::{RowConverter, SortField};
use criterion::{criterion_group, criterion_main, Criterion, SamplingMode, Throughput};
use datafusion::catalog::TableProvider;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfig, FileScanConfigBuilder};
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::{collect, ExecutionPlan};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use futures::StreamExt;
use roaring::RoaringTreemap;
use tokio::runtime::Runtime;
use vortex::array::memory::MemorySessionExt;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::io::std_file::FileReadAt;
use vortex::io::VortexReadAt;
use vortex::metrics::{DefaultMetricsRegistry, MetricValue, MetricsRegistry};
use vortex::scan::selection::Selection as ScanSelection;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_datafusion::reader::{DefaultVortexReaderFactory, VortexReaderFactory};
use vortex_datafusion::{VortexAccessPlan, VortexSource};
use vortex_ohlc::datafusion::{vortex_format, BarsAdapterFactory, PrunedTable, ZonePruner};
use vortex_ohlc::layout::record_metrics;
use vortex_ohlc::strategy::DEFAULT_ROW_BLOCK;

use common::needle::{needles, vortex_filter, Needle, Op, Value};
use common::vortex_read::{matching_rows, new_session, open_vortex, read_plan};
use common::{data, Selection};

/// The table every query reads.
const TABLE: &str = "bars";

/// How a table reads its file.
#[derive(Clone, Copy, PartialEq)]
enum Reader {
    Parquet,
    /// Parquet with `pushdown_filters`: the filter decodes its columns first, and the others
    /// only for the rows that pass, as the needle bench's Parquet reader does.
    ParquetPushdown,
    /// The file with the bars layout, its scans pruned by a `ZonePruner` before they start.
    Vortex,
    /// The file with the bars layout, pruned by its readers as the scan runs.
    VortexLayout,
    /// The file without the bars layout, its scans pruned by a `ZonePruner`.
    VortexZones,
    /// The file without the bars layout, pruned on Vortex's own zone maps as the scan runs.
    VortexPlain,
    /// The file without the bars layout or the OHLC struct, the prices rescaled on their own,
    /// its scans pruned by a `ZonePruner`.
    VortexPrices,
    /// Vortex with each needle's scan given exactly the zones that hold a match before it
    /// starts (see [`IdealSelection`]): the best that pruning before the scan can do.
    VortexIdeal,
}

impl Reader {
    const ALL: [Reader; 8] = [
        Reader::Parquet,
        Reader::ParquetPushdown,
        Reader::Vortex,
        Reader::VortexLayout,
        Reader::VortexZones,
        Reader::VortexPlain,
        Reader::VortexPrices,
        Reader::VortexIdeal,
    ];

    fn name(self) -> &'static str {
        match self {
            Reader::Parquet => "parquet",
            Reader::ParquetPushdown => "parquet pushdown",
            Reader::Vortex => "vortex",
            Reader::VortexLayout => "vortex layout",
            Reader::VortexZones => "vortex zones",
            Reader::VortexPlain => "vortex plain",
            Reader::VortexPrices => "vortex prices",
            Reader::VortexIdeal => "vortex ideal",
        }
    }

    /// For a Vortex reader, the file it reads and whether a `ZonePruner` prunes its scans.
    fn vortex(self) -> Option<(VortexFile, bool)> {
        Some(match self {
            Reader::Parquet | Reader::ParquetPushdown => return None,
            Reader::Vortex => (VortexFile::Bars, true),
            Reader::VortexLayout | Reader::VortexIdeal => (VortexFile::Bars, false),
            Reader::VortexZones => (VortexFile::Plain, true),
            Reader::VortexPlain => (VortexFile::Plain, false),
            Reader::VortexPrices => (VortexFile::Prices, true),
        })
    }
}

/// The Vortex files (see `common::data`).
#[derive(Clone, Copy)]
enum VortexFile {
    Bars,
    Plain,
    Prices,
}

/// The files, and the Parquet table's schema, which the Vortex table takes too.
struct Tables {
    parquet: String,
    vortex: String,
    vortex_plain: String,
    vortex_prices: String,
    schema: SchemaRef,
}

impl Tables {
    fn new(runtime: &Runtime) -> Self {
        let setup = CurrentThreadRuntime::new();
        let files = data::files(&setup, &new_session(&setup));
        let parquet = files.parquet.to_str().unwrap().to_string();
        let vortex = files.vortex.to_str().unwrap().to_string();
        let vortex_plain = files.vortex_plain.to_str().unwrap().to_string();
        let vortex_prices = files.vortex_prices.to_str().unwrap().to_string();
        let ctx = SessionContext::new();
        runtime
            .block_on(ctx.register_parquet(TABLE, &parquet, ParquetReadOptions::default()))
            .unwrap();
        let schema = Arc::clone(runtime.block_on(ctx.table(TABLE)).unwrap().schema().inner());
        Self {
            parquet,
            vortex,
            vortex_plain,
            vortex_prices,
            schema,
        }
    }

    /// A session on `partitions` partitions with [`TABLE`] read by `reader`; a Vortex table's
    /// scans record what the bars layout pruned in `registry`.
    fn context(
        &self,
        runtime: &Runtime,
        reader: Reader,
        partitions: usize,
        registry: Option<Arc<dyn MetricsRegistry>>,
        ideal: Option<&ScanSelection>,
    ) -> SessionContext {
        let mut config = SessionConfig::new().with_target_partitions(partitions);
        if reader == Reader::ParquetPushdown {
            config.options_mut().execution.parquet.pushdown_filters = true;
        }
        let mut ctx = SessionContext::new_with_config(config);
        match reader {
            Reader::Parquet | Reader::ParquetPushdown => runtime
                .block_on(ctx.register_parquet(TABLE, &self.parquet, ParquetReadOptions::default()))
                .unwrap(),
            _ => {
                let (file, pruned) = reader.vortex().unwrap();
                let path = match file {
                    VortexFile::Bars => &self.vortex,
                    VortexFile::Plain => &self.vortex_plain,
                    VortexFile::Prices => &self.vortex_prices,
                };
                // Vortex does its I/O on the runtime it finds when the session is made.
                let _runtime = runtime.enter();
                let session = VortexSession::default();
                vortex_ohlc::register(&session);
                if let Some(registry) = registry {
                    record_metrics(&session, registry);
                }
                let pruner = Arc::new(ZonePruner::new(session.clone()));
                let options = ListingOptions::new(Arc::new(vortex_format(session)))
                    .with_file_extension(".vortex");
                let config = ListingTableConfig::new(ListingTableUrl::parse(path).unwrap())
                    .with_listing_options(options)
                    .with_schema(Arc::clone(&self.schema))
                    .with_expr_adapter_factory(Arc::new(BarsAdapterFactory));
                let mut table: Arc<dyn TableProvider> =
                    Arc::new(ListingTable::try_new(config).unwrap());
                // Pruned with zone maps read through the object store, as production reads S3.
                if pruned {
                    let url = ObjectStoreUrl::local_filesystem();
                    let store = ctx.runtime_env().object_store(&url).unwrap();
                    let readers = Arc::new(DefaultVortexReaderFactory::new(store));
                    table = Arc::new(PrunedTable::new(table, pruner, readers));
                }
                ctx.register_table(TABLE, table).unwrap();
                let mut state = SessionStateBuilder::new_from_existing(ctx.state())
                    .with_physical_optimizer_rule(Arc::new(LocalReaders));
                if let (Reader::VortexIdeal, Some(ideal)) = (reader, ideal) {
                    state =
                        state.with_physical_optimizer_rule(Arc::new(IdealSelection(ideal.clone())));
                }
                let state = state.build();
                ctx = SessionContext::new_with_state(state);
            }
        }
        ctx
    }
}

/// Gives Vortex scans a reader that keeps the file open, as dft's `VortexReaders` rule does for
/// a local file: vortex-datafusion otherwise reads it through the `ObjectStore`, which opens the
/// file again for every read.
#[derive(Debug)]
struct LocalReaders;

impl PhysicalOptimizerRule for LocalReaders {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|plan| {
            let Some(conf) = plan
                .downcast_ref::<DataSourceExec>()
                .and_then(|exec| exec.data_source().downcast_ref::<FileScanConfig>())
            else {
                return Ok(Transformed::no(plan));
            };
            let Some(source) = conf.file_source().downcast_ref::<VortexSource>() else {
                return Ok(Transformed::no(plan));
            };
            let source = source
                .clone()
                .with_vortex_reader_factory(Arc::new(LocalReaderFactory));
            let conf = FileScanConfigBuilder::from(conf.clone())
                .with_source(Arc::new(source))
                .build();
            Ok(Transformed::yes(
                DataSourceExec::from_data_source(conf) as Arc<dyn ExecutionPlan>
            ))
        })
        .data()
    }

    fn name(&self) -> &str {
        "local_readers"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[derive(Debug)]
struct LocalReaderFactory;

impl VortexReaderFactory for LocalReaderFactory {
    fn create_reader(
        &self,
        file: &PartitionedFile,
        session: &VortexSession,
    ) -> datafusion::common::Result<Arc<dyn VortexReadAt>> {
        let path = format!("/{}", file.path());
        let file = FileReadAt::open_with_allocator(path, session.handle(), session.allocator())
            .map_err(|e| datafusion::common::DataFusionError::External(Box::new(e)))?;
        Ok(Arc::new(file))
    }
}

/// Gives every file of a Vortex scan the selection it holds, as a rule that prunes before the
/// scan would.
#[derive(Debug)]
struct IdealSelection(ScanSelection);

impl PhysicalOptimizerRule for IdealSelection {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|plan| {
            let Some(conf) = plan
                .downcast_ref::<DataSourceExec>()
                .and_then(|exec| exec.data_source().downcast_ref::<FileScanConfig>())
            else {
                return Ok(Transformed::no(plan));
            };
            if conf.file_source().downcast_ref::<VortexSource>().is_none() {
                return Ok(Transformed::no(plan));
            }
            let groups = conf
                .file_groups
                .iter()
                .map(|group| {
                    let files = group.iter().map(|file| {
                        let mut file = file.clone();
                        let plan = VortexAccessPlan::default().with_selection(self.0.clone());
                        file.extensions.insert(plan);
                        file
                    });
                    FileGroup::new(files.collect())
                })
                .collect();
            let conf = FileScanConfigBuilder::from(conf.clone())
                .with_file_groups(groups)
                .build();
            Ok(Transformed::yes(
                DataSourceExec::from_data_source(conf) as Arc<dyn ExecutionPlan>
            ))
        })
        .data()
    }

    fn name(&self) -> &str {
        "ideal_selection"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// The rows of the zones of the Vortex file at `path` that hold a match for `needle`, found by
/// a scan; `None` if the needle doesn't go into a Vortex filter whole.
fn ideal_selection(path: &std::path::Path, needle: &Needle) -> Option<ScanSelection> {
    let runtime = CurrentThreadRuntime::new();
    let session = new_session(&runtime);
    let file = open_vortex(&runtime, &session, path);
    let plan = read_plan(&runtime, &session, path);
    let (Some(filter), after) = vortex_filter(&file, needle, &plan) else {
        return None;
    };
    if !after.is_empty() {
        return None;
    }
    let zone = DEFAULT_ROW_BLOCK as u64;
    let rows = file.row_count();
    let mut zones: Vec<u64> = matching_rows(&runtime, &session, &file, &filter)
        .into_iter()
        .map(|row| row / zone)
        .collect();
    zones.dedup();
    let mut selected = RoaringTreemap::new();
    for z in zones {
        selected.insert_range(z * zone..((z + 1) * zone).min(rows));
    }
    Some(ScanSelection::IncludeRoaring(selected))
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

/// The counters in `registry`, summed by name.
fn counters(registry: &DefaultMetricsRegistry) -> BTreeMap<String, u64> {
    let mut counters: BTreeMap<String, u64> = BTreeMap::new();
    for metric in registry.snapshot() {
        if let MetricValue::Counter(counter) = metric.value() {
            *counters.entry(metric.name().to_string()).or_default() += counter.value();
        }
    }
    counters
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
    fn bench(
        &self,
        c: &mut Criterion,
        id: &str,
        label: &str,
        sql: &str,
        scan: bool,
        ideal: Option<&ScanSelection>,
    ) {
        let runs: Vec<(Reader, usize)> = Reader::ALL
            .into_iter()
            .flat_map(|reader| self.partitions.map(|partitions| (reader, partitions)))
            .filter(|(reader, _)| *reader != Reader::VortexIdeal || ideal.is_some())
            .filter(|(reader, partitions)| {
                self.selection
                    .runs(&format!("{id}/{}/{partitions}", reader.name()))
            })
            .collect();
        if runs.is_empty() {
            return;
        }
        let rt = &self.runtime;
        let baseline = self.tables.context(rt, Reader::Parquet, 1, None, None);
        let (expected, metrics) = rt.block_on(check(&baseline, sql));
        println!("{label}: {} rows; {sql}", expected.rows);
        println!("  parquet/1: {}", metrics.join(" "));
        for &(reader, partitions) in &runs {
            let registry = Arc::new(DefaultMetricsRegistry::default());
            let as_registry = Arc::clone(&registry) as Arc<dyn MetricsRegistry>;
            let ctx = self
                .tables
                .context(rt, reader, partitions, Some(as_registry), ideal);
            let name = format!("{}/{partitions}", reader.name());
            // Three times, a moment apart: the first reads the zone maps (see `ZonePruner`), and
            // the last runs as the timed ones do, with them cached.
            let mut before = BTreeMap::new();
            let mut metrics = Vec::new();
            let mut first = Vec::new();
            for run in 0..3 {
                before = counters(&registry);
                let (actual, last) = rt.block_on(check(&ctx, sql));
                assert_eq!(actual, expected, "{label}: {name}");
                if run == 0 {
                    first = last
                        .iter()
                        .filter_map(|m| m.strip_prefix("selected_rows="))
                        .map(|rows| format!("first_selected_rows={rows}"))
                        .collect();
                }
                metrics = last;
                std::thread::sleep(Duration::from_millis(100));
            }
            metrics.extend(first);
            if reader.vortex().is_some() {
                let after = counters(&registry);
                metrics.extend(
                    after
                        .iter()
                        .map(|(k, v)| format!("{k}={}", v - before.get(k).copied().unwrap_or(0))),
                );
            }
            if reader != Reader::Parquet || partitions != 1 {
                println!("  {name}: {}", metrics.join(" "));
            }
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
        for (reader, partitions) in runs {
            let ctx = self.tables.context(rt, reader, partitions, None, ideal);
            group.bench_function(format!("{}/{partitions}", reader.name()), |b| {
                b.iter(|| rt.block_on(run(&ctx, sql)))
            });
        }
        group.finish();
    }
}

fn df_needles(c: &mut Criterion) {
    let setup = Setup::new();
    for needle in &needles() {
        let sql = needle_sql(needle, &setup.tables.schema);
        let id = format!("df needle/{}", needle.name);
        let ideal = setup
            .partitions
            .iter()
            .any(|p| setup.selection.runs(&format!("{id}/vortex ideal/{p}")))
            .then(|| ideal_selection(std::path::Path::new(&setup.tables.vortex), needle))
            .flatten();
        setup.bench(c, &id, needle.name, &sql, false, ideal.as_ref());
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
        setup.bench(c, &format!("df scan/{label}"), label, &sql, true, None);
    }
}

criterion_group!(benches, df_needles, df_scans);
criterion_main!(benches);
