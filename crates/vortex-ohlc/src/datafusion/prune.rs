//! Pruning a Vortex scan's zones before it starts, as DataFusion does a Parquet file's row groups
//! from its footer.
//!
//! Vortex prunes a scan split by split as the scan runs: every one of a file's zones becomes a
//! task that asks the zone maps whether the filter can match there. Each task costs a few
//! microseconds, and the tasks are too short for the runtime to spread over its threads, so for a
//! selective filter that walk is most of the query: about a millisecond for the 386 zones of a
//! month of bars. A scan given a [`Selection`] doesn't start the splits it rules out at all.
//!
//! [`ZonePruner`] gives each file of a plan's Vortex scans the selection of the zones a filter
//! can match. It keeps each file's zone maps, read once, as DataFusion [`PruningStatistics`] over
//! the table's columns, and prunes them with DataFusion's own
//! [`PruningPredicate`](datafusion::physical_optimizer::pruning::PruningPredicate), a zone to a
//! container:
//!
//! - a column the file stores as it is has its zone maps' min and max, in the table's type;
//! - a rescaled decimal has them scaled back up to the table's scale;
//! - a price has bounds from those of the [`OHLC`] parts it is rebuilt from (see
//!   [`price_bounds`]).
//!
//! [`ZonePruner::prune`] reads the zone maps it doesn't have yet and waits for them, so every
//! query is pruned. It reads them through a [`VortexReaderFactory`], the one the scans read their
//! data with, so the reads go wherever the data's do (S3, a disk cache in front of it, or a local
//! file). [`PrunedTable`](super::PrunedTable) calls it when it's scanned.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use arrow::array::cast::AsArray;
use arrow::array::types::Decimal128Type;
use arrow::array::{Array, ArrayRef, BooleanArray, Decimal128Array};
use arrow::compute::{cast, nullif};
use arrow::datatypes::{DataType, Schema};
use datafusion::common::pruning::PruningStatistics;
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result, ScalarValue};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{
    FileGroup, FileScanConfig, FileScanConfigBuilder, FileSource,
};
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::object_store::ObjectMeta;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_optimizer::pruning::PruningPredicateBuilder;
use datafusion::physical_plan::ExecutionPlan;
use futures::future::{BoxFuture, Shared};
use futures::{FutureExt, StreamExt};
use moka::sync::Cache;
use roaring::RoaringTreemap;
use vortex::array::{MaskFuture, VortexSessionExecute};
use vortex::arrow::ArrowSessionExt;
use vortex::error::{vortex_err, VortexResult};
use vortex::expr::root;
use vortex::file::{OpenOptionsSessionExt, VortexFile};
use vortex::io::VortexReadAt;
use vortex::layout::layouts::zoned::Zoned;
use vortex::layout::{LayoutReaderContext, LayoutRef};
use vortex::scan::selection::Selection;
use vortex::session::VortexSession;
use vortex_datafusion::reader::VortexReaderFactory;
use vortex_datafusion::{VortexAccessPlan, VortexSource};

use crate::layout::{find_child, find_zoned};
use crate::plan::{OHLC, OHLC_PARTS, OHLC_PRICES};

/// How many bytes of zone maps a [`ZonePruner`] keeps by default: tens of KB a file, so some
/// thousands of files.
pub const DEFAULT_CACHE_BYTES: u64 = 256 << 20;

/// The largest share of a file's zones a selection keeps: a scan that keeps more than this skips
/// too few splits to make up for checking its selection on the rest.
const MAX_KEPT: f64 = 0.5;

/// How many files' zone maps [`ZonePruner::prune`] reads at once.
const CONCURRENT_READS: usize = 16;

/// A read of a file's zone maps, which every query that wants them joins; `None` if it failed.
type ZonesRead = Shared<BoxFuture<'static, Option<Arc<FileZones>>>>;

/// Gives each file of a plan's Vortex scans the zones a filter can match: see the module docs.
///
/// The session needs the crate registered (see [`crate::register`]).
pub struct ZonePruner {
    session: VortexSession,
    /// Zone maps by object store and path, up to a number of bytes, the least recently used
    /// evicted first.
    files: Cache<String, Arc<FileZones>>,
    /// The reads under way, by object store and path; shared with the reads, which remove
    /// themselves once done.
    reading: Arc<Mutex<HashMap<String, ZonesRead>>>,
}

impl ZonePruner {
    /// Keeping up to [`DEFAULT_CACHE_BYTES`] of zone maps.
    pub fn new(session: VortexSession) -> Self {
        Self::with_cache_bytes(session, DEFAULT_CACHE_BYTES)
    }

    /// Keeping up to `cache_bytes` of zone maps.
    pub fn with_cache_bytes(session: VortexSession, cache_bytes: u64) -> Self {
        let files = Cache::builder()
            .weigher(|_, zones: &Arc<FileZones>| zones.bytes)
            .max_capacity(cache_bytes)
            .build();
        Self {
            session,
            files,
            reading: Arc::default(),
        }
    }

    /// Give the Vortex scans in `plan` the zones `predicate` can match, or, without one, those
    /// each scan's own pushed-down predicate can. The zone maps of every file it doesn't have
    /// are read first, through `readers`.
    pub async fn prune(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        predicate: Option<&Arc<dyn PhysicalExpr>>,
        readers: &dyn VortexReaderFactory,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut reads = Vec::new();
        plan.apply(|node| {
            if let Some((conf, _)) = vortex_scan(node) {
                for file in conf.file_groups.iter().flat_map(|group| group.iter()) {
                    let url = &conf.object_store_url;
                    if self.cached(url, file).is_none() {
                        let reader = readers.create_reader(file, &self.session)?;
                        reads.push(self.read(url, file, reader));
                    }
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        futures::stream::iter(reads)
            .buffer_unordered(CONCURRENT_READS)
            .collect::<Vec<_>>()
            .await;
        plan.transform_up(|plan| {
            let Some((conf, source)) = vortex_scan(&plan) else {
                return Ok(Transformed::no(plan));
            };
            let Some(predicate) = predicate.or(source.predicate()) else {
                return Ok(Transformed::no(plan));
            };
            let (url, table) = (&conf.object_store_url, source.table_schema().table_schema());
            let mut pruned = false;
            let groups = conf
                .file_groups
                .iter()
                .map(|group| {
                    let files = group.iter().map(|file| {
                        let mut file = file.clone();
                        // A file whose zone maps couldn't be read is scanned whole.
                        let Some(zones) = self.cached(url, &file) else {
                            return file;
                        };
                        if let Some(selection) = selection(&zones, predicate, table) {
                            file.extensions
                                .insert(VortexAccessPlan::default().with_selection(selection));
                            pruned = true;
                        }
                        file
                    });
                    FileGroup::new(files.collect())
                })
                .collect();
            if !pruned {
                return Ok(Transformed::no(plan));
            }
            let conf = FileScanConfigBuilder::from(conf.clone())
                .with_file_groups(groups)
                .build();
            Ok(Transformed::yes(
                DataSourceExec::from_data_source(conf) as Arc<dyn ExecutionPlan>
            ))
        })
        .data()
    }

    /// The zone maps of `file` in the object store at `url`, if they're kept and the file
    /// hasn't changed since they were read.
    fn cached(&self, url: &ObjectStoreUrl, file: &PartitionedFile) -> Option<Arc<FileZones>> {
        let zones = self.files.get(&key(url, file))?;
        let meta = &file.object_meta;
        (zones.meta.size == meta.size && zones.meta.last_modified == meta.last_modified)
            .then_some(zones)
    }

    /// A read of the zone maps of `file` in the object store at `url` through `reader`, or the
    /// one under way, which another query started. It keeps them once done.
    fn read(
        &self,
        url: &ObjectStoreUrl,
        file: &PartitionedFile,
        reader: Arc<dyn VortexReadAt>,
    ) -> ZonesRead {
        let key = key(url, file);
        let mut reading = self.reading.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(read) = reading.get(&key) {
            return read.clone();
        }
        let (session, meta) = (self.session.clone(), file.object_meta.clone());
        let (files, reads, done) = (self.files.clone(), Arc::clone(&self.reading), key.clone());
        let read = async move {
            let zones = async {
                let file = session
                    .open_options()
                    .with_file_size(meta.size)
                    .open_read(reader)
                    .await?;
                FileZones::read(&session, &file, meta).await
            };
            // On an error, left unpruned until a later query tries again.
            let zones = zones.await.ok().map(Arc::new);
            if let Some(zones) = &zones {
                files.insert(done.clone(), Arc::clone(zones));
            }
            reads
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&done);
            zones
        }
        .boxed()
        .shared();
        reading.insert(key, read.clone());
        read
    }
}

impl fmt::Debug for ZonePruner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ZonePruner").finish_non_exhaustive()
    }
}

/// The cache key of `file` in the object store at `url`.
fn key(url: &ObjectStoreUrl, file: &PartitionedFile) -> String {
    format!("{}{}", url.as_str(), file.path())
}

/// `plan`'s file scan config and source, if it is a Vortex scan.
fn vortex_scan(plan: &Arc<dyn ExecutionPlan>) -> Option<(&FileScanConfig, &VortexSource)> {
    let conf = plan
        .downcast_ref::<DataSourceExec>()?
        .data_source()
        .downcast_ref::<FileScanConfig>()?;
    let source = conf.file_source().downcast_ref::<VortexSource>()?;
    Some((conf, source))
}

/// The rows in the zones of a file with the zone maps `zones` that `predicate` (over `table`,
/// the table's schema) can match; `None` to scan it whole.
fn selection(
    zones: &FileZones,
    predicate: &Arc<dyn PhysicalExpr>,
    table: &Arc<Schema>,
) -> Option<Selection> {
    let pruning = PruningPredicateBuilder::new()
        .with_file_schema(Arc::clone(table))
        .build(Arc::clone(predicate))?;
    let stats = ZoneStatistics { zones, table };
    let keep = pruning.prune(&stats).ok()?;
    let kept = keep.iter().filter(|&&keep| keep).count();
    if kept as f64 > keep.len() as f64 * MAX_KEPT {
        return None;
    }
    let mut selected = RoaringTreemap::new();
    for (zone, _) in keep.iter().enumerate().filter(|(_, &keep)| keep) {
        let start = zone as u64 * zones.zone_len;
        selected.insert_range(start..(start + zones.zone_len).min(zones.row_count));
    }
    Some(Selection::IncludeRoaring(selected))
}

/// A file's zone maps, as of its size and modification time (in `meta`): every zoned column's
/// min and max per zone, as Arrow arrays of its stored type, null where the zone map has no
/// bound. A column is named by its path, `ohlc.open` for a part.
struct FileZones {
    meta: ObjectMeta,
    /// What the min and max arrays take in memory, as the cache weighs them.
    bytes: u32,
    zone_len: u64,
    row_count: u64,
    nzones: usize,
    columns: HashMap<String, (ArrayRef, ArrayRef)>,
}

impl FileZones {
    /// Every zoned column's zone map in `file`; an error unless they all have the same zones.
    async fn read(
        session: &VortexSession,
        file: &VortexFile,
        meta: ObjectMeta,
    ) -> VortexResult<Self> {
        let root_layout = file.footer().layout();
        let fields = file
            .dtype()
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("not a table"))?;
        // Each zoned column's zoned layout, by path.
        let mut zoned: Vec<(String, LayoutRef)> = Vec::new();
        for (name, dtype) in fields.names().iter().zip(fields.fields()) {
            let Some(column) = find_child(root_layout, name.as_ref()) else {
                continue;
            };
            match dtype.as_struct_fields_opt() {
                Some(parts) => {
                    for part in parts.names().iter() {
                        let layout =
                            find_child(&column, part.as_ref()).and_then(|l| find_zoned(&l));
                        if let Some(layout) = layout {
                            zoned.push((format!("{name}.{part}"), layout));
                        }
                    }
                }
                None => {
                    if let Some(layout) = find_zoned(&column) {
                        zoned.push((name.to_string(), layout));
                    }
                }
            }
        }
        let mut shape = None;
        let mut columns = HashMap::new();
        for (path, layout) in zoned {
            let typed = layout
                .as_opt::<Zoned>()
                .ok_or_else(|| vortex_err!("{path} is not zoned"))?;
            let this = (typed.zone_len() as u64, typed.nzones());
            if *shape.get_or_insert(this) != this {
                return Err(vortex_err!("{path} has zones of its own"));
            }
            let zones = Arc::clone(&layout.children()?[1]);
            let bounds = read_zone_map(session, file, &path, &zones).await?;
            columns.insert(path, bounds);
        }
        let (zone_len, nzones) = shape.ok_or_else(|| vortex_err!("no zoned columns"))?;
        let bytes = columns
            .values()
            .map(|(min, max)| min.get_array_memory_size() + max.get_array_memory_size())
            .sum::<usize>();
        Ok(Self {
            meta,
            bytes: u32::try_from(bytes).unwrap_or(u32::MAX),
            zone_len,
            row_count: file.row_count(),
            nzones,
            columns,
        })
    }
}

/// The min and max per zone in the zone map `zones` of the column at `path`, as Arrow arrays;
/// null where there is no bound.
async fn read_zone_map(
    session: &VortexSession,
    file: &VortexFile,
    path: &str,
    zones: &LayoutRef,
) -> VortexResult<(ArrayRef, ArrayRef)> {
    let n = zones.row_count();
    let reader = zones.new_reader(
        format!("{path}.zones").into(),
        file.segment_source(),
        session,
        &LayoutReaderContext::new(),
    )?;
    let all = root().bind(zones.dtype())?;
    let table = reader
        .projection_evaluation(&(0..n), &all, MaskFuture::new_true(n as usize))?
        .await?;
    let mut ctx = session.create_execution_ctx();
    let table = session.arrow().execute_arrow(table, None, &mut ctx)?;
    let table = table.as_struct();
    let named = |prefix: &str| {
        table
            .column_names()
            .iter()
            .position(|name| name.starts_with(prefix))
            .map(|i| table.column(i))
    };
    // `vortex.min()`, or a bound below the values for a string (`vortex.bounded_min(..)`).
    let min = named("vortex.min()")
        .or_else(|| named("vortex.bounded_min("))
        .ok_or_else(|| vortex_err!("{path} has no min"))?;
    let max = match named("vortex.max()") {
        Some(max) => Arc::clone(max),
        // A bound above the values, unknown where they were too long to bound.
        None => {
            let bounded = named("vortex.bounded_max(")
                .ok_or_else(|| vortex_err!("{path} has no max"))?
                .as_struct();
            let bound = bounded
                .column_by_name("bound")
                .ok_or_else(|| vortex_err!("{path} has no max bound"))?;
            let unknown: BooleanArray = (0..bounded.len())
                .map(|i| {
                    let unknown = bounded
                        .column_by_name("unknown")
                        .is_some_and(|unknown| unknown.as_boolean().value(i));
                    Some(bounded.is_null(i) || unknown)
                })
                .collect();
            nullif(bound, &unknown)?
        }
    };
    Ok((Arc::clone(min), max))
}

/// A file's zone maps as DataFusion's statistics over the table's columns, a zone a container.
struct ZoneStatistics<'a> {
    zones: &'a FileZones,
    table: &'a Schema,
}

impl ZoneStatistics<'_> {
    /// The zones' min (`max` false) or max of the table's `column`, in the table's type.
    fn bound(&self, column: &Column, max: bool) -> Option<ArrayRef> {
        let field = self.table.field_with_name(&column.name).ok()?;
        if let Some((min_values, max_values)) = self.zones.columns.get(&column.name) {
            let stored = if max { max_values } else { min_values };
            return cast(stored, field.data_type()).ok();
        }
        if OHLC_PRICES.contains(&column.name.as_str()) {
            return price_bounds(self.zones, &column.name, max, field.data_type());
        }
        None
    }
}

impl PruningStatistics for ZoneStatistics<'_> {
    fn min_values(&self, column: &Column) -> Option<ArrayRef> {
        self.bound(column, false)
    }

    fn max_values(&self, column: &Column) -> Option<ArrayRef> {
        self.bound(column, true)
    }

    fn num_containers(&self) -> usize {
        self.zones.nzones
    }

    fn null_counts(&self, _column: &Column) -> Option<ArrayRef> {
        None
    }

    fn row_counts(&self) -> Option<ArrayRef> {
        None
    }

    fn contained(&self, _column: &Column, _values: &HashSet<ScalarValue>) -> Option<BooleanArray> {
        None
    }
}

/// The zones' min (`max` false) or max of `price` as a `Decimal128` of `data_type`, from the
/// bounds of the [`OHLC`] parts it is rebuilt from. Every price grows with open and the body,
/// high with the upper wick, and low shrinks with the lower one:
///
/// - open = open, close = open + body;
/// - high = open + max(body, 0) + upper;
/// - low = open + min(body, 0) - lower.
///
/// So a price's min is at least what its parts' mins give (the lower wick's max, for low), and
/// its max at most what their maxes give. `None` unless the file has the parts and `data_type`
/// is a `Decimal128` at a scale at least theirs.
fn price_bounds(
    zones: &FileZones,
    price: &str,
    max: bool,
    data_type: &DataType,
) -> Option<ArrayRef> {
    let DataType::Decimal128(precision, scale) = *data_type else {
        return None;
    };
    let part = |name: &str, max: bool| -> Option<&Decimal128Array> {
        let (min_values, max_values) = zones.columns.get(&format!("{OHLC}.{name}"))?;
        (if max { max_values } else { min_values }).as_primitive_opt::<Decimal128Type>()
    };
    let [open, body, upper, lower] = OHLC_PARTS;
    let (o, b) = (part(open, max)?, part(body, max)?);
    let shift = scale.checked_sub(o.scale()).filter(|s| *s >= 0)?;
    let factor = 10i128.pow(shift as u32);
    let at = |array: &Decimal128Array, i: usize| array.is_valid(i).then(|| array.value(i));
    let bound: Decimal128Array = match price {
        "open" => (0..o.len()).map(|i| at(o, i)).collect(),
        "close" => (0..o.len()).map(|i| Some(at(o, i)? + at(b, i)?)).collect(),
        "high" => {
            let u = part(upper, max)?;
            (0..o.len())
                .map(|i| Some(at(o, i)? + at(b, i)?.max(0) + at(u, i)?))
                .collect()
        }
        _ => {
            let l = part(lower, !max)?;
            (0..o.len())
                .map(|i| Some(at(o, i)? + at(b, i)?.min(0) - at(l, i)?))
                .collect()
        }
    };
    let bound = bound
        .unary::<_, Decimal128Type>(|v| v * factor)
        .with_precision_and_scale(precision, scale)
        .ok()?;
    Some(Arc::new(bound))
}
