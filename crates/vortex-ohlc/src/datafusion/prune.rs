//! Pruning a Vortex scan's zones while the query is planned, as DataFusion does a Parquet file's
//! row groups from its footer.
//!
//! Vortex prunes a scan split by split as the scan runs: every one of a file's zones becomes a
//! task that asks the zone maps whether the filter can match there. Each task costs a few
//! microseconds, and the tasks are too short for the runtime to spread over its threads, so for a
//! selective filter that walk is most of the query: about a millisecond for the 386 zones of a
//! month of bars. A scan given a [`Selection`] doesn't start the splits it rules out at all.
//!
//! [`BarsPruning`] is a physical optimizer rule that gives each file of a Vortex scan the
//! selection of the zones its filter can match. It keeps each file's zone maps, read once, as
//! DataFusion [`PruningStatistics`] over the table's columns, and prunes them with DataFusion's
//! own [`PruningPredicate`](datafusion::physical_optimizer::pruning::PruningPredicate), a zone
//! to a container:
//!
//! - a column the file stores as it is has its zone maps' min and max, in the table's type;
//! - a rescaled decimal has them scaled back up to the table's scale;
//! - a price has bounds from those of the [`OHLC`] parts it is rebuilt from (see
//!   [`price_bounds`]).
//!
//! A file's zone maps are read in the background the first time a query sees it, through the
//! object store the scan reads it from (S3, a cache in front of it, or the local filesystem),
//! so that query, and any that plans before the reads are done, scans every zone as before.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use arrow::array::cast::AsArray;
use arrow::array::types::Decimal128Type;
use arrow::array::{Array, ArrayRef, BooleanArray, Decimal128Array};
use arrow::compute::{cast, nullif};
use arrow::datatypes::{DataType, Schema};
use datafusion::common::config::ConfigOptions;
use datafusion::common::pruning::PruningStatistics;
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::common::{Column, Result, ScalarValue};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{
    FileGroup, FileScanConfig, FileScanConfigBuilder, FileSource,
};
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::object_store::{ObjectStoreRegistry, ObjectStoreUrl};
use datafusion::object_store::ObjectMeta;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_optimizer::pruning::PruningPredicateBuilder;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::ExecutionPlan;
use moka::sync::Cache;
use roaring::RoaringTreemap;
use vortex::array::memory::MemorySessionExt;
use vortex::array::{MaskFuture, VortexSessionExecute};
use vortex::arrow::ArrowSessionExt;
use vortex::error::{vortex_err, VortexResult};
use vortex::expr::root;
use vortex::file::{OpenOptionsSessionExt, VortexFile};
use vortex::io::object_store::ObjectStoreReadAt;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::layouts::zoned::Zoned;
use vortex::layout::{LayoutReaderContext, LayoutRef};
use vortex::scan::selection::Selection;
use vortex::session::VortexSession;
use vortex_datafusion::{VortexAccessPlan, VortexSource};

use crate::layout::{find_child, find_zoned};
use crate::plan::{OHLC, OHLC_PARTS, OHLC_PRICES};

/// How many bytes of zone maps [`BarsPruning`] keeps by default: tens of KB a file, so some
/// thousands of files.
pub const DEFAULT_CACHE_BYTES: u64 = 256 << 20;

/// The largest share of a file's zones a selection keeps: a scan that keeps more than this skips
/// too few splits to make up for checking its selection on the rest.
const MAX_KEPT: f64 = 0.5;

/// Gives each file of a Vortex scan the zones its filter can match: see the module docs.
///
/// The session needs the crate registered (see [`crate::register`]).
pub struct BarsPruning {
    session: VortexSession,
    /// The object stores the scans read their files from, as DataFusion's runtime has them.
    stores: Arc<dyn ObjectStoreRegistry>,
    /// Zone maps by object store and path, up to a number of bytes, the least recently used
    /// evicted first.
    files: Cache<String, Arc<FileZones>>,
    /// The files whose zone maps are being read; shared with the tasks that read them.
    reading: Arc<Mutex<HashSet<String>>>,
}

impl BarsPruning {
    /// Keeping up to [`DEFAULT_CACHE_BYTES`] of zone maps.
    pub fn new(session: VortexSession, stores: Arc<dyn ObjectStoreRegistry>) -> Self {
        Self::with_cache_bytes(session, stores, DEFAULT_CACHE_BYTES)
    }

    /// Keeping up to `cache_bytes` of zone maps.
    pub fn with_cache_bytes(
        session: VortexSession,
        stores: Arc<dyn ObjectStoreRegistry>,
        cache_bytes: u64,
    ) -> Self {
        let files = Cache::builder()
            .weigher(|_, zones: &Arc<FileZones>| zones.bytes)
            .max_capacity(cache_bytes)
            .build();
        Self {
            session,
            stores,
            files,
            reading: Arc::default(),
        }
    }

    /// The zone maps of `file` in the object store at `url`, or `None` if they aren't read yet,
    /// in which case it starts reading them in the background (on the Tokio runtime the query
    /// is planned on, if there is one).
    fn zones(&self, url: &ObjectStoreUrl, file: &PartitionedFile) -> Option<Arc<FileZones>> {
        let key = format!("{}{}", url.as_str(), file.path());
        let meta = &file.object_meta;
        if let Some(zones) = self.files.get(&key) {
            if zones.meta.size == meta.size && zones.meta.last_modified == meta.last_modified {
                return Some(zones);
            }
        }
        let mut reading = self.reading.lock().unwrap_or_else(|e| e.into_inner());
        if reading.contains(&key) {
            return None;
        }
        let runtime = tokio::runtime::Handle::try_current().ok()?;
        let store = self.stores.get_store(url.as_ref()).ok()?;
        reading.insert(key.clone());
        let (session, path, meta) = (self.session.clone(), file.path().clone(), meta.clone());
        let (files, reading) = (self.files.clone(), Arc::clone(&self.reading));
        runtime.spawn(async move {
            let read = async {
                let read = ObjectStoreReadAt::new_with_allocator(
                    store,
                    path,
                    session.handle(),
                    session.allocator(),
                );
                let file = session
                    .open_options()
                    .with_file_size(meta.size)
                    .open_read(Arc::new(read))
                    .await?;
                FileZones::read(&session, &file, meta).await
            };
            // On an error, left unpruned until the next query tries again.
            if let Ok(zones) = read.await {
                files.insert(key.clone(), Arc::new(zones));
            }
            reading
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&key);
        });
        None
    }

    /// The rows of `file` in the zones that `predicate` (over `table`, the table's schema) can
    /// match; `None` to scan it whole.
    fn selection(
        &self,
        url: &ObjectStoreUrl,
        file: &PartitionedFile,
        predicate: &Arc<dyn PhysicalExpr>,
        table: &Arc<Schema>,
    ) -> Option<Selection> {
        let zones = self.zones(url, file)?;
        let pruning = PruningPredicateBuilder::new()
            .with_file_schema(Arc::clone(table))
            .build(Arc::clone(predicate))?;
        let stats = ZoneStatistics {
            zones: &zones,
            table,
        };
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
}

impl fmt::Debug for BarsPruning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BarsPruning").finish_non_exhaustive()
    }
}

impl PhysicalOptimizerRule for BarsPruning {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
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
            let Some(predicate) = source.predicate() else {
                return Ok(Transformed::no(plan));
            };
            let table = source.table_schema().table_schema();
            let mut pruned = false;
            let groups = conf
                .file_groups
                .iter()
                .map(|group| {
                    let files = group.iter().map(|file| {
                        let mut file = file.clone();
                        if let Some(selection) =
                            self.selection(&conf.object_store_url, &file, predicate, table)
                        {
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

    fn name(&self) -> &str {
        "vortex_ohlc_bars_pruning"
    }

    fn schema_check(&self) -> bool {
        true
    }
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
