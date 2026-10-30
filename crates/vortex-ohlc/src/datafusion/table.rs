//! Vortex scans that skip the zones their filter can't match, set up as Atlas sets up its scans:
//! a table turns a scan's filters into one physical predicate and hands it, with the scan's
//! config, to a factory that builds the scan ([`VortexExecFactory`], after Atlas's), which prunes
//! the config's files before the format plans it.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{ScanArgs, ScanResult, Session, TableProvider};
use datafusion::common::{
    internal_datafusion_err, Constraints, DFSchema, DataFusionError, Result, Statistics,
};
use datafusion::datasource::file_format::FileFormat;
use datafusion::datasource::listing::{ListingTable, ListingTableConfig, PartitionedFile};
use datafusion::datasource::physical_plan::{FileScanConfig, FileScanConfigBuilder};
use datafusion::datasource::source::DataSourceExec;
use datafusion::datasource::TableType;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::utils::conjunction;
use datafusion::logical_expr::{Expr, LogicalPlan, TableProviderFilterPushDown};
use datafusion::object_store::local::LocalFileSystem;
use datafusion::object_store::ObjectStore;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::ExecutionPlan;
use futures::StreamExt;
use vortex::array::memory::MemorySessionExt;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::io::std_file::FileReadAt;
use vortex::io::VortexReadAt;
use vortex::session::VortexSession;
use vortex_datafusion::reader::{DefaultVortexReaderFactory, VortexReaderFactory};
use vortex_datafusion::VortexSource;

use super::{BarsAdapterFactory, ZonePruner};
use crate::write::file_source_schema;

/// Builds Vortex scans as Atlas's `VortexExecFactory` does, from a scan's config and filter: the
/// config's files read through `readers` and, given a [`ZonePruner`] (see
/// [`with_pruner`](Self::with_pruner)), pruned by it as Atlas's default factory prunes file
/// groups with its index, the zone maps read through `readers` too; then planned by the format.
///
/// Atlas's own factory would prune its scans the same way with one line at the top of its
/// `create_execution_plan`, given a `ZonePruner` kept next to its reader factory:
///
/// ```ignore
/// let file_scan_config = self.pruner.prune(file_scan_config, filter, &*self.reader_factory).await?;
/// ```
///
/// Atlas pushes the filter into the scan itself, so it can keep doing that after this line.
#[derive(Debug)]
pub struct VortexExecFactory {
    readers: Arc<dyn VortexReaderFactory>,
    pruner: Option<Arc<ZonePruner>>,
}

impl VortexExecFactory {
    /// Scans that read through `readers` and aren't pruned.
    pub fn new(readers: Arc<dyn VortexReaderFactory>) -> Self {
        Self {
            readers,
            pruner: None,
        }
    }

    /// Scans pruned by `pruner` before they start.
    pub fn with_pruner(mut self, pruner: Arc<ZonePruner>) -> Self {
        self.pruner = Some(pruner);
        self
    }

    /// A scan of `file_scan_config`, a Vortex scan's config, by `file_format`, with the zones
    /// `filter` can match. The filter itself isn't pushed into the scan: DataFusion's optimizer
    /// pushes it into a listing table's scans, and a scan given it twice would evaluate it twice.
    pub async fn create_execution_plan(
        &self,
        state: &dyn Session,
        file_scan_config: FileScanConfig,
        file_format: &Arc<dyn FileFormat>,
        filter: Option<&Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let conf = match &self.pruner {
            Some(pruner) => {
                pruner
                    .prune(file_scan_config, filter, self.readers.as_ref())
                    .await?
            }
            None => file_scan_config,
        };
        let source = conf
            .file_source()
            .downcast_ref::<VortexSource>()
            .cloned()
            .ok_or_else(|| internal_datafusion_err!("VortexExecFactory requires a VortexSource"))?
            .with_vortex_reader_factory(Arc::clone(&self.readers));
        let conf = FileScanConfigBuilder::from(conf)
            .with_source(Arc::new(source))
            .build();
        file_format.create_physical_plan(state, conf).await
    }
}

/// The readers for files in the object store at `url`, `store`, as a table's scans should read
/// them: a local file directly, and anything else through `store`, as vortex-datafusion reads
/// every file by default. Through the `ObjectStore`, a local file is opened again for every read
/// and nearby reads are merged into requests of up to 16 MiB, a size meant for object storage;
/// Vortex's file reader keeps it open and merges reads into requests of at most 4 MiB.
pub fn reader_factory(
    url: &ObjectStoreUrl,
    store: Arc<dyn ObjectStore>,
) -> Arc<dyn VortexReaderFactory> {
    // DataFusion registers a `LocalFileSystem` rooted at `/` for `file://`.
    if *url == ObjectStoreUrl::local_filesystem() {
        Arc::new(LocalFileReaders(LocalFileSystem::new()))
    } else {
        Arc::new(DefaultVortexReaderFactory::new(store))
    }
}

/// Vortex's own reader for a local file, at the path `LocalFileSystem` maps it to.
#[derive(Debug)]
struct LocalFileReaders(LocalFileSystem);

impl VortexReaderFactory for LocalFileReaders {
    fn create_reader(
        &self,
        file: &PartitionedFile,
        session: &VortexSession,
    ) -> Result<Arc<dyn VortexReadAt>> {
        let path = self.0.path_to_filesystem(file.path())?;
        let file = FileReadAt::open_with_allocator(path, session.handle(), session.allocator())
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(Arc::new(file))
    }
}

/// `listing`, a listing table over Vortex files, as a table of the source they were written from,
/// if the first of them records a plan (see [`file_source_schema`]): with the source's schema
/// (columns the table already has as the source has them are kept), and every file read through a
/// [`BarsAdapterFactory`]. Its definition, constraints and column defaults are kept, and its files'
/// statistics are kept in `state`'s cache, as DataFusion's `CREATE EXTERNAL TABLE` keeps them.
/// The first file is opened on `session`, through `readers`. `None` if the table has no files, or
/// the first records no plan that fits the table.
async fn source_table(
    listing: &ListingTable,
    state: &dyn Session,
    readers: &dyn VortexReaderFactory,
    session: &VortexSession,
) -> Result<Option<ListingTable>> {
    let Some(url) = listing.table_paths().first() else {
        return Ok(None);
    };
    let store = state.runtime_env().object_store(url.object_store())?;
    let extension = &listing.options().file_extension;
    let first = url
        .list_all_files(state, store.as_ref(), extension)
        .await?
        .next()
        .await
        .transpose()?;
    let Some(first) = first else {
        return Ok(None);
    };
    let file = PartitionedFile::from(first);
    let reader = readers.create_reader(&file, session)?;
    let file = session
        .open_options()
        .include_metadata()
        .with_file_size(file.object_meta.size)
        .open_read(reader)
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    // The table's schema has the partition columns after the files'.
    let table = listing.schema();
    let partitions = listing.options().table_partition_cols.len();
    let files = &table.fields()[..table.fields().len() - partitions];
    let stored = Schema::new_with_metadata(files.to_vec(), table.metadata().clone());
    let Some(source) = file_source_schema(&file, &stored) else {
        return Ok(None);
    };
    let column_defaults: HashMap<String, Expr> = table
        .fields()
        .iter()
        .filter_map(|field| {
            let default = listing.get_column_default(field.name())?;
            Some((field.name().clone(), default.clone()))
        })
        .collect();
    let config = ListingTableConfig::new_with_multi_paths(listing.table_paths().clone())
        .with_listing_options(listing.options().clone())
        .with_schema(Arc::new(source))
        .with_expr_adapter_factory(Arc::new(BarsAdapterFactory));
    let cache = state.runtime_env().cache_manager.get_file_statistic_cache();
    Ok(Some(
        ListingTable::try_new(config)?
            .with_cache(cache)
            .with_definition(listing.get_table_definition().map(str::to_string))
            .with_constraints(listing.constraints().cloned().unwrap_or_default())
            .with_column_defaults(column_defaults),
    ))
}

/// A listing table over Vortex files whose scans a [`VortexExecFactory`] builds, given each scan's
/// filters, as Atlas's `FileScanTable` hands its scans to its factory. Everything else is
/// `inner`'s.
#[derive(Debug)]
pub struct PrunedTable {
    inner: Arc<ListingTable>,
    factory: VortexExecFactory,
}

impl PrunedTable {
    pub fn new(inner: Arc<ListingTable>, factory: VortexExecFactory) -> Self {
        Self { inner, factory }
    }

    /// `listing`, a listing table over Vortex files, read as the source they were written from
    /// where the first of them records a plan (the source's schema, every file read through a
    /// [`BarsAdapterFactory`]), its scans pruned by `pruner` before they start. Its files, and
    /// their zone maps, are read as [`reader_factory`] picks for its object store.
    pub async fn over(
        listing: &ListingTable,
        state: &dyn Session,
        pruner: Arc<ZonePruner>,
    ) -> Result<Self> {
        let url = listing
            .table_paths()
            .first()
            .ok_or_else(|| internal_datafusion_err!("a listing table without paths"))?
            .object_store();
        let store = state.runtime_env().object_store(&url)?;
        let readers = reader_factory(&url, store);
        let source = source_table(listing, state, readers.as_ref(), pruner.session()).await?;
        let listing = source.unwrap_or_else(|| listing.clone());
        let factory = VortexExecFactory::new(readers).with_pruner(pruner);
        Ok(Self::new(Arc::new(listing), factory))
    }

    /// `plan`, a scan of `inner`, rebuilt by the factory with `filters`.
    async fn rebuild(
        &self,
        state: &dyn Session,
        plan: Arc<dyn ExecutionPlan>,
        filters: &[Expr],
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // A table with no files scans none.
        let Some(conf) = plan
            .downcast_ref::<DataSourceExec>()
            .and_then(|exec| exec.data_source().downcast_ref::<FileScanConfig>())
        else {
            return Ok(plan);
        };
        let filter = conjunction(filters.iter().cloned())
            .map(|filter| state.create_physical_expr(filter, &DFSchema::try_from(self.schema())?))
            .transpose()?;
        let format = &self.inner.options().format;
        self.factory
            .create_execution_plan(state, conf.clone(), format, filter.as_ref())
            .await
    }
}

#[async_trait]
impl TableProvider for PrunedTable {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    fn constraints(&self) -> Option<&Constraints> {
        self.inner.constraints()
    }

    fn table_type(&self) -> TableType {
        self.inner.table_type()
    }

    fn get_table_definition(&self) -> Option<&str> {
        self.inner.get_table_definition()
    }

    fn get_logical_plan(&'_ self) -> Option<Cow<'_, LogicalPlan>> {
        self.inner.get_logical_plan()
    }

    fn get_column_default(&self, column: &str) -> Option<&Expr> {
        self.inner.get_column_default(column)
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let plan = self.inner.scan(state, projection, filters, limit).await?;
        self.rebuild(state, plan, filters).await
    }

    async fn scan_with_args<'a>(
        &self,
        state: &dyn Session,
        args: ScanArgs<'a>,
    ) -> Result<ScanResult> {
        let filters = args.filters().unwrap_or(&[]).to_vec();
        let plan = self.inner.scan_with_args(state, args).await?.into_inner();
        Ok(self.rebuild(state, plan, &filters).await?.into())
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        self.inner.supports_filters_pushdown(filters)
    }

    fn statistics(&self) -> Option<Statistics> {
        self.inner.statistics()
    }

    async fn insert_into(
        &self,
        state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.inner.insert_into(state, input, insert_op).await
    }
}
