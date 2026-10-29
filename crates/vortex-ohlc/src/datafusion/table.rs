//! Vortex scans that skip the zones their filter can't match, set up as Atlas sets up its scans:
//! a table turns a scan's filters into one physical predicate and hands it, with the scan's
//! config, to a factory that builds the scan ([`VortexExecFactory`], after Atlas's), which prunes
//! the config's files before the format plans it.

use std::borrow::Cow;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{ScanArgs, ScanResult, Session, TableProvider};
use datafusion::common::{
    internal_datafusion_err, Constraints, DFSchema, DataFusionError, Result, Statistics,
};
use datafusion::datasource::file_format::FileFormat;
use datafusion::datasource::listing::{ListingTable, PartitionedFile};
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
use vortex::array::memory::MemorySessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::io::std_file::FileReadAt;
use vortex::io::VortexReadAt;
use vortex::session::VortexSession;
use vortex_datafusion::reader::{DefaultVortexReaderFactory, VortexReaderFactory};
use vortex_datafusion::VortexSource;

use super::ZonePruner;

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
