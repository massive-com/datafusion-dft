//! Vortex scans that skip the zones their filter can't match, set up as Atlas sets up its scans:
//! a table turns a scan's filters into one physical predicate and hands it, with the scan's
//! config, to a factory that builds the scan ([`VortexExecFactory`], after Atlas's), which prunes
//! the config's files before the format plans it.

use std::borrow::Cow;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{ScanArgs, ScanResult, Session, TableProvider};
use datafusion::common::{internal_datafusion_err, Constraints, DFSchema, Result, Statistics};
use datafusion::datasource::file_format::FileFormat;
use datafusion::datasource::listing::ListingTable;
use datafusion::datasource::physical_plan::{FileScanConfig, FileScanConfigBuilder};
use datafusion::datasource::source::DataSourceExec;
use datafusion::datasource::TableType;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::utils::conjunction;
use datafusion::logical_expr::{Expr, LogicalPlan, TableProviderFilterPushDown};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::ExecutionPlan;
use vortex_datafusion::reader::VortexReaderFactory;
use vortex_datafusion::VortexSource;

use super::ZonePruner;

/// Builds Vortex scans as Atlas's `VortexExecFactory` does, from a scan's config and filter: the
/// config's files pruned by a [`ZonePruner`] (as Atlas's default factory prunes file groups with
/// its index) and read through `readers`, the zone maps too, then planned by the format.
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
    pruner: Arc<ZonePruner>,
    readers: Arc<dyn VortexReaderFactory>,
}

impl VortexExecFactory {
    pub fn new(pruner: Arc<ZonePruner>, readers: Arc<dyn VortexReaderFactory>) -> Self {
        Self { pruner, readers }
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
        let conf = self
            .pruner
            .prune(file_scan_config, filter, self.readers.as_ref())
            .await?;
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
