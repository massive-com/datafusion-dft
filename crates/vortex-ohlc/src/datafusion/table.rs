//! [`PrunedTable`]: a table whose Vortex scans skip the zones their filter can't match, with the
//! zone maps read while the scan is set up, so every query is pruned.

use std::borrow::Cow;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{ScanArgs, ScanResult, Session, TableProvider};
use datafusion::common::{Constraints, DFSchema, Result, Statistics};
use datafusion::datasource::TableType;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::utils::conjunction;
use datafusion::logical_expr::{Expr, LogicalPlan, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use vortex_datafusion::reader::VortexReaderFactory;

use super::ZonePruner;

/// A table that scans `inner` and prunes its Vortex scans with a [`ZonePruner`], by the filters
/// the scan is given, reading the zone maps it doesn't have through `readers` (as the scans
/// read their data, so through the same cache). Everything else is `inner`'s.
#[derive(Debug)]
pub struct PrunedTable {
    inner: Arc<dyn TableProvider>,
    pruner: Arc<ZonePruner>,
    readers: Arc<dyn VortexReaderFactory>,
}

impl PrunedTable {
    pub fn new(
        inner: Arc<dyn TableProvider>,
        pruner: Arc<ZonePruner>,
        readers: Arc<dyn VortexReaderFactory>,
    ) -> Self {
        Self {
            inner,
            pruner,
            readers,
        }
    }

    /// `plan`, a scan of `inner`, with its Vortex scans pruned by `filters`.
    async fn prune(
        &self,
        state: &dyn Session,
        plan: Arc<dyn ExecutionPlan>,
        filters: &[Expr],
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let Some(filter) = conjunction(filters.iter().cloned()) else {
            return Ok(plan);
        };
        let schema = DFSchema::try_from(self.schema())?;
        let predicate = state.create_physical_expr(filter, &schema)?;
        self.pruner
            .prune(plan, Some(&predicate), self.readers.as_ref())
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
        self.prune(state, plan, filters).await
    }

    async fn scan_with_args<'a>(
        &self,
        state: &dyn Session,
        args: ScanArgs<'a>,
    ) -> Result<ScanResult> {
        let filters = args.filters().unwrap_or(&[]).to_vec();
        let plan = self.inner.scan_with_args(state, args).await?.into_inner();
        Ok(self.prune(state, plan, &filters).await?.into())
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
