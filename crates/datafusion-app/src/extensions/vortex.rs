// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Vortex file format integration: [VortexExtension]

use crate::config::ExecutionConfig;
use crate::extensions::{DftSessionStateBuilder, Extension};
use datafusion::catalog::{
    DynamicFileCatalog, Session, TableProvider, TableProviderFactory, UrlTableFactory,
};
use datafusion::common::Result;
use datafusion::datasource::dynamic_file::DynamicListTableFactory;
use datafusion::datasource::listing::ListingTable;
use datafusion::datasource::provider::DefaultTableFactory;
use datafusion::execution::SessionState;
use datafusion::logical_expr::CreateExternalTable;
use datafusion::prelude::SessionContext;
use datafusion_session::SessionStore;
use std::sync::Arc;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_datafusion::{VortexFormat, VortexTableOptions};
use vortex_ohlc::datafusion::{vortex_format_factory, PrunedTable, ZonePruner};

#[derive(Debug, Default)]
pub struct VortexExtension {}

impl VortexExtension {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait::async_trait]
impl Extension for VortexExtension {
    async fn register(
        &self,
        _config: ExecutionConfig,
        builder: &mut DftSessionStateBuilder,
    ) -> datafusion::common::Result<()> {
        // The factory reads the session's `vortex` options when it creates a format, so
        // registering them makes `SET vortex.<option>` apply to later queries.
        builder.add_config_extension(VortexTableOptions::default())?;
        // Filters on the columns of files written by vortex-ohlc (see `pruned`) go to Vortex on
        // the columns as stored; anything else as vortex-datafusion converts it by default.
        let session = VortexSession::default();
        builder.add_file_format_factory(Arc::new(vortex_format_factory(session.clone())));
        // `CREATE EXTERNAL TABLE ... STORED AS VORTEX`, created the way DataFusion creates its
        // built-in formats' tables, then pruned (see `pruned`) with the session's zone maps.
        builder.add_table_factory("VORTEX", Arc::new(PrunedTableFactory::default()));
        let pruner = ZonePruner::new(session);
        builder.add_session_extension(Arc::new(pruner));
        Ok(())
    }
}

/// `table`, if it's a listing table of Vortex files, with its scans pruned to the zones their
/// filter can match before they start, by the [`ZonePruner`] `state` keeps, which keeps the zone
/// maps for as long as the session lasts (see [`PrunedTable::over`]). Files written by
/// vortex-ohlc are read as the source they were written from: the table has the source's
/// columns, not the ones the files store.
async fn pruned(
    table: Arc<dyn TableProvider>,
    state: &dyn Session,
) -> Result<Arc<dyn TableProvider>> {
    let Some(listing) = table.downcast_ref::<ListingTable>() else {
        return Ok(table);
    };
    let (Some(_), Some(pruner)) = (
        listing.options().format.downcast_ref::<VortexFormat>(),
        state.config().get_extension::<ZonePruner>(),
    ) else {
        return Ok(table);
    };
    Ok(Arc::new(PrunedTable::over(listing, state, pruner).await?))
}

/// DataFusion's table factory, its Vortex tables [`pruned`].
#[derive(Debug, Default)]
struct PrunedTableFactory(DefaultTableFactory);

#[async_trait::async_trait]
impl TableProviderFactory for PrunedTableFactory {
    async fn create(
        &self,
        state: &dyn Session,
        cmd: &CreateExternalTable,
    ) -> Result<Arc<dyn TableProvider>> {
        pruned(self.0.create(state, cmd).await?, state).await
    }
}

/// DataFusion's tables for a file or directory queried by its path, its Vortex tables [`pruned`].
#[derive(Debug)]
struct PrunedUrlTables(DynamicListTableFactory);

#[async_trait::async_trait]
impl UrlTableFactory for PrunedUrlTables {
    async fn try_new(&self, url: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        let Some(table) = self.0.try_new(url).await? else {
            return Ok(None);
        };
        let session = self.0.session_store().get_session().upgrade();
        let Some(session) = session else {
            return Ok(Some(table));
        };
        // A copy, as DataFusion takes one to create the table: the lock isn't held while the
        // files are read.
        let state = session
            .read()
            .as_any()
            .downcast_ref::<SessionState>()
            .cloned();
        let Some(state) = state else {
            return Ok(Some(table));
        };
        pruned(table, &state).await.map(Some)
    }
}

/// [`SessionContext::enable_url_table`], with the Vortex tables [`pruned`].
pub(crate) fn enable_url_table(ctx: SessionContext) -> SessionContext {
    let catalogs = Arc::clone(ctx.state().catalog_list());
    let tables = Arc::new(PrunedUrlTables(DynamicListTableFactory::new(
        SessionStore::new(),
    )));
    let session_id = ctx.session_id();
    let state: SessionState = ctx
        .into_state_builder()
        .with_session_id(session_id)
        .with_catalog_list(Arc::new(DynamicFileCatalog::new(
            catalogs,
            Arc::clone(&tables) as Arc<dyn UrlTableFactory>,
        )))
        .build();
    let ctx = SessionContext::new_with_state(state);
    tables.0.session_store().with_state(ctx.state_weak_ref());
    ctx
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::datasource::physical_plan::FileScanConfig;
    use datafusion::datasource::source::DataSourceExec;
    use datafusion::physical_plan::ExecutionPlan;
    use vortex_datafusion::VortexAccessPlan;

    /// For each file of each Vortex scan in `plan`, whether it was given a selection.
    fn selected(plan: &Arc<dyn ExecutionPlan>) -> Vec<bool> {
        let mut selected = Vec::new();
        plan.apply(|node| {
            let conf = node
                .downcast_ref::<DataSourceExec>()
                .and_then(|exec| exec.data_source().downcast_ref::<FileScanConfig>());
            let groups = conf.iter().flat_map(|conf| conf.file_groups.iter());
            for file in groups.flat_map(|group| group.iter()) {
                let plan = file.extensions.get::<VortexAccessPlan>();
                selected.push(plan.and_then(VortexAccessPlan::selection).is_some());
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        selected
    }

    #[tokio::test]
    async fn tables_by_path_and_external_are_pruned() -> Result<()> {
        let mut builder = DftSessionStateBuilder::default();
        VortexExtension::new()
            .register(ExecutionConfig::default(), &mut builder)
            .await?;
        let ctx = enable_url_table(SessionContext::new_with_state(builder.build()?));
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("ids.vortex");
        let path = path.to_str().unwrap();
        let sql = |sql: String| {
            let ctx = ctx.clone();
            async move { ctx.sql(&sql).await }
        };
        // Enough rows for some zones.
        sql(format!(
            "COPY (SELECT value AS id FROM generate_series(1, 100000)) TO '{path}' STORED AS VORTEX"
        ))
        .await?
        .collect()
        .await?;
        sql(format!(
            "CREATE EXTERNAL TABLE ids STORED AS VORTEX LOCATION '{path}'"
        ))
        .await?
        .collect()
        .await?;

        for table in [format!("'{path}'"), "ids".to_string()] {
            // One zone of the file can match, so the scan reads that one only.
            let query = sql(format!("SELECT * FROM {table} WHERE id > 99000")).await?;
            let plan = query.clone().create_physical_plan().await?;
            assert_eq!(selected(&plan), [true], "{table}");
            assert_eq!(query.count().await?, 1000, "{table}");
            // Every zone can, so the scan reads them all as it would have.
            let query = sql(format!("SELECT * FROM {table} WHERE id > 1000")).await?;
            let plan = query.clone().create_physical_plan().await?;
            assert!(!selected(&plan).contains(&true), "{table}");
            assert_eq!(query.count().await?, 99000, "{table}");
        }
        Ok(())
    }
}
