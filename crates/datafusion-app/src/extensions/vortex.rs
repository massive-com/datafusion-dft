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
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::common::{DataFusionError, Result};
use datafusion::datasource::dynamic_file::DynamicListTableFactory;
use datafusion::datasource::listing::ListingTable;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{FileScanConfig, FileScanConfigBuilder};
use datafusion::datasource::provider::DefaultTableFactory;
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::SessionState;
use datafusion::logical_expr::CreateExternalTable;
use datafusion::object_store::local::LocalFileSystem;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;
use datafusion_session::SessionStore;
use std::sync::Arc;
use vortex::array::memory::MemorySessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::io::std_file::FileReadAt;
use vortex::io::VortexReadAt;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_datafusion::reader::{DefaultVortexReaderFactory, VortexReaderFactory};
use vortex_datafusion::{VortexFormat, VortexFormatFactory, VortexSource, VortexTableOptions};
use vortex_ohlc::datafusion::{PrunedTable, VortexExecFactory, ZonePruner};

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
        builder.add_file_format_factory(Arc::new(VortexFormatFactory::new()));
        // `CREATE EXTERNAL TABLE ... STORED AS VORTEX`, created the way DataFusion creates its
        // built-in formats' tables, then pruned (see `pruned`) with the session's zone maps.
        builder.add_table_factory("VORTEX", Arc::new(PrunedTableFactory::default()));
        let pruner = ZonePruner::new(VortexSession::default());
        builder.add_session_extension(Arc::new(pruner));
        builder.add_physical_optimizer_rule(Arc::new(VortexReaders));
        Ok(())
    }
}

/// `table`, if it's a listing table of Vortex files, with its scans pruned to the zones their
/// filter can match before they start (see [`PrunedTable`]), by the [`ZonePruner`] `state` keeps,
/// which keeps the zone maps for as long as the session lasts. The zone maps are read through the
/// table's object store.
fn pruned(table: Arc<dyn TableProvider>, state: &dyn Session) -> Result<Arc<dyn TableProvider>> {
    let Some(listing) = table.downcast_ref::<ListingTable>() else {
        return Ok(table);
    };
    let (Some(_), Some(url), Some(pruner)) = (
        listing.options().format.downcast_ref::<VortexFormat>(),
        listing.table_paths().first(),
        state.config().get_extension::<ZonePruner>(),
    ) else {
        return Ok(table);
    };
    let store = state.runtime_env().object_store(url.object_store())?;
    let readers = Arc::new(DefaultVortexReaderFactory::new(store));
    let factory = VortexExecFactory::new(pruner, readers);
    Ok(Arc::new(PrunedTable::new(
        Arc::new(listing.clone()),
        factory,
    )))
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
        pruned(self.0.create(state, cmd).await?, state)
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
        let state = session.read();
        pruned(table, &*state).map(Some)
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

/// Gives Vortex scans of local files a reader that reads the file directly.
///
/// vortex-datafusion otherwise reads them through the `ObjectStore`, which opens the file
/// again for every read and merges nearby reads into requests of up to 16 MiB, a size meant for
/// object storage. Vortex's file reader keeps the file open and merges reads into requests of
/// at most 4 MiB. Vortex records every scan's reads in its own metrics, whichever reader it has
/// (see `stats.rs`).
#[derive(Debug)]
struct VortexReaders;

impl PhysicalOptimizerRule for VortexReaders {
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
            // DataFusion registers a `LocalFileSystem` rooted at `/` for `file://`.
            if conf.object_store_url != ObjectStoreUrl::local_filesystem() {
                return Ok(Transformed::no(plan));
            }
            let factory = LocalReaderFactory {
                store: LocalFileSystem::new(),
            };
            let source = source.clone().with_vortex_reader_factory(Arc::new(factory));
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
        "vortex_readers"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[derive(Debug)]
struct LocalReaderFactory {
    store: LocalFileSystem,
}

impl VortexReaderFactory for LocalReaderFactory {
    fn create_reader(
        &self,
        file: &PartitionedFile,
        session: &VortexSession,
    ) -> Result<Arc<dyn VortexReadAt>> {
        let path = self.store.path_to_filesystem(file.path())?;
        let file = FileReadAt::open_with_allocator(path, session.handle(), session.allocator())
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(Arc::new(file))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::common::tree_node::TreeNodeRecursion;
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
