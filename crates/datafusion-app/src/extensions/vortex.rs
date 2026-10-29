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
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::common::{DataFusionError, Result};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{FileScanConfig, FileScanConfigBuilder};
use datafusion::datasource::provider::DefaultTableFactory;
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::object_store::local::LocalFileSystem;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::ExecutionPlan;
use std::sync::Arc;
use vortex::array::memory::MemorySessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::io::std_file::FileReadAt;
use vortex::io::VortexReadAt;
use vortex::session::VortexSession;
use vortex_datafusion::reader::VortexReaderFactory;
use vortex_datafusion::{VortexFormatFactory, VortexSource, VortexTableOptions};

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
        // built-in formats' tables.
        builder.add_table_factory("VORTEX", Arc::new(DefaultTableFactory::new()));
        builder.add_physical_optimizer_rule(Arc::new(VortexReaders));
        Ok(())
    }
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
