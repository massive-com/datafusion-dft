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
use datafusion::datasource::physical_plan::{FileScanConfig, FileScanConfigBuilder, FileSource};
use datafusion::datasource::provider::DefaultTableFactory;
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::object_store::{ObjectStoreRegistry, ObjectStoreUrl};
use datafusion::object_store::local::LocalFileSystem;
use datafusion::object_store::ObjectStore;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::metrics::{
    Count, ExecutionPlanMetricsSet, MetricBuilder, MetricCategory, MetricType,
};
use datafusion::physical_plan::ExecutionPlan;
use futures::future::BoxFuture;
use std::sync::Arc;
use vortex::array::buffer::BufferHandle;
use vortex::array::memory::MemorySessionExt;
use vortex::buffer::Alignment;
use vortex::error::VortexResult;
use vortex::io::object_store::ObjectStoreReadAt;
use vortex::io::session::RuntimeSessionExt;
use vortex::io::std_file::FileReadAt;
use vortex::io::{CoalesceConfig, ReadAtRequest, ReadAtStream, VortexReadAt};
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
        let stores = Arc::clone(&builder.runtime_env().object_store_registry);
        builder.add_physical_optimizer_rule(Arc::new(VortexReaders { stores }));
        Ok(())
    }
}

/// Gives Vortex scans a reader that records the scan's reads as `bytes_scanned`
/// and `io_requests` metrics.
///
/// Local files are read directly. vortex-datafusion otherwise reads them through
/// the `ObjectStore`, which opens the file again for every read and merges nearby
/// reads into requests of up to 16 MiB, a size meant for object storage. Vortex's
/// file reader keeps the file open and merges reads into requests of at most
/// 4 MiB. Vortex records its own I/O metrics outside the plan, so without these
/// metrics `EXPLAIN ANALYZE` shows none.
#[derive(Debug)]
struct VortexReaders {
    stores: Arc<dyn ObjectStoreRegistry>,
}

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
            let metrics = ReadMetrics::new(source.metrics());
            // DataFusion registers a `LocalFileSystem` rooted at `/` for `file://`.
            let factory: Arc<dyn VortexReaderFactory> =
                if conf.object_store_url == ObjectStoreUrl::local_filesystem() {
                    Arc::new(LocalReaderFactory {
                        store: LocalFileSystem::new(),
                        metrics,
                    })
                } else {
                    // Without a registered store the scan fails when it opens the file, with
                    // DataFusion's own error, so it is left as it is.
                    let Ok(store) = self.stores.get_store(conf.object_store_url.as_ref()) else {
                        return Ok(Transformed::no(plan));
                    };
                    Arc::new(ObjectStoreReaderFactory { store, metrics })
                };
            let source = source.clone().with_vortex_reader_factory(factory);
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
    metrics: ReadMetrics,
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
        Ok(Arc::new(MeteredReadAt {
            inner: file,
            metrics: self.metrics.clone(),
        }))
    }
}

#[derive(Debug)]
struct ObjectStoreReaderFactory {
    store: Arc<dyn ObjectStore>,
    metrics: ReadMetrics,
}

impl VortexReaderFactory for ObjectStoreReaderFactory {
    fn create_reader(
        &self,
        file: &PartitionedFile,
        session: &VortexSession,
    ) -> Result<Arc<dyn VortexReadAt>> {
        let object = ObjectStoreReadAt::new_with_allocator(
            Arc::clone(&self.store),
            file.path().clone(),
            session.handle(),
            session.allocator(),
        );
        Ok(Arc::new(MeteredReadAt {
            inner: object,
            metrics: self.metrics.clone(),
        }))
    }
}

/// Read metrics registered with a scan's DataFusion metrics.
#[derive(Clone, Debug)]
struct ReadMetrics {
    bytes_scanned: Count,
    io_requests: Count,
}

impl ReadMetrics {
    fn new(metrics: &ExecutionPlanMetricsSet) -> Self {
        Self {
            bytes_scanned: MetricBuilder::new(metrics)
                .with_type(MetricType::Summary)
                .with_category(MetricCategory::Bytes)
                .global_counter("bytes_scanned"),
            io_requests: MetricBuilder::new(metrics)
                .with_type(MetricType::Summary)
                .global_counter("io_requests"),
        }
    }

    fn record(&self, length: usize) {
        self.bytes_scanned.add(length);
        self.io_requests.add(1);
    }
}

struct MeteredReadAt<R> {
    inner: R,
    metrics: ReadMetrics,
}

impl<R: VortexReadAt> VortexReadAt for MeteredReadAt<R> {
    fn uri(&self) -> Option<&Arc<str>> {
        self.inner.uri()
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.inner.coalesce_config()
    }

    fn concurrency(&self) -> usize {
        self.inner.concurrency()
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.inner.size()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        self.metrics.record(length);
        self.inner.read_at(offset, length, alignment)
    }

    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        for request in requests.iter() {
            self.metrics.record(request.length);
        }
        self.inner.read_ranges(requests)
    }
}
