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
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::object_store::local::LocalFileSystem;
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
        builder.add_physical_optimizer_rule(Arc::new(LocalVortexReaders));
        Ok(())
    }
}

/// Gives Vortex scans of local files a reader that reads the file directly and
/// records the scan's reads as `bytes_scanned` and `io_requests` metrics.
///
/// vortex-datafusion otherwise reads local files through the `ObjectStore`, which
/// opens the file again for every read and merges nearby reads into requests of up
/// to 16 MiB, a size meant for object storage. Vortex's file reader keeps the file
/// open and merges reads into requests of at most 4 MiB. Vortex records its own
/// I/O metrics outside the plan, so without these metrics `EXPLAIN ANALYZE` shows
/// none.
#[derive(Debug)]
struct LocalVortexReaders;

impl PhysicalOptimizerRule for LocalVortexReaders {
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
            // DataFusion registers a `LocalFileSystem` rooted at `/` for `file://`.
            if conf.object_store_url != ObjectStoreUrl::local_filesystem() {
                return Ok(Transformed::no(plan));
            }
            let Some(source) = conf.file_source().downcast_ref::<VortexSource>() else {
                return Ok(Transformed::no(plan));
            };
            let factory = LocalReaderFactory {
                store: LocalFileSystem::new(),
                metrics: ReadMetrics::new(source.metrics()),
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
        "local_vortex_readers"
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
