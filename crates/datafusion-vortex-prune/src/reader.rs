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

//! Which readers a table's files are read through. This has nothing to do with pruning, and is
//! here only until vortex-datafusion reads a local file with Vortex's own reader by default.

use std::sync::Arc;

use datafusion::common::{DataFusionError, Result};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::object_store::local::LocalFileSystem;
use datafusion::object_store::ObjectStore;
use url::Url;
use vortex::array::memory::MemorySessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::io::std_file::FileReadAt;
use vortex::io::VortexReadAt;
use vortex::session::VortexSession;
use vortex_datafusion::reader::{DefaultVortexReaderFactory, VortexReaderFactory};

/// The readers for files in the object store at `url`, `store`, as a table's scans should read
/// them: a local file directly, and anything else through `store`, as vortex-datafusion reads
/// every file by default. Through the `ObjectStore`, a local file is opened again for every read
/// and nearby reads are merged into requests of up to 16 MiB, a size meant for object storage;
/// Vortex's file reader keeps it open and merges reads into requests of at most 4 MiB.
///
/// A `file://` URL is local when it names a path on this machine (no host, or `localhost`); its
/// files are found under that path, as a `LocalFileSystem` rooted there finds them.
pub fn reader_factory(
    url: &ObjectStoreUrl,
    store: Arc<dyn ObjectStore>,
) -> Result<Arc<dyn VortexReaderFactory>> {
    let url: &Url = url.as_ref();
    if url.scheme() == "file" {
        if let Ok(root) = url.to_file_path() {
            let files = LocalFileSystem::new_with_prefix(root)?;
            return Ok(Arc::new(LocalFileReaderFactory(files)));
        }
    }
    Ok(Arc::new(DefaultVortexReaderFactory::new(store)))
}

#[derive(Debug)]
struct LocalFileReaderFactory(LocalFileSystem);

impl VortexReaderFactory for LocalFileReaderFactory {
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
