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

//! Cassandra integration: [CassandraExtension]

use crate::catalog::cassandra::CassandraCatalogProvider;
use crate::config::ExecutionConfig;
use crate::extensions::{DftSessionStateBuilder, Extension};
use datafusion::common::Result;
use log::info;
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct CassandraExtension {}

impl CassandraExtension {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait::async_trait]
impl Extension for CassandraExtension {
    async fn register(
        &self,
        config: ExecutionConfig,
        builder: &mut DftSessionStateBuilder,
    ) -> Result<()> {
        for cassandra_config in config.cassandra.iter().flatten() {
            let session =
                datafusion_cassandra::connect(&cassandra_config.to_connection_options()).await?;
            let catalog = CassandraCatalogProvider::try_new(
                Arc::new(session),
                cassandra_config.keyspace.clone(),
            )?;
            builder.add_catalog_provider(&cassandra_config.name, Arc::new(catalog));
            info!(
                "Registered Cassandra catalog '{}' for {:?}",
                cassandra_config.name, cassandra_config.contact_points
            );
        }
        Ok(())
    }
}
