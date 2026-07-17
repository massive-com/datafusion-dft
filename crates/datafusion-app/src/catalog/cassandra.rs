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

//! [`CatalogProvider`] and [`SchemaProvider`] implementations backed by a Cassandra cluster

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::catalog::{CatalogProvider, SchemaProvider, TableProvider};
use datafusion::common::{plan_err, Result};
use datafusion_cassandra::scylla::client::session::Session;
use datafusion_cassandra::{keyspace_table_names, user_keyspace_names, CassandraTable};
use log::warn;

/// A [`CatalogProvider`] that exposes the keyspaces of a Cassandra cluster as schemas. The
/// available keyspaces and tables come from the driver's schema metadata, fetched once when the
/// catalog is created; the data itself is queried from Cassandra on demand.
#[derive(Debug)]
pub struct CassandraCatalogProvider {
    schemas: HashMap<String, Arc<CassandraSchemaProvider>>,
}

impl CassandraCatalogProvider {
    /// Discover the keyspaces (optionally limited to `keyspace`) and tables available in the
    /// cluster behind `session`
    pub fn try_new(session: Arc<Session>, keyspace: Option<String>) -> Result<Self> {
        let keyspace_names = match keyspace {
            Some(keyspace) => {
                if session
                    .get_cluster_state()
                    .get_keyspace(&keyspace)
                    .is_none()
                {
                    return plan_err!("Cassandra keyspace '{keyspace}' not found");
                }
                vec![keyspace]
            }
            None => user_keyspace_names(&session),
        };

        let mut schemas = HashMap::with_capacity(keyspace_names.len());
        for keyspace_name in keyspace_names {
            let tables = keyspace_table_names(&session, &keyspace_name);
            let provider = CassandraSchemaProvider {
                session: Arc::clone(&session),
                keyspace: keyspace_name.clone(),
                tables,
            };
            schemas.insert(keyspace_name, Arc::new(provider));
        }

        Ok(Self { schemas })
    }
}

impl CatalogProvider for CassandraCatalogProvider {
    fn schema_names(&self) -> Vec<String> {
        self.schemas.keys().cloned().collect()
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        self.schemas
            .get(name)
            .map(|s| Arc::clone(s) as Arc<dyn SchemaProvider>)
    }
}

/// A [`SchemaProvider`] for a single Cassandra keyspace. Table names are cached at creation
/// time; the Arrow schema for a table comes from the driver's cached metadata when the table is
/// used in a query.
#[derive(Debug)]
pub struct CassandraSchemaProvider {
    session: Arc<Session>,
    keyspace: String,
    tables: Vec<String>,
}

#[async_trait::async_trait]
impl SchemaProvider for CassandraSchemaProvider {
    fn table_names(&self) -> Vec<String> {
        self.tables.clone()
    }

    async fn table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        if !self.table_exist(name) {
            return Ok(None);
        }
        match CassandraTable::try_new(Arc::clone(&self.session), &self.keyspace, name) {
            Ok(provider) => Ok(Some(Arc::new(provider))),
            Err(e) => {
                warn!(
                    "Failed to build provider for Cassandra table '{}.{}', hiding it from the catalog: {e}",
                    self.keyspace, name
                );
                Ok(None)
            }
        }
    }

    fn table_exist(&self, name: &str) -> bool {
        self.tables.iter().any(|t| t == name)
    }
}
