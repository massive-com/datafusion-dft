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

//! Building a Cassandra [`Session`] from connection options

use std::time::Duration;

use datafusion::common::{DataFusionError, Result};
use scylla::client::session::Session;
use scylla::client::session_builder::SessionBuilder;

/// Options for connecting to a Cassandra cluster
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CassandraConnectionOptions {
    /// Contact points, each as `host` or `host:port` (default port 9042)
    pub contact_points: Vec<String>,
    /// Username for `PasswordAuthenticator` clusters
    pub user: Option<String>,
    /// Password for `PasswordAuthenticator` clusters
    pub password: Option<String>,
    /// Per-node connection timeout in milliseconds (driver default if unset)
    pub connection_timeout_ms: Option<u64>,
}

/// Connects to a Cassandra cluster and fetches its schema metadata
pub async fn connect(options: &CassandraConnectionOptions) -> Result<Session> {
    if options.contact_points.is_empty() {
        return Err(DataFusionError::Configuration(
            "Cassandra connection requires at least one contact point".to_string(),
        ));
    }
    let mut builder = SessionBuilder::new().known_nodes(&options.contact_points);
    if let (Some(user), Some(password)) = (&options.user, &options.password) {
        builder = builder.user(user, password);
    }
    if let Some(timeout_ms) = options.connection_timeout_ms {
        builder = builder.connection_timeout(Duration::from_millis(timeout_ms));
    }
    builder
        .build()
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))
}
