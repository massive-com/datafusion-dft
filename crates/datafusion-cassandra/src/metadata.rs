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

//! Keyspace and table discovery from the driver's cached cluster metadata.
//! The driver fetches `system_schema` on connect (and keeps it fresh), so
//! these helpers never query the cluster.

use scylla::client::session::Session;
use scylla::cluster::metadata::Table;

/// Keyspaces that hold Cassandra internals rather than user data
pub const SYSTEM_KEYSPACES: [&str; 7] = [
    "system",
    "system_auth",
    "system_distributed",
    "system_schema",
    "system_traces",
    "system_views",
    "system_virtual_schema",
];

/// Names of all non-system keyspaces, sorted
pub fn user_keyspace_names(session: &Session) -> Vec<String> {
    let state = session.get_cluster_state();
    let mut names: Vec<String> = state
        .keyspaces_iter()
        .map(|(name, _)| name.to_string())
        .filter(|name| !SYSTEM_KEYSPACES.contains(&name.as_str()))
        .collect();
    names.sort();
    names
}

/// Names of all tables and materialized views in a keyspace, sorted.
/// Returns an empty list for unknown keyspaces.
pub fn keyspace_table_names(session: &Session, keyspace: &str) -> Vec<String> {
    let state = session.get_cluster_state();
    let Some(ks) = state.get_keyspace(keyspace) else {
        return Vec::new();
    };
    let mut names: Vec<String> = ks.tables.keys().chain(ks.views.keys()).cloned().collect();
    names.sort();
    names
}

/// Looks up the metadata of a table or materialized view
pub(crate) fn table_metadata(session: &Session, keyspace: &str, table: &str) -> Option<Table> {
    let state = session.get_cluster_state();
    let ks = state.get_keyspace(keyspace)?;
    ks.tables
        .get(table)
        .cloned()
        .or_else(|| ks.views.get(table).map(|view| view.view_metadata.clone()))
}
