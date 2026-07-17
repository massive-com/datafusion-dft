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

//! DataFusion [`TableProvider`](datafusion::catalog::TableProvider) for
//! querying Apache Cassandra (and ScyllaDB) tables with SQL.
//!
//! Connect with [`connect`], discover keyspaces and tables with
//! [`user_keyspace_names`] / [`keyspace_table_names`], and register
//! [`CassandraTable`]s with a DataFusion `SessionContext`:
//!
//! ```ignore
//! let session = Arc::new(
//!     connect(&CassandraConnectionOptions {
//!         contact_points: vec!["localhost:9042".to_string()],
//!         ..Default::default()
//!     })
//!     .await?,
//! );
//! let table = CassandraTable::try_new(Arc::clone(&session), "ks", "users")?;
//! ctx.register_table("users", Arc::new(table))?;
//! ```
//!
//! Projection and `LIMIT` are pushed into the generated CQL, as are equality
//! filters that cover the full partition key. All other filtering happens in
//! DataFusion.

mod connection;
mod cql;
mod exec;
mod metadata;
mod provider;
mod types;

pub use connection::{connect, CassandraConnectionOptions};
pub use metadata::{keyspace_table_names, user_keyspace_names, SYSTEM_KEYSPACES};
pub use provider::CassandraTable;

/// Re-exported so downstream crates can name driver types (e.g.
/// [`scylla::client::session::Session`]) without their own dependency
pub use scylla;
