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

//! Tests for the Cassandra catalog integration. Tests that query data require a running
//! Cassandra instance in a container named `cassandra` (seeding is done via `docker exec`):
//!
//! ```bash
//! docker run -d --name cassandra -p 9042:9042 \
//!   -e MAX_HEAP_SIZE=512M -e HEAP_NEWSIZE=128M \
//!   cassandra:5.0
//! ```
//!
//! (The heap caps keep Cassandra from being OOM-killed on memory-constrained machines.)
//!
//! Cassandra takes a minute or two to accept connections after starting; wait until
//! `docker exec cassandra cqlsh -e "DESCRIBE KEYSPACES"` succeeds before running the tests.

use assert_cmd::Command;

use crate::{cli_cases::contains_str, config::TestConfigBuilder};

const CASSANDRA_CONTAINER: &str = "cassandra";
const CASSANDRA_CONTACT_POINT: &str = "localhost:9042";

/// Run a CQL script inside the Cassandra container
fn cassandra_exec(script: &str) {
    let output = std::process::Command::new("docker")
        .args(["exec", "-i", CASSANDRA_CONTAINER, "cqlsh", "-e", script])
        .output()
        .expect("Failed to run cqlsh via docker exec");
    assert!(
        output.status.success(),
        "Cassandra script failed: {script}\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Create a `users` table with two rows in `keyspace`. Each test seeds its own keyspace so
/// that concurrently running tests do not interfere with each other.
fn seed_cassandra(keyspace: &str) {
    cassandra_exec(&format!(
        "CREATE KEYSPACE IF NOT EXISTS {keyspace} WITH replication = \
         {{'class': 'SimpleStrategy', 'replication_factor': 1}}; \
         DROP TABLE IF EXISTS {keyspace}.users; \
         CREATE TABLE {keyspace}.users (id int PRIMARY KEY, name text); \
         INSERT INTO {keyspace}.users (id, name) VALUES (1, 'alice'); \
         INSERT INTO {keyspace}.users (id, name) VALUES (2, 'bob');"
    ));
}

fn cassandra_config(keyspace: Option<&str>) -> crate::config::TestConfig {
    let mut config_builder = TestConfigBuilder::default();
    config_builder.with_cassandra(
        "cli",
        "cassandra",
        &[CASSANDRA_CONTACT_POINT],
        None,
        None,
        keyspace,
        false,
    );
    config_builder.build("my_config.toml")
}

#[test]
fn test_cassandra_config_parsing() {
    let config = cassandra_config(Some("dft_cassandra_parsing"));
    let app_config = datafusion_dft::config::create_config(config.path, &[]);
    let cassandra_configs = app_config
        .cli
        .execution
        .cassandra
        .expect("Cassandra config missing");
    assert_eq!(cassandra_configs.len(), 1);
    let cassandra_config = &cassandra_configs[0];
    assert_eq!(cassandra_config.name, "cassandra");
    assert_eq!(
        cassandra_config.contact_points,
        vec![CASSANDRA_CONTACT_POINT.to_string()]
    );
    assert_eq!(cassandra_config.user, None);
    assert_eq!(cassandra_config.password, None);
    assert_eq!(
        cassandra_config.keyspace.as_deref(),
        Some("dft_cassandra_parsing")
    );

    let options = cassandra_config.to_connection_options();
    assert_eq!(
        options.contact_points,
        vec![CASSANDRA_CONTACT_POINT.to_string()]
    );
    assert_eq!(options.user, None);
    assert_eq!(options.password, None);
    assert_eq!(options.connection_timeout_ms, None);
    assert!(!options.single_endpoint);
}

#[test]
fn test_cassandra_config_parsing_single_endpoint() {
    let mut config_builder = TestConfigBuilder::default();
    config_builder.with_cassandra(
        "cli",
        "cassandra",
        &[CASSANDRA_CONTACT_POINT],
        None,
        None,
        None,
        true,
    );
    let config = config_builder.build("my_config.toml");
    let app_config = datafusion_dft::config::create_config(config.path, &[]);
    let cassandra_configs = app_config
        .cli
        .execution
        .cassandra
        .expect("Cassandra config missing");
    assert!(cassandra_configs[0].single_endpoint);
    assert!(cassandra_configs[0].to_connection_options().single_endpoint);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cassandra_query() {
    seed_cassandra("dft_cassandra_query");
    let config = cassandra_config(Some("dft_cassandra_query"));

    let assert = Command::cargo_bin("dft")
        .unwrap()
        .arg("--config")
        .arg(config.path)
        .arg("-c")
        .arg("SELECT id, name FROM cassandra.dft_cassandra_query.users ORDER BY id")
        .assert()
        .success();

    let expected = r#"
+----+-------+
| id | name  |
+----+-------+
| 1  | alice |
| 2  | bob   |
+----+-------+
"#;

    assert.stdout(contains_str(expected));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cassandra_keyspace_discovery() {
    seed_cassandra("dft_cassandra_discovery");
    // No keyspace configured, so all non-system keyspaces are discovered as schemas
    let config = cassandra_config(None);

    let assert = Command::cargo_bin("dft")
        .unwrap()
        .arg("--config")
        .arg(config.path)
        .arg("-c")
        .arg("SELECT id, name FROM cassandra.dft_cassandra_discovery.users ORDER BY id")
        .assert()
        .success();

    let expected = r#"
+----+-------+
| id | name  |
+----+-------+
| 1  | alice |
| 2  | bob   |
+----+-------+
"#;

    assert.stdout(contains_str(expected));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cassandra_filter_and_limit() {
    seed_cassandra("dft_cassandra_filter");
    let config = cassandra_config(Some("dft_cassandra_filter"));

    // Equality on the full partition key is pushed down to CQL
    let assert = Command::cargo_bin("dft")
        .unwrap()
        .arg("--config")
        .arg(config.path)
        .arg("-c")
        .arg("SELECT name FROM cassandra.dft_cassandra_filter.users WHERE id = 2")
        .assert()
        .success();

    let expected = r#"
+------+
| name |
+------+
| bob  |
+------+
"#;

    assert.stdout(contains_str(expected));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cassandra_clustering_range_pushdown() {
    cassandra_exec(
        "CREATE KEYSPACE IF NOT EXISTS dft_cassandra_range WITH replication = \
         {'class': 'SimpleStrategy', 'replication_factor': 1}; \
         DROP TABLE IF EXISTS dft_cassandra_range.ticks; \
         CREATE TABLE dft_cassandra_range.ticks \
         (id int, seq bigint, val text, PRIMARY KEY (id, seq)); \
         INSERT INTO dft_cassandra_range.ticks (id, seq, val) VALUES (1, 1, 'a'); \
         INSERT INTO dft_cassandra_range.ticks (id, seq, val) VALUES (1, 2, 'b'); \
         INSERT INTO dft_cassandra_range.ticks (id, seq, val) VALUES (1, 3, 'c');",
    );
    let config = cassandra_config(Some("dft_cassandra_range"));

    // Partition key equality plus a clustering key range is pushed down to CQL
    let assert = Command::cargo_bin("dft")
        .unwrap()
        .arg("--config")
        .arg(config.path)
        .arg("-c")
        .arg(
            "SELECT seq, val FROM cassandra.dft_cassandra_range.ticks \
             WHERE id = 1 AND seq >= 2 AND seq < 3",
        )
        .assert()
        .success();

    let expected = r#"
+-----+-----+
| seq | val |
+-----+-----+
| 2   | b   |
+-----+-----+
"#;

    assert.stdout(contains_str(expected));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cassandra_information_schema() {
    seed_cassandra("dft_cassandra_info");
    let config = cassandra_config(Some("dft_cassandra_info"));

    let assert = Command::cargo_bin("dft")
        .unwrap()
        .arg("--config")
        .arg(config.path)
        .arg("-c")
        .arg(
            "SELECT table_catalog, table_schema, table_name FROM information_schema.tables \
             WHERE table_catalog = 'cassandra' AND table_schema = 'dft_cassandra_info'",
        )
        .assert()
        .success();

    let expected = r#"
+---------------+--------------------+------------+
| table_catalog | table_schema       | table_name |
+---------------+--------------------+------------+
| cassandra     | dft_cassandra_info | users      |
+---------------+--------------------+------------+
"#;

    assert.stdout(contains_str(expected));
}
