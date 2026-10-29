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

use std::collections::HashMap;
use std::sync::Arc;

use datafusion_dft::config::AppConfig;

use crate::extension_cases::TestExecution;

#[tokio::test(flavor = "multi_thread")]
async fn test_vortex_extension_loads() {
    // This test verifies that the Vortex extension loads successfully
    // and is registered with the ExecutionContext
    let _test_exec = TestExecution::new().await;

    // If we get here without panicking, the extension loaded successfully
    // TODO: Add test data and query tests once we have sample .vortex files
    // Example usage would be:
    // let cwd = std::env::current_dir().unwrap();
    // let path = Url::from_file_path(cwd.join("data/vortex/sample_data.vortex")).unwrap();
    //
    // let test_exec = test_exec
    //     .with_setup(&format!(
    //         "CREATE EXTERNAL TABLE v STORED AS VORTEX LOCATION '{}';",
    //         path
    //     ))
    //     .await;
    //
    // let output = test_exec
    //     .run_and_format("SELECT * FROM v LIMIT 10")
    //     .await;
}

/// Write a small `.vortex` file into `dir` and return a query that filters it.
async fn filter_query(test_exec: &TestExecution, dir: &std::path::Path) -> String {
    let path = dir.join("ids.vortex");
    let path = path.to_str().unwrap();
    test_exec
        .run(&format!(
            "COPY (SELECT value AS id FROM generate_series(1, 1000)) TO '{path}' STORED AS VORTEX"
        ))
        .await
        .unwrap();
    format!("EXPLAIN SELECT count(*) FROM '{path}' WHERE id = 5")
}

/// Whether the filter in `explain` is evaluated by DataFusion rather than inside the Vortex scan.
fn filters_after_scan(explain: &[String]) -> bool {
    explain.iter().any(|line| line.contains("FilterExec"))
}

#[tokio::test(flavor = "multi_thread")]
async fn test_vortex_session_options_can_be_set() {
    let dir = tempfile::tempdir().unwrap();
    let test_exec = TestExecution::new().await;
    let query = filter_query(&test_exec, dir.path()).await;

    assert!(!filters_after_scan(&test_exec.run_and_format(&query).await));

    test_exec
        .run("SET vortex.predicate_pushdown = false")
        .await
        .unwrap();
    assert!(filters_after_scan(&test_exec.run_and_format(&query).await));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_vortex_options_from_config() {
    let mut config = AppConfig::default();
    config.cli.execution.datafusion = Some(HashMap::from([(
        "vortex.predicate_pushdown".to_string(),
        "false".to_string(),
    )]));
    let dir = tempfile::tempdir().unwrap();
    let test_exec = TestExecution::new_with_config(config).await;
    let query = filter_query(&test_exec, dir.path()).await;

    assert!(filters_after_scan(&test_exec.run_and_format(&query).await));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_vortex_create_external_table() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ids.vortex");
    let path = path.to_str().unwrap();
    let test_exec = TestExecution::new().await;
    test_exec
        .run(&format!(
            "COPY (SELECT value AS id FROM generate_series(1, 1000)) TO '{path}' STORED AS VORTEX"
        ))
        .await
        .unwrap();
    test_exec
        .run(&format!(
            "CREATE EXTERNAL TABLE ids STORED AS VORTEX LOCATION '{path}'"
        ))
        .await
        .unwrap();
    test_exec
        .run(&format!(
            "CREATE EXTERNAL TABLE ids_unpushed STORED AS VORTEX LOCATION '{path}' \
             OPTIONS (predicate_pushdown 'false')"
        ))
        .await
        .unwrap();

    let output = test_exec
        .run_and_format("SELECT count(*) AS n FROM ids WHERE id > 990")
        .await;
    assert!(
        output.iter().any(|line| line == "| 10 |"),
        "unexpected output: {output:?}"
    );

    // Table options reach the Vortex format.
    let explain = "EXPLAIN SELECT count(*) FROM {table} WHERE id = 5";
    assert!(!filters_after_scan(
        &test_exec
            .run_and_format(&explain.replace("{table}", "ids"))
            .await
    ));
    assert!(filters_after_scan(
        &test_exec
            .run_and_format(&explain.replace("{table}", "ids_unpushed"))
            .await
    ));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_vortex_keeps_default_table_factories() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ids.csv");
    let path = path.to_str().unwrap();
    let test_exec = TestExecution::new().await;
    test_exec
        .run(&format!("COPY (SELECT 1 AS id) TO '{path}' STORED AS CSV"))
        .await
        .unwrap();

    test_exec
        .run(&format!(
            "CREATE EXTERNAL TABLE ids STORED AS CSV LOCATION '{path}' OPTIONS ('format.has_header' 'true')"
        ))
        .await
        .unwrap();
    let output = test_exec.run_and_format("SELECT id FROM ids").await;
    assert!(
        output.iter().any(|line| line == "| 1  |"),
        "unexpected output: {output:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_vortex_local_scan_reports_reads() {
    let dir = tempfile::tempdir().unwrap();
    // The space checks that the file's object store path is decoded before it is opened.
    let path = dir.path().join("local ids.vortex");
    let path = path.to_str().unwrap();
    let test_exec = TestExecution::new().await;
    test_exec
        .run(&format!(
            "COPY (SELECT value AS id FROM generate_series(1, 1000)) TO '{path}' STORED AS VORTEX"
        ))
        .await
        .unwrap();

    let query = format!("SELECT count(*) AS n FROM '{path}' WHERE id > 990");
    let output = test_exec.run_and_format(&query).await;
    assert!(
        output.iter().any(|line| line == "| 10 |"),
        "unexpected output: {output:?}"
    );

    assert!(bytes_scanned(&test_exec, &query).await > 0);
}

/// The bytes `query`'s scans read, as `--analyze` reports them.
async fn bytes_scanned(test_exec: &TestExecution, query: &str) -> usize {
    let plan = test_exec
        .session_ctx()
        .sql(query)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let task_ctx = test_exec.session_ctx().task_ctx();
    datafusion::physical_plan::collect(Arc::clone(&plan), task_ctx)
        .await
        .unwrap();
    let stats = datafusion_app::stats::collect_plan_io_stats(plan).expect("plan has a scan");
    stats.bytes_scanned().expect("scan reports its reads")
}

#[tokio::test(flavor = "multi_thread")]
async fn test_vortex_object_store_scan_reports_reads() {
    let test_exec = TestExecution::new().await;
    let url = url::Url::parse("memory://bucket").unwrap();
    test_exec.session_ctx().register_object_store(
        &url,
        Arc::new(datafusion::object_store::memory::InMemory::new()),
    );
    test_exec
        .run(
            "COPY (SELECT value AS id FROM generate_series(1, 1000)) \
             TO 'memory://bucket/ids.vortex' STORED AS VORTEX",
        )
        .await
        .unwrap();

    let query = "SELECT count(*) AS n FROM 'memory://bucket/ids.vortex' WHERE id > 990";
    let output = test_exec.run_and_format(query).await;
    assert!(
        output.iter().any(|line| line == "| 10 |"),
        "unexpected output: {output:?}"
    );
    assert!(bytes_scanned(&test_exec, query).await > 0);
}
