//! Vortex tables pruned before their scans start, as dft has them: a file written by vortex-ohlc,
//! read with its prices rebuilt (see `PrunedTable::over`), and tables queried by path or created
//! with `CREATE EXTERNAL TABLE`.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::compute::{cast, concat_batches};
use arrow::datatypes::DataType;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::Result;
use datafusion::datasource::file_format::FileFormatFactory;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::datasource::physical_plan::FileScanConfig;
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_vortex_ohlc::vortex_format_factory;
use datafusion_vortex_prune::{enable_url_table, PrunedTable, PrunedTableFactory, ZonePruner};
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_datafusion::{VortexAccessPlan, VortexFormatFactory};

use test_util::{source, source_plan, write};

/// For each file of each Vortex scan in `plan`, whether it was given a selection.
fn selected(plan: &Arc<dyn ExecutionPlan>) -> Vec<bool> {
    let mut selected = Vec::new();
    plan.apply(|node| {
        let conf = node
            .downcast_ref::<DataSourceExec>()
            .and_then(|exec| exec.data_source().downcast_ref::<FileScanConfig>());
        let groups = conf.iter().flat_map(|conf| conf.file_groups.iter());
        for file in groups.flat_map(|group| group.iter()) {
            let plan = file.extensions.get::<VortexAccessPlan>();
            selected.push(plan.and_then(VortexAccessPlan::selection).is_some());
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    selected
}

#[tokio::test]
async fn a_vortex_ohlc_file_is_read_and_pruned() -> Result<()> {
    let source = source();
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("bars.vortex");
    std::fs::write(&path, write(&source, &source_plan(&source)))?;

    // One partition, so the rows come back in the file's order.
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
    let state = ctx.state();
    let session = VortexSession::default();
    let options = ListingOptions::new(vortex_format_factory(session.clone()).default())
        .with_file_extension(".vortex");
    let url = ListingTableUrl::parse(path.to_str().unwrap())?;
    let config = ListingTableConfig::new(url)
        .with_listing_options(options)
        .infer_schema(&state)
        .await?;
    let listing = ListingTable::try_new(config)?;
    let pruner = Arc::new(ZonePruner::new(session));
    let table = PrunedTable::over(&listing, &state, pruner)?;
    ctx.register_table("bars", Arc::new(table))?;

    // The source's values at the scales they are stored at: the prices at their parts', and the
    // volumes in whole shares.
    let read = ctx.sql("SELECT * FROM bars").await?.collect().await?;
    let schema = read[0].schema();
    let types: Vec<(&str, &DataType)> = schema
        .fields()
        .iter()
        .map(|field| (field.name().as_str(), field.data_type()))
        .collect();
    let price = DataType::Decimal128(18, 2);
    assert_eq!(
        types,
        [
            ("ticker", &DataType::Utf8View),
            ("open", &price),
            ("high", &price),
            ("low", &price),
            ("close", &price),
            ("volume", &DataType::Decimal128(38, 0)),
            ("transactions", &DataType::Int64),
        ]
    );
    let read = concat_batches(&schema, &read)?;
    for (read, source) in read.columns().iter().zip(source.columns()) {
        assert_eq!(read, &cast(source, read.data_type())?);
    }

    // Only the last ticker's highs are this high and only the first's lows this low, so each
    // scan reads the one zone of three holding them, pruned on the bounds of the parts the prices
    // are rebuilt from. Only the last ticker's volumes are wider than an i64, and stored rescaled.
    for filter in ["high > 360", "low < 20", "volume > 9223372036854775807"] {
        let query = ctx
            .sql(&format!("SELECT * FROM bars WHERE {filter}"))
            .await?;
        let plan = query.clone().create_physical_plan().await?;
        assert_eq!(selected(&plan), [true], "{filter}");
        assert_eq!(query.count().await?, 500, "{filter}");
    }
    Ok(())
}

#[tokio::test]
async fn tables_by_path_and_external_are_pruned() -> Result<()> {
    let session = VortexSession::default();
    let config = SessionConfig::new().with_extension(Arc::new(ZonePruner::new(session.clone())));
    let mut builder = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features();
    builder
        .file_formats()
        .get_or_insert_with(Vec::new)
        .push(Arc::new(VortexFormatFactory::new_with_session(session)));
    builder
        .table_factories()
        .get_or_insert_with(HashMap::new)
        .insert(
            "VORTEX".to_string(),
            Arc::new(PrunedTableFactory::default()),
        );
    let ctx = enable_url_table(SessionContext::new_with_state(builder.build()));
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("ids.vortex");
    let path = path.to_str().unwrap();
    let sql = |sql: String| {
        let ctx = ctx.clone();
        async move { ctx.sql(&sql).await }
    };
    // Enough rows for some zones.
    sql(format!(
        "COPY (SELECT value AS id FROM generate_series(1, 100000)) TO '{path}' STORED AS VORTEX"
    ))
    .await?
    .collect()
    .await?;
    sql(format!(
        "CREATE EXTERNAL TABLE ids STORED AS VORTEX LOCATION '{path}'"
    ))
    .await?
    .collect()
    .await?;

    for table in [format!("'{path}'"), "ids".to_string()] {
        // One zone of the file can match, so the scan reads that one only.
        let query = sql(format!("SELECT * FROM {table} WHERE id > 99000")).await?;
        let plan = query.clone().create_physical_plan().await?;
        assert_eq!(selected(&plan), [true], "{table}");
        assert_eq!(query.count().await?, 1000, "{table}");
        // Every zone can, so the scan reads them all as it would have.
        let query = sql(format!("SELECT * FROM {table} WHERE id > 1000")).await?;
        let plan = query.clone().create_physical_plan().await?;
        assert!(!selected(&plan).contains(&true), "{table}");
        assert_eq!(query.count().await?, 99000, "{table}");
    }
    Ok(())
}
