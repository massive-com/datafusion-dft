//! A file written by vortex-ohlc, read from DataFusion as dft reads it (see `PrunedTable::over`):
//! the source's values, the prices rebuilt from their parts, and filters on the prices and the
//! rescaled columns prune its zones.

use std::sync::Arc;

use arrow::compute::{cast, concat_batches};
use arrow::datatypes::DataType;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::datasource::file_format::FileFormatFactory;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::datasource::physical_plan::FileScanConfig;
use datafusion::datasource::source::DataSourceExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_vortex_ohlc::vortex_format_factory;
use datafusion_vortex_prune::{PrunedTable, ZonePruner};
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_datafusion::VortexAccessPlan;

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
async fn read_from_datafusion() -> datafusion::common::Result<()> {
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
