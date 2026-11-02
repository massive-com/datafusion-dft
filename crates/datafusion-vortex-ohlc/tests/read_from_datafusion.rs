//! A file written by vortex-ohlc, read from DataFusion as dft reads it (see `PrunedTable::over`):
//! the file is the source again, and filters on the rebuilt prices and the rescaled columns prune
//! its zones.

use std::sync::Arc;

use arrow::compute::concat_batches;
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
    let table = PrunedTable::over(&listing, &state, pruner).await?;
    ctx.register_table("bars", Arc::new(table))?;

    let read = ctx.sql("SELECT * FROM bars").await?.collect().await?;
    assert_eq!(read[0].schema().fields(), source.schema().fields());
    let read = concat_batches(&source.schema(), &read)?;
    assert_eq!(read.columns(), source.columns());

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
