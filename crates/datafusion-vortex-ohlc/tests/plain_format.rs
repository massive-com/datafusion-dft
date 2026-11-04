//! A file written by vortex-ohlc, read by vortex-datafusion's own format with nothing of ours but
//! `vortex_ohlc::register` on its Vortex session: the bars layout is a Vortex component, so any
//! Vortex reader gets the source's prices, and its filters on them run inside Vortex.

use std::sync::Arc;

use arrow::compute::{cast, concat_batches};
use arrow::datatypes::DataType;
use datafusion::common::Result;
use datafusion::datasource::file_format::FileFormatFactory;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::physical_plan::displayable;
use datafusion::prelude::{SessionConfig, SessionContext};
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_datafusion::VortexFormatFactory;

use test_util::{source, write};

#[tokio::test]
async fn vortex_datafusion_reads_the_source() -> Result<()> {
    let source = source();
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("bars.vortex");
    std::fs::write(&path, write(&source))?;

    // One partition, so the rows come back in the file's order.
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
    let session = VortexSession::default();
    vortex_ohlc::register(&session);
    let format = VortexFormatFactory::new_with_session(session).default();
    let options = ListingOptions::new(format).with_file_extension(".vortex");
    let url = ListingTableUrl::parse(path.to_str().unwrap())?;
    let config = ListingTableConfig::new(url)
        .with_listing_options(options)
        .infer_schema(&ctx.state())
        .await?;
    ctx.register_table("bars", Arc::new(ListingTable::try_new(config)?))?;

    // The source's prices, as the source types them; the volumes at the scale they are stored at,
    // in whole shares.
    let read = ctx.sql("SELECT * FROM bars").await?.collect().await?;
    let schema = read[0].schema();
    let price = DataType::Decimal128(38, 4);
    for name in ["open", "high", "low", "close"] {
        assert_eq!(schema.field_with_name(name)?.data_type(), &price, "{name}");
    }
    let read = concat_batches(&schema, &read)?;
    for (read, source) in read.columns().iter().zip(source.columns()) {
        assert_eq!(read, &cast(source, read.data_type())?);
    }

    // Only the last ticker's highs are this high and only the first's lows this low. The filters
    // go to Vortex whole: no `FilterExec` is left above the scan.
    for filter in ["high > 360", "low < 20"] {
        let query = ctx
            .sql(&format!("SELECT * FROM bars WHERE {filter}"))
            .await?;
        let plan = query.clone().create_physical_plan().await?;
        let plan = displayable(plan.as_ref()).indent(true).to_string();
        assert!(!plan.contains("FilterExec"), "{filter}:\n{plan}");
        assert_eq!(query.count().await?, 500, "{filter}");
    }
    Ok(())
}
