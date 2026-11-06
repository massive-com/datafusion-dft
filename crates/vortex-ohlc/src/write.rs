//! Writing a table, under the [`bars`](crate::bars) layout if it is a table of bars:
//!
//! ```ignore
//! write_options(&session, DEFAULT_ROW_BLOCK, vec![ProductResidual::dollar_volume()])
//!     .blocking(&runtime)
//!     .write(file, arrays(&session, &schema, batches)?)?;
//! ```

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use vortex::array::iter::{ArrayIterator, ArrayIteratorAdapter};
use vortex::arrow::ArrowSessionExt;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::error::VortexResult;
use vortex::file::{VortexWriteOptions, WriteOptionsSessionExt, WriteStrategyBuilder};
use vortex::session::VortexSession;
use vortex_btrblocks::schemes::decimal::DecimalScheme;
use vortex_btrblocks::SchemeExt;

use crate::bars::{BarsStrategy, ProductResidual};
use crate::scaled::DecimalPow10Scheme;

/// Rows per chunk (row block) by default: 65,536, the row group size of the Parquet files the
/// bars come from.
pub const DEFAULT_ROW_BLOCK: usize = 65_536;

/// The options to write a table in `row_block`-row chunks, a table of bars under the bars layout
/// ([`BarsStrategy`]), with the columns of `residuals` it has stored as residuals (see
/// [`BarsStrategy::with_product_residuals`], and [`ProductResidual::dollar_volume`] for the
/// trade).
///
/// The strategy is Vortex's default on `session`, which must enable [`crate::edition`]: its
/// compressor may then use our encodings, blocked FoR, blocked bit-packing and DecimalByteParts
/// with lower parts, and each of a bar's parts is a column of its own, with its own zone map.
/// Vortex's default also coalesces blocks up to about 1 MiB, which would merge the narrow columns
/// into bigger chunks, so that is turned off: every chunk is one row block. Its decimal scheme
/// gives way to [`DecimalPow10Scheme`], which stores decimals as it does, but without the
/// trailing zeros a chunk's values all have.
pub fn write_options(
    session: &VortexSession,
    row_block: usize,
    residuals: Vec<ProductResidual>,
) -> VortexWriteOptions {
    write_options_zoned(session, row_block, row_block as u64, residuals)
}

/// [`write_options`], with the bars layout's zones (the prices' min and max, and the residuals'
/// targets') every `zone_len` rows instead of once a chunk; the other columns' zone maps stay
/// once a chunk.
pub fn write_options_zoned(
    session: &VortexSession,
    row_block: usize,
    zone_len: u64,
    residuals: Vec<ProductResidual>,
) -> VortexWriteOptions {
    let compressor = with_decimal_pow10(BtrBlocksCompressorBuilder::from_session(session));
    let strategy = WriteStrategyBuilder::from_session(session)
        .with_btrblocks_builder(compressor)
        .with_row_block_size(row_block)
        .with_data_block_target_bytes(None)
        .build();
    let bars = BarsStrategy::new(strategy, zone_len).with_product_residuals(residuals);
    session.write_options().with_strategy(Arc::new(bars))
}

/// `compressor` with [`DecimalPow10Scheme`] in the place of Vortex's decimal scheme, for a writer
/// that builds its own strategy and wraps it in a [`BarsStrategy`].
pub fn with_decimal_pow10(compressor: BtrBlocksCompressorBuilder) -> BtrBlocksCompressorBuilder {
    compressor
        .exclude_schemes([DecimalScheme::v1().id()])
        .with_new_scheme(&DecimalPow10Scheme)
}

/// `batches` with `schema`, as the arrays to write with [`write_options`].
pub fn arrays<I>(
    session: &VortexSession,
    schema: &Arc<Schema>,
    batches: I,
) -> VortexResult<impl ArrayIterator + Send + 'static>
where
    I: IntoIterator<Item = RecordBatch>,
    I::IntoIter: Send + 'static,
{
    let dtype = session.arrow().from_arrow_schema(schema)?;
    let (session, schema) = (session.clone(), Arc::clone(schema));
    let arrays = batches
        .into_iter()
        .map(move |batch| session.arrow().from_arrow_record_batch(batch, &schema));
    Ok(ArrayIteratorAdapter::new(dtype, arrays))
}
