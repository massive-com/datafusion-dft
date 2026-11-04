//! Writing a table, under the [`bars`](crate::bars) layout if it is a table of bars:
//!
//! ```ignore
//! write_options(&session, DEFAULT_ROW_BLOCK, true)
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

use crate::bars::BarsStrategy;
use crate::scaled::DecimalPow10Scheme;

/// Rows per chunk (row block) by default: 65,536, the row group size of the Parquet files the
/// bars come from.
pub const DEFAULT_ROW_BLOCK: usize = 65_536;

/// The options to write a table in `row_block`-row chunks, a table of bars under the bars layout
/// ([`BarsStrategy`]), with its dollar volume stored as a residual if `dollar_volume_residual`
/// and the table can (see [`BarsStrategy::with_dollar_volume_residual`] for the trade).
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
    dollar_volume_residual: bool,
) -> VortexWriteOptions {
    let compressor = BtrBlocksCompressorBuilder::from_session(session)
        .exclude_schemes([DecimalScheme::v1().id()])
        .with_new_scheme(&DecimalPow10Scheme);
    let strategy = WriteStrategyBuilder::from_session(session)
        .with_btrblocks_builder(compressor)
        .with_row_block_size(row_block)
        .with_data_block_target_bytes(None)
        .build();
    let bars = BarsStrategy::new(strategy, row_block as u64)
        .with_dollar_volume_residual(dollar_volume_residual);
    session.write_options().with_strategy(Arc::new(bars))
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
