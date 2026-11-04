//! Writing a table, under the [`bars`](crate::bars) layout if it is a table of bars:
//!
//! ```ignore
//! write_options(&session, DEFAULT_ROW_BLOCK)
//!     .blocking(&runtime)
//!     .write(file, arrays(&session, &schema, batches)?)?;
//! ```

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use vortex::array::iter::{ArrayIterator, ArrayIteratorAdapter};
use vortex::arrow::ArrowSessionExt;
use vortex::error::VortexResult;
use vortex::file::{VortexWriteOptions, WriteOptionsSessionExt, WriteStrategyBuilder};
use vortex::session::VortexSession;

use crate::bars::BarsStrategy;

/// Rows per chunk (row block) by default: 65,536, the row group size of the Parquet files the
/// bars come from.
pub const DEFAULT_ROW_BLOCK: usize = 65_536;

/// The options to write a table in `row_block`-row chunks, a table of bars under the bars layout
/// ([`BarsStrategy`]).
///
/// The strategy is Vortex's default on `session`, which must enable [`crate::edition`]: its
/// compressor may then use blocked FoR, blocked bit-packing and DecimalByteParts with lower
/// parts, and each of a bar's parts is a column of its own, with its own zone map. Vortex's
/// default also coalesces blocks up to about 1 MiB, which would merge the narrow columns into
/// bigger chunks, so that is turned off: every chunk is one row block.
pub fn write_options(session: &VortexSession, row_block: usize) -> VortexWriteOptions {
    let strategy = WriteStrategyBuilder::from_session(session)
        .with_row_block_size(row_block)
        .with_data_block_target_bytes(None)
        .build();
    session
        .write_options()
        .with_strategy(Arc::new(BarsStrategy::new(strategy)))
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
