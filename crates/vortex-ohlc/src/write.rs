//! Writing a table of bars stored as a [`Plan`] says:
//!
//! ```ignore
//! write_options(&session, DEFAULT_ROW_BLOCK)
//!     .blocking(&runtime)
//!     .write(file, stored_arrays(&session, &source, &plan, batches)?)?;
//! ```

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use vortex::array::iter::{ArrayIterator, ArrayIteratorAdapter};
use vortex::arrow::ArrowSessionExt;
use vortex::error::VortexResult;
use vortex::file::{VortexWriteOptions, WriteOptionsSessionExt, WriteStrategyBuilder};
use vortex::session::VortexSession;

use crate::plan::{apply_plan, planned_schema, Plan};

/// Rows per chunk (row block) by default: 65,536, the row group size of the Parquet files the
/// bars come from.
pub const DEFAULT_ROW_BLOCK: usize = 65_536;

/// The options to write in `row_block`-row chunks.
///
/// The strategy is Vortex's default on `session`, which must enable [`crate::edition`]: its
/// compressor may then use blocked FoR, blocked bit-packing and DecimalByteParts with lower parts,
/// and each part of the [`OHLC`](crate::plan::OHLC) struct is a column of its own, with its own
/// zone map. Vortex's default also coalesces blocks up to about 1 MiB, which would merge the
/// narrow columns into bigger chunks, so that is turned off: every chunk is one row block.
pub fn write_options(session: &VortexSession, row_block: usize) -> VortexWriteOptions {
    session.write_options().with_strategy(
        WriteStrategyBuilder::from_session(session)
            .with_row_block_size(row_block)
            .with_data_block_target_bytes(None)
            .build(),
    )
}

/// `batches` with `source`'s schema, stored as `plan` says (see [`apply_plan`]), as the arrays
/// to write with [`write_options`].
pub fn stored_arrays<I>(
    session: &VortexSession,
    source: &Schema,
    plan: &Plan,
    batches: I,
) -> VortexResult<impl ArrayIterator + Send + 'static>
where
    I: IntoIterator<Item = RecordBatch>,
    I::IntoIter: Send + 'static,
{
    let planned = planned_schema(plan, source);
    let dtype = session.arrow().from_arrow_schema(&planned)?;
    let (session, plan) = (session.clone(), plan.clone());
    let arrays = batches.into_iter().map(move |batch| {
        let batch = apply_plan(&plan, &planned, batch);
        session.arrow().from_arrow_record_batch(batch, &planned)
    });
    Ok(ArrayIteratorAdapter::new(dtype, arrays))
}
