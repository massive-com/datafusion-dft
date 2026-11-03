//! Writing a table of bars stored as a [`Plan`] says, with the plan recorded in the file, and
//! reading the plan back:
//!
//! ```ignore
//! write_options(&session, &source, &plan, DEFAULT_ROW_BLOCK)
//!     .blocking(&runtime)
//!     .write(file, stored_arrays(&session, &plan, &source, batches)?)?;
//! ```

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use vortex::array::iter::{ArrayIterator, ArrayIteratorAdapter};
use vortex::arrow::ArrowSessionExt;
use vortex::buffer::ByteBuffer;
use vortex::error::VortexResult;
use vortex::file::{VortexFile, VortexWriteOptions, WriteOptionsSessionExt, WriteStrategyBuilder};
use vortex::session::VortexSession;

use crate::plan::{
    apply_plan, plan_from_json, planned_schema, source_schema, transforms_json, Plan,
    TRANSFORMS_KEY,
};

/// Rows per chunk (row block) by default: 65,536, the row group size of the Parquet files the
/// bars come from.
pub const DEFAULT_ROW_BLOCK: usize = 65_536;

/// The options to write a table with `source`'s schema, stored as `plan` says, in
/// `row_block`-row chunks, with the plan recorded in the file (see [`transforms_json`]).
///
/// The strategy is Vortex's default on `session`, which must enable [`crate::edition`]: its
/// compressor may then use blocked FoR, blocked bit-packing and DecimalByteParts with lower parts,
/// and each part of the [`OHLC`](crate::plan::OHLC) struct is a column of its own, with its own
/// zone map. Vortex's default also coalesces blocks up to about 1 MiB, which would merge the
/// narrow columns into bigger chunks, so that is turned off: every chunk is one row block.
pub fn write_options(
    session: &VortexSession,
    source: &Schema,
    plan: &Plan,
    row_block: usize,
) -> VortexWriteOptions {
    let transforms =
        serde_json::to_vec(&transforms_json(plan, source)).expect("a plan serializes to JSON");
    session
        .write_options()
        .with_strategy(
            WriteStrategyBuilder::from_session(session)
                .with_row_block_size(row_block)
                .with_data_block_target_bytes(None)
                .build(),
        )
        .with_metadata_segment(TRANSFORMS_KEY, ByteBuffer::copy_from(transforms))
}

/// `batches` with `source`'s schema, stored as `plan` says (see [`apply_plan`]), as the arrays
/// to write with [`write_options`].
pub fn stored_arrays<I>(
    session: &VortexSession,
    plan: &Plan,
    source: &Schema,
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

/// The plan `file` records, if it records one of this version. A file only loads its metadata
/// when opened with `include_metadata`.
pub fn file_plan(file: &VortexFile) -> Option<Plan> {
    plan_from_json(&file_record(file)?)
}

/// The schema of the source `file` was written from, given `stored`, its schema as written (see
/// [`source_schema`]); `None` unless the file records a plan of this version that fits `stored`.
/// A file only loads its metadata when opened with `include_metadata`.
pub fn file_source_schema(file: &VortexFile, stored: &Schema) -> Option<Schema> {
    source_schema(&file_record(file)?, stored)
}

/// The record of its plan `file` holds (see [`transforms_json`]), if any.
fn file_record(file: &VortexFile) -> Option<serde_json::Value> {
    let json = file.metadata_segment(TRANSFORMS_KEY)?;
    serde_json::from_slice(json.as_ref()).ok()
}
