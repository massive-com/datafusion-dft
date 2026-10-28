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
use vortex::file::{VortexFile, VortexWriteOptions, WriteOptionsSessionExt};
use vortex::session::VortexSession;

use crate::plan::{
    apply_plan, plan_from_json, planned_schema, transforms_json, Plan, TRANSFORMS_KEY,
};
use crate::strategy::write_strategy;

/// The options to write a table with `source`'s schema, stored as `plan` says, in
/// `row_block`-row chunks: [`write_strategy`], and the plan recorded in the file (see
/// [`transforms_json`]). The session must enable [`crate::edition`].
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
        .with_strategy(write_strategy(source, plan, row_block))
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
    let json = file.metadata_segment(TRANSFORMS_KEY)?;
    plan_from_json(&serde_json::from_slice(json.as_ref()).ok()?)
}
