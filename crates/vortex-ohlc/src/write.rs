//! Writing a table of bars stored as a [`Plan`] says:
//!
//! ```ignore
//! write_options(&session, &source, &plan, DEFAULT_ROW_BLOCK)?
//!     .blocking(&runtime)
//!     .write(file, stored_arrays(&session, &source, &plan, batches)?)?;
//! ```

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Schema};
use vortex::array::iter::{ArrayIterator, ArrayIteratorAdapter};
use vortex::arrow::ArrowSessionExt;
use vortex::dtype::DecimalDType;
use vortex::error::{vortex_bail, vortex_err, VortexResult};
use vortex::file::{VortexWriteOptions, WriteOptionsSessionExt, WriteStrategyBuilder};
use vortex::session::VortexSession;

use crate::bars::BarsStrategy;
use crate::ohlc::Price;
use crate::plan::{apply_plan, planned_schema, Plan, Transform};

/// Rows per chunk (row block) by default: 65,536, the row group size of the Parquet files the
/// bars come from.
pub const DEFAULT_ROW_BLOCK: usize = 65_536;

/// The options to write `source`'s table, stored as `plan` says, in `row_block`-row chunks.
///
/// The strategy is Vortex's default on `session`, which must enable [`crate::edition`]: its
/// compressor may then use blocked FoR, blocked bit-packing and DecimalByteParts with lower parts,
/// and each part of the [`OHLC`](crate::plan::OHLC) struct is a column of its own, with its own
/// zone map. Vortex's default also coalesces blocks up to about 1 MiB, which would merge the
/// narrow columns into bigger chunks, so that is turned off: every chunk is one row block.
///
/// When the plan stores the prices as parts, the table goes under the [`bars`](crate::bars)
/// layout, which reads them back as the source's prices, and the file has no statistics: Vortex
/// computes them over the columns as stored, which the layout's dtype doesn't have.
pub fn write_options(
    session: &VortexSession,
    source: &Schema,
    plan: &Plan,
    row_block: usize,
) -> VortexResult<VortexWriteOptions> {
    let strategy = WriteStrategyBuilder::from_session(session)
        .with_row_block_size(row_block)
        .with_data_block_target_bytes(None)
        .build();
    let open = Price::Open.name();
    if !matches!(plan.get(open), Some(Transform::Ohlc(_))) {
        return Ok(session.write_options().with_strategy(strategy));
    }
    let field = source
        .field_with_name(open)
        .map_err(|e| vortex_err!("{e}"))?;
    let DataType::Decimal128(precision, scale) = *field.data_type() else {
        vortex_bail!("{open} is {}, not a Decimal128", field.data_type());
    };
    let price = DecimalDType::new(precision, scale);
    Ok(session
        .write_options()
        .with_strategy(Arc::new(BarsStrategy::new(strategy, price)))
        .with_file_statistics(vec![]))
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
