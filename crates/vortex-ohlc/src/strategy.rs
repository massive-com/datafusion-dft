//! The write strategy: Vortex's default, with every decimal column, the integer columns the plan
//! packs ([`Transform::Packed`]) and each part of the [`OHLC`] struct compressed by Vortex's
//! compressor on the writer's session, whose edition (see [`crate::edition`]) lets it use blocked
//! FoR, blocked bit-packing and DecimalByteParts with lower parts. Each part of the OHLC struct is
//! a column of its own with its own zone map, so a scan can prune on the prices (see
//! datafusion-vortex-prune's `ZonePruner`).

use std::sync::Arc;

use arrow::datatypes::{DataType, Schema};
use vortex::compressor::BtrBlocksCompressor;
use vortex::dtype::FieldPath;
use vortex::file::WriteStrategyBuilder;
use vortex::layout::layouts::collect::CollectStrategy;
use vortex::layout::layouts::compressed::CompressingStrategy;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::layout::layouts::table::TableStrategy;
use vortex::layout::LayoutStrategy;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;

use crate::plan::{Plan, Transform, OHLC, OHLC_PARTS, OHLC_PRICES};

/// Rows per chunk (row block) by default: 65,536, the row group size of the Parquet files the
/// bars come from.
pub const DEFAULT_ROW_BLOCK: usize = 65_536;

/// The write strategy for `source` stored with `plan` (see [`crate::plan::apply_plan`]), in
/// `row_block`-row chunks, on `session`, the writing session, which must enable
/// [`crate::edition`]; any Vortex session reads the file.
pub fn write_strategy(
    session: &VortexSession,
    source: &Schema,
    plan: &Plan,
    row_block: usize,
) -> Arc<dyn LayoutStrategy> {
    // Vortex's defaults, for the columns the plan leaves alone. Not the writer's session: its
    // edition would let them pick encodings Vortex doesn't by default.
    let defaults = VortexSession::default();
    let mut builder = builder(&defaults, row_block);
    for field in source.fields() {
        let name = field.name().as_str();
        match (plan.get(name), field.data_type()) {
            (Some(Transform::Ohlc(_)), _) if name == OHLC_PRICES[0] => {
                let validity = CollectStrategy::new(CompressingStrategy::new(
                    FlatLayoutStrategy::default(),
                    BtrBlocksCompressor::from_session(&defaults),
                ));
                let mut parts =
                    TableStrategy::new(Arc::new(validity), Arc::new(FlatLayoutStrategy::default()));
                for part in OHLC_PARTS {
                    parts = parts
                        .with_field_writer(FieldPath::from_name(part), packed(session, row_block));
                }
                builder = builder.with_field_writer(FieldPath::from_name(OHLC), Arc::new(parts));
            }
            (Some(Transform::Ohlc(_)), _) => {}
            (Some(Transform::Packed), _) | (_, DataType::Decimal128(..)) => {
                builder = builder
                    .with_field_writer(FieldPath::from_name(name), packed(session, row_block));
            }
            _ => {}
        }
    }
    builder.build()
}

/// Vortex's default write strategy, but with `row_block`-row chunks. The default also coalesces
/// blocks up to about 1 MiB, which would merge the narrow columns into bigger chunks, so that is
/// turned off: every chunk is exactly one row block.
fn builder(session: &VortexSession, row_block: usize) -> WriteStrategyBuilder {
    WriteStrategyBuilder::from_session(session)
        .with_row_block_size(row_block)
        .with_data_block_target_bytes(None)
}

/// A full [`builder`] pipeline (row blocks, zone maps) for one column, on `session`: its
/// compressor may use what the writer's edition allows.
fn packed(session: &VortexSession, row_block: usize) -> Arc<dyn LayoutStrategy> {
    builder(session, row_block).build()
}
