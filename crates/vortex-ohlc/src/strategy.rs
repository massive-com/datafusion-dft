//! The write strategy: Vortex's default, with every decimal column, the integer columns the plan
//! packs ([`Transform::Packed`]) and each part of the [`OHLC`] struct compressed by
//! [`compress_chunk`]. Each part of the OHLC struct is a column of its own with its own zone map,
//! so a scan can prune on the prices (see [`crate::datafusion::ZonePruner`]).

use std::sync::Arc;

use arrow::datatypes::{DataType, Schema};
use vortex::array::{ArrayRef, ExecutionCtx};
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

use crate::compress::compress_chunk;
use crate::plan::{Plan, Transform, OHLC, OHLC_PARTS, OHLC_PRICES};

/// Rows per chunk (row block) by default: 65,536, the row group size of the Parquet files the
/// bars come from.
pub const DEFAULT_ROW_BLOCK: usize = 65_536;

/// The write strategy for `source` stored with `plan` (see [`crate::plan::apply_plan`]), in
/// `row_block`-row chunks. The writing session must enable [`crate::edition`]; any Vortex session
/// reads the file.
pub fn write_strategy(source: &Schema, plan: &Plan, row_block: usize) -> Arc<dyn LayoutStrategy> {
    // Vortex's defaults, for the compressors and write strategies Vortex builds from a session.
    // Not the writer's: its editions would let them pick encodings Vortex doesn't by default.
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
                    parts = parts.with_field_writer(
                        FieldPath::from_name(part),
                        packed(&defaults, row_block),
                    );
                }
                builder = builder.with_field_writer(FieldPath::from_name(OHLC), Arc::new(parts));
            }
            (Some(Transform::Ohlc(_)), _) => {}
            (Some(Transform::Packed), _) | (_, DataType::Decimal128(..)) => {
                builder = builder
                    .with_field_writer(FieldPath::from_name(name), packed(&defaults, row_block));
            }
            _ => {}
        }
    }
    builder.build()
}

/// Vortex's default write strategy, but with `row_block`-row chunks. The default also coalesces
/// blocks up to about 1 MiB, which would merge the narrow columns into bigger chunks, so that is
/// turned off: every chunk is exactly one row block.
fn builder(defaults: &VortexSession, row_block: usize) -> WriteStrategyBuilder {
    WriteStrategyBuilder::from_session(defaults)
        .with_row_block_size(row_block)
        .with_data_block_target_bytes(None)
}

/// A full [`builder`] pipeline (row blocks, zone maps) for one column, with only the compressor
/// swapped for [`compress_chunk`].
fn packed(defaults: &VortexSession, row_block: usize) -> Arc<dyn LayoutStrategy> {
    let fallback = BtrBlocksCompressor::from_session(defaults);
    let probe_fallback = BtrBlocksCompressor::from_session(defaults);
    builder(defaults, row_block)
        .with_compressor(move |chunk: &ArrayRef, ctx: &mut ExecutionCtx| {
            compress_chunk(chunk, ctx, &fallback)
        })
        // The writer's probe decides whether to build one file-wide dictionary for the column.
        // It never sees a dictionary from this compressor, so the column gets none.
        .with_probe_compressor(move |chunk: &ArrayRef, ctx: &mut ExecutionCtx| {
            compress_chunk(chunk, ctx, &probe_fallback)
        })
        .build()
}
