//! [`BlockedForScheme`]: Vortex's FoR with a reference per block of 1024 values, over
//! bit-packing with a width per block, for decimals, where each block's range says it pays.
//!
//! Vortex 0.89's own FoR scheme estimates from one reference for the whole array, so it doesn't
//! pick a reference per block where only that would help (a chunk of many tickers' prices: each
//! block is narrow, the chunk wide), and it packs what is left at one width, the widest block's.
//! This one estimates from the widths blocked bit-packing picks for each block's residuals, and
//! packs each block at its own width.

use std::sync::Arc;

use vortex::array::arrays::PrimitiveArray;
use vortex::array::{ArrayId, ArrayRef, Canonical, ExecutionCtx, IntoArray};
use vortex::encodings::fastlanes::bitpack_compress::find_best_bit_widths;
use vortex::encodings::fastlanes::{
    bitpacked_v1_id, bitpacked_v2_id, for_v1_id, for_v2_id, FoR, FoRArray, FoRArrayExt,
    FoRArraySlotsExt, FoRReferences, FL_CHUNK_SIZE,
};
use vortex::error::{SharedVortexResult, VortexResult};
use vortex_btrblocks::schemes::integer::BitPackingScheme;
use vortex_btrblocks::{ArrayAndStats, CascadingCompressor, CompressorContext, Scheme, SchemeExt};
use vortex_compressor::scheme::{CompressionEstimate, EstimateVerdict};

use crate::scaled::{failed_estimate, is_decimal_values};

/// Blocked FoR over blocked bit-packing for a decimal's integer part, or the quotients
/// [`Scaled`](crate::scaled::Scaled) made of it, when its blocks' ranges pack smaller than the
/// values do without a reference.
///
/// A workaround, to be fixed upstream: Vortex's own FoR scheme should estimate a reference per
/// block from each block's range, and pack its residuals with a width per block. Once it does,
/// this scheme goes. The estimate's fix is drafted in `site/upstream/btrblocks-for-v2-estimate/`.
#[derive(Debug)]
pub struct BlockedForScheme;

impl Scheme for BlockedForScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex_ohlc.int.for_blocked"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    /// FoR and bit-packing, each with one reference or width or one per block.
    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![
            for_v1_id(),
            for_v2_id(),
            bitpacked_v1_id(),
            bitpacked_v2_id(),
        ]
    }

    /// Children: the references.
    fn num_children(&self) -> usize {
        1
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let eligible =
            !compress_ctx.finished_cascading() && is_decimal_values(compress_ctx.cascade_history());
        if !eligible {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        match &*blocked(data, exec_ctx) {
            Ok(blocked) => CompressionEstimate::Verdict(
                (blocked.as_ref())
                    .map_or(EstimateVerdict::Skip, |b| EstimateVerdict::Ratio(b.ratio)),
            ),
            Err(error) => failed_estimate(error.into()),
        }
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let for_ = match &*blocked(data, exec_ctx) {
            Ok(Some(blocked)) => blocked.for_.clone(),
            Err(error) => return Err(error.into()),
            Ok(None) => FoR::encode_chunked(data.array_as_primitive().into_owned(), exec_ctx)?,
        };
        let residuals =
            ArrayAndStats::new(for_.encoded().clone(), compress_ctx.merged_stats_options());
        let leaf = compress_ctx.clone().as_leaf();
        let packed = BitPackingScheme::v2().compress(compressor, &residuals, leaf, exec_ctx)?;
        let compressed = match for_.references() {
            FoRReferences::Global(reference) => FoR::try_new(packed, reference.clone())?,
            FoRReferences::Blocked(references) => {
                let references =
                    compressor.compress_child(references, &compress_ctx, self.id(), 0, exec_ctx)?;
                FoR::try_new_chunked(packed, references, for_.offset())?
            }
        };
        Ok(compressed.into_array())
    }
}

/// The values with a reference per block, and their size against what they pack to.
struct Blocked {
    for_: FoRArray,
    ratio: f64,
}

/// [`blocked_for`] of `data`, or the error computing it, once for both the estimate and
/// [`BlockedForScheme::compress`], which get the same `data`.
fn blocked(
    data: &ArrayAndStats,
    ctx: &mut ExecutionCtx,
) -> Arc<SharedVortexResult<Option<Blocked>>> {
    data.get_or_insert_with(|| {
        blocked_for(&data.array_as_primitive().into_owned(), ctx).map_err(Arc::new)
    })
}

/// `values` with a reference per block, if their residuals bit-pack (at the widths blocked
/// bit-packing picks, [`find_best_bit_widths`]) smaller with the references than the values do,
/// and than the values packed without one. `None` if not, or if a block's range doesn't fit the
/// type above zero (it can't be bit-packed).
fn blocked_for(values: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<Option<Blocked>> {
    let full = values.ptype().bit_width() as u64;
    let canonical = values.len() as u64 * full;
    let for_ = FoR::encode_chunked(values.clone(), ctx)?;
    let residuals = for_.encoded().clone().execute::<PrimitiveArray>(ctx)?;
    let Ok(packed) = find_best_bit_widths(&residuals, ctx) else {
        return Ok(None);
    };
    let for_bits = packed_bits(&packed, full) + packed.0.len() as u64 * full;
    // Negative values pack only zigzagged, which this leaves to the compressor's other schemes.
    let plain_bits = find_best_bit_widths(values, ctx).map_or(canonical, |p| packed_bits(&p, full));
    Ok((for_bits < canonical.min(plain_bits)).then(|| Blocked {
        for_,
        ratio: canonical as f64 / for_bits as f64,
    }))
}

/// The bits of blocks bit-packed at `widths` with `exceptions` patched: each block at its width
/// with a 32-bit offset, and each exception a 32-bit row and its value.
fn packed_bits((widths, exceptions): &(Vec<u8>, usize), full: u64) -> u64 {
    let blocks: u64 = (widths.iter())
        .map(|&width| FL_CHUNK_SIZE as u64 * u64::from(width) + 32)
        .sum();
    blocks + *exceptions as u64 * (32 + full)
}
