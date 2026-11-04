//! [`BlockedForScheme`]: Vortex's FoR with a reference per block of 1024 values, over
//! bit-packing with a width per block, for decimals, where each block's range says it pays.
//!
//! Vortex 0.89's own FoR scheme estimates from one reference for the whole array, so it doesn't
//! pick a reference per block where only that would help (a chunk of many tickers' prices: each
//! block is narrow, the chunk wide), and it packs what is left at one width, the widest block's.
//! This one measures each block's range and packs each block at its own width.

use vortex::array::arrays::PrimitiveArray;
use vortex::array::{
    match_each_integer_ptype, ArrayId, ArrayRef, Canonical, ExecutionCtx, IntoArray,
};
use vortex::encodings::fastlanes::{
    bitpacked_v1_id, bitpacked_v2_id, for_v1_id, for_v2_id, FoR, FoRArrayExt, FoRArraySlotsExt,
    FoRReferences, FL_CHUNK_SIZE,
};
use vortex::error::VortexResult;
use vortex::mask::Mask;
use vortex_btrblocks::schemes::integer::BitPackingScheme;
use vortex_btrblocks::{ArrayAndStats, CascadingCompressor, CompressorContext, Scheme, SchemeExt};
use vortex_compressor::scheme::{CompressionEstimate, EstimateVerdict};

use crate::pow10::is_decimal_values;

/// Blocked FoR over blocked bit-packing for a decimal's integer part, or the quotients
/// [`Pow10`](crate::pow10::Pow10) made of it, when its blocks' ranges pack smaller than the
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
        let ratio = match eligible {
            true => data
                .array()
                .clone()
                .execute::<PrimitiveArray>(exec_ctx)
                .and_then(|values| blocked_ratio(&values, exec_ctx))
                .ok()
                .flatten(),
            false => None,
        };
        CompressionEstimate::Verdict(ratio.map_or(EstimateVerdict::Skip, EstimateVerdict::Ratio))
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let values = data.array().clone().execute::<PrimitiveArray>(exec_ctx)?;
        let for_ = FoR::encode_chunked(values, exec_ctx)?;
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

/// `values`' size against what each block packs to at the width of its range, with its reference
/// and its 32-bit offset into the packed values. `None` if that's no smaller than the values, or
/// than the blocks bit-packed without a reference (zigzagged, if any is negative), or if a block's
/// range doesn't fit the type above zero (it can't be bit-packed).
fn blocked_ratio(values: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<Option<f64>> {
    let valid = values.validity()?.execute_mask(values.len(), ctx)?;
    let full = values.ptype().bit_width() as u64;
    let widest = match values.ptype().is_signed_int() {
        true => full - 1,
        false => full,
    };
    let blocks = match_each_integer_ptype!(values.ptype(), |T| {
        block_ranges(values.as_slice::<T>(), &valid)
    });
    if blocks.iter().any(|block| block.range_width > widest) {
        return Ok(None);
    }
    let negative = blocks.iter().any(|block| block.lo < 0);
    let (mut for_, mut plain) = (0, 0);
    for block in &blocks {
        for_ += block.len * block.range_width + full + 32;
        let top = match negative {
            true => block.hi.max(-block.lo) * 2,
            false => block.hi,
        };
        plain += block.len * width(top as u128) + 32;
    }
    let canonical = values.len() as u64 * full;
    Ok((for_ < canonical.min(plain)).then(|| canonical as f64 / for_ as f64))
}

/// A block's length, its valid values' least and greatest (0 for none), and the bit width of the
/// range between them.
struct BlockRange {
    len: u64,
    lo: i128,
    hi: i128,
    range_width: u64,
}

fn block_ranges<T: Copy + Ord + Into<i128>>(values: &[T], valid: &Mask) -> Vec<BlockRange> {
    values
        .chunks(FL_CHUNK_SIZE)
        .enumerate()
        .map(|(block, chunk)| {
            let start = block * FL_CHUNK_SIZE;
            let (lo, hi) = (0..chunk.len())
                .filter(|&i| valid.value(start + i))
                .map(|i| chunk[i].into())
                .fold(None, |range: Option<(i128, i128)>, v| match range {
                    None => Some((v, v)),
                    Some((lo, hi)) => Some((lo.min(v), hi.max(v))),
                })
                .unwrap_or((0, 0));
            BlockRange {
                len: chunk.len() as u64,
                lo,
                hi,
                range_width: width((hi - lo) as u128),
            }
        })
        .collect()
}

/// The bits `v` takes.
fn width(v: u128) -> u64 {
    u64::from(u128::BITS - v.leading_zeros())
}
