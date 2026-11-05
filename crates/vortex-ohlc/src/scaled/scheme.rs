//! [`Pow10Scheme`]: when Vortex's compressor stores integers as [`Scaled`].

use std::sync::Arc;

use vortex::array::vtable::VTable;
use vortex::array::{ArrayId, ArrayRef, Canonical, ExecutionCtx, IntoArray};
use vortex::encodings::decimal_byte_parts::DecimalBytePartsSlots;
use vortex::error::{vortex_err, VortexResult};
use vortex_btrblocks::schemes::decimal::DecimalScheme;
use vortex_btrblocks::{
    compress_patches, ArrayAndStats, CascadingCompressor, CompressorContext, Scheme, SchemeExt,
    SchemeId,
};
use vortex_compressor::scheme::{CompressionEstimate, EstimateVerdict};

use super::encode::{block_exponents, scaled_encode_with};
use super::{Scaled, ScaledArrayExt, ScaledArraySlotsExt};

/// Stores a decimal's integer part as [`Scaled`] whenever a block of it is worth an exponent, then
/// compresses its quotients, exponents and patches' values in turn. A block's kept quotients span
/// no more than its values do, so whatever would have compressed the values compresses them about
/// as well.
///
/// Only decimals, whose trailing zeros are their scale's: other integers (counts, a dictionary's
/// codes, which a file stores as a column of their own) have no use for them.
#[derive(Debug)]
pub struct Pow10Scheme;

impl Scheme for Pow10Scheme {
    fn scheme_name(&self) -> &'static str {
        "vortex_ohlc.int.pow10"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Scaled.id()]
    }

    /// Children: the quotients, the exponents, and the patches' values.
    fn num_children(&self) -> usize {
        3
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // A sample mixes rows from all over the array into one block, so it can't tell.
        let eligible = !compress_ctx.is_sample()
            && !compress_ctx.finished_cascading()
            && is_decimal_part(compress_ctx.cascade_history());
        let worth_it = eligible
            && (exponents(data, exec_ctx).0.as_ref())
                .is_some_and(|exponents| exponents.iter().any(|&e| e != 0));
        CompressionEstimate::Verdict(match worth_it {
            true => EstimateVerdict::AlwaysUse,
            false => EstimateVerdict::Skip,
        })
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let values = data.array_as_primitive().into_owned();
        let exponents = exponents(data, exec_ctx);
        let exponents = (exponents.0.as_ref()).ok_or_else(|| vortex_err!("no exponents"))?;
        let Some(array) = scaled_encode_with(&values, exponents, exec_ctx)? else {
            return Ok(values.into_array());
        };
        let compress = |child: &ArrayRef, index: usize, exec_ctx: &mut ExecutionCtx| {
            compressor.compress_child(child, &compress_ctx, self.id(), index, exec_ctx)
        };
        let encoded = compress(array.encoded(), 0, exec_ctx)?;
        let exponents = compress(array.exponents(), 1, exec_ctx)?;
        let patches = match array.patches() {
            None => None,
            Some(patches) => Some(
                compress_patches(patches, exec_ctx)?
                    .map_values(|values| compress(&values, 2, exec_ctx))?,
            ),
        };
        Ok(Scaled::try_new(encoded, exponents, patches)?.into_array())
    }
}

/// Each block's exponent, for both the estimate and [`Pow10Scheme::compress`], which get the same
/// `data`; `None` if they couldn't be computed.
struct Exponents(Option<Vec<u8>>);

fn exponents(data: &ArrayAndStats, ctx: &mut ExecutionCtx) -> Arc<Exponents> {
    data.get_or_insert_with(|| {
        Exponents(block_exponents(&data.array_as_primitive().into_owned(), ctx).ok())
    })
}

/// Whether the compressor is at a decimal column's integer part.
fn is_decimal_part(history: &[(SchemeId, usize)]) -> bool {
    matches!(history, [(parent, child)]
        if *parent == DecimalScheme::v1().id() && *child == DecimalBytePartsSlots::MSP)
}

/// Whether the compressor is at a decimal column's integer part, or at the quotients
/// [`Pow10Scheme`] made of one.
pub(crate) fn is_decimal_values(history: &[(SchemeId, usize)]) -> bool {
    match history {
        [_] => is_decimal_part(history),
        [part, (parent, 0)] => {
            is_decimal_part(std::slice::from_ref(part)) && *parent == Pow10Scheme.id()
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::arrays::DecimalArray;
    use vortex::array::validity::Validity;
    use vortex::array::VortexSessionExecute;
    use vortex::buffer::Buffer;
    use vortex::compressor::BtrBlocksCompressor;
    use vortex::dtype::DecimalDType;
    use vortex::session::VortexSession;
    use vortex::VortexSessionDefault;

    use super::*;

    /// The encoding IDs of `array` and everything under it.
    fn encoding_ids(array: &ArrayRef) -> Vec<String> {
        let mut ids = vec![array.encoding_id().to_string()];
        for child in array.children() {
            ids.extend(encoding_ids(&child));
        }
        ids
    }

    /// The compressor of a session with vortex-ohlc stores a decimal column whose blocks move in
    /// different steps (cents, then odd ten-thousandths) as Scaled under its integer part, and
    /// it reads back as it was.
    #[test]
    fn compressor_picks_pow10_for_decimals() {
        let session = VortexSession::default();
        crate::edition::enable(&session).unwrap();
        crate::register(&session);
        let mut ctx = session.create_execution_ctx();
        let values: Buffer<i64> = (0..8192i64)
            .map(|i| match (i / 1024) % 2 {
                0 => 1_000_000 + 100 * (i % 300),
                _ => 50_000 + 7 * (i % 300),
            })
            .collect();
        let decimal = DecimalArray::new(
            values.clone(),
            DecimalDType::new(18, 4),
            Validity::NonNullable,
        )
        .into_array();
        let compressed = BtrBlocksCompressor::from_session(&session)
            .compress(&decimal, &mut ctx)
            .unwrap();
        let ids = encoding_ids(&compressed);
        assert!(ids.iter().any(|id| id == "vortex_ohlc.scaled"), "{ids:?}");
        // Read back at the narrowest width that holds the values.
        let read = compressed.execute::<DecimalArray>(&mut ctx).unwrap();
        let read: Vec<i64> = read.buffer::<i32>().iter().map(|&v| v.into()).collect();
        assert_eq!(read, values.to_vec());
    }
}
