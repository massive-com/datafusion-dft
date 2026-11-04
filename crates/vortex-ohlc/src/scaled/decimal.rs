//! [`DecimalPow10Scheme`]: a chunk of decimals stored at a smaller scale, without the trailing
//! zeros all its values have.
//!
//! A decimal column has the source's scale, not its values': prices at a scale of 4 that move in
//! cents, volumes at a scale of 6 that are whole shares, dollar volumes at a scale of 12 with at
//! most 4 places. Vortex's DecimalByteParts stores the unscaled integers, zeros and all. For a
//! dollar volume they are wider than an i64, so it splits them into an integer part and a lower
//! `u64` part, and [`Pow10Scheme`] can only divide the integer part. Stored instead as a
//! [`Scaled`] decimal of one exponent, `k`, the most trailing zeros its values all have (up to the
//! scale), a chunk's quotients fit DecimalByteParts' integer part, and Pow10Scheme divides out
//! what more a block shares.

use vortex::array::arrays::decimal::{converted_buffer, narrowed_decimal};
use vortex::array::arrays::DecimalArray;
use vortex::array::vtable::VTable;
use vortex::array::{ArrayId, ArrayRef, Canonical, ExecutionCtx, IntoArray};
use vortex::buffer::Buffer;
use vortex::dtype::DecimalDType;
use vortex::encodings::decimal_byte_parts::{
    decimal_byte_parts_v1_id, decimal_byte_parts_v2_id, split_decimal, DecimalByteParts,
    DecimalBytePartsSlots,
};
use vortex::error::{vortex_err, VortexResult};
use vortex_btrblocks::schemes::decimal::DecimalScheme;
use vortex_btrblocks::{ArrayAndStats, CascadingCompressor, CompressorContext, Scheme, SchemeExt};
use vortex_compressor::scheme::{CompressionEstimate, EstimateVerdict};

#[cfg(doc)]
use super::Pow10Scheme;
use super::{Scaled, ScaledArrayExt, ScaledArraySlotsExt};

/// `array` decoded, through a [`Scaled`] decimal divided by one `10^k` for every row to its
/// values: the same numbers at `k` less scale, at the width they're stored at rather than an
/// i128's.
///
/// *Upstream*: with vortex#10701, DecimalByteParts (vortex#10905) and narrow `vortex.decimal`
/// nodes (vortex#10995) decode to a `Widen` of the values at their stored width, whose compares
/// and arithmetic stay narrow. This should then look through `Widen` as it does `Scaled`:
/// executed to a `DecimalArray`, the values would be copied out at the precision's width.
pub(crate) fn decoded(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<DecimalArray> {
    let Some(scaled) = array.as_opt::<Scaled>() else {
        return array.execute::<DecimalArray>(ctx);
    };
    let k = scaled.exponent().filter(|_| scaled.divisor().is_some());
    let Some(k) = k.filter(|_| scaled.base() == 10) else {
        return array.execute::<DecimalArray>(ctx);
    };
    let values = scaled.values().clone().execute::<DecimalArray>(ctx)?;
    let decimal = values.decimal_dtype();
    let scale = i8::try_from(i16::from(decimal.scale()) - i16::from(k))
        .map_err(|_| vortex_err!("a scale of {} less {k}", decimal.scale()))?;
    DecimalArray::try_new_handle(
        values.buffer_handle().clone(),
        values.values_type(),
        DecimalDType::try_new(decimal.precision(), scale)?,
        values.validity()?,
    )
}

/// `10^k`.
fn factor(k: u8) -> i128 {
    10i128.pow(u32::from(k))
}

/// The largest `k`, up to the scale, such that every valid value of `decimal` is a multiple of
/// `10^k`.
pub fn chunk_exponent(decimal: &DecimalArray, ctx: &mut ExecutionCtx) -> VortexResult<u8> {
    // 10^38 is the largest power of ten an i128 holds.
    let Ok(max) = u8::try_from(decimal.decimal_dtype().scale().clamp(0, 38)) else {
        return Ok(0);
    };
    let valid = decimal.validity()?.execute_mask(decimal.len(), ctx)?;
    let values = converted_buffer::<i128>(decimal, &valid)?;
    let mut k = max;
    for (i, &value) in values.iter().enumerate() {
        if !valid.value(i) {
            continue;
        }
        while k > 0 && value % factor(k) != 0 {
            k -= 1;
        }
        if k == 0 {
            break;
        }
    }
    Ok(k)
}

/// `decimal`'s values divided by `10^k`, which they are all multiples of, at its dtype: the
/// values of a [`Scaled`] decimal of the exponent `k`.
fn divide(decimal: &DecimalArray, k: u8, ctx: &mut ExecutionCtx) -> VortexResult<DecimalArray> {
    let valid = decimal.validity()?.execute_mask(decimal.len(), ctx)?;
    let factor = factor(k);
    let values: Buffer<i128> = (converted_buffer::<i128>(decimal, &valid)?.iter())
        .map(|v| v / factor)
        .collect();
    Ok(DecimalArray::new(
        values,
        decimal.decimal_dtype(),
        decimal.validity()?,
    ))
}

/// Stores a chunk of decimals as Vortex's decimal scheme does, as DecimalByteParts, but under
/// [`Scaled`] when its values share trailing zeros: divided by them first, so that the integer
/// part [`Pow10Scheme`] and blocked FoR compress is narrower, and fits an i64 where the unscaled
/// values don't. One cascade step, as Vortex's is, so the integer part's encodings are as deep as
/// they'd be under Vortex's.
///
/// Every chunk of decimals is this scheme's, so it takes the place of Vortex's decimal scheme in
/// a compressor rather than competing with it (both always win, so the first registered would):
/// see [`crate::write::write_options`].
#[derive(Debug)]
pub struct DecimalPow10Scheme;

/// Vortex's decimal scheme, for chunks with no trailing zeros to drop. Allowed to split values
/// wider than an i64 into parts, as [`crate::edition`] lets a writer.
const DECIMAL: DecimalScheme = DecimalScheme::v2();

impl Scheme for DecimalPow10Scheme {
    fn scheme_name(&self) -> &'static str {
        "vortex_ohlc.decimal.pow10"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches!(canonical, Canonical::Decimal(_))
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![
            Scaled.id(),
            decimal_byte_parts_v1_id(),
            decimal_byte_parts_v2_id(),
        ]
    }

    /// Children: DecimalByteParts', its integer part and up to three lower parts.
    fn num_children(&self) -> usize {
        DECIMAL.num_children()
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let decimal = data.array().clone().execute::<DecimalArray>(exec_ctx)?;
        let k = chunk_exponent(&decimal, exec_ctx)?;
        if k == 0 {
            return DECIMAL.compress(compressor, data, compress_ctx, exec_ctx);
        }
        let divided = narrowed_decimal(divide(&decimal, k, exec_ctx)?);
        let parts = split_decimal(&divided, exec_ctx)?;
        let msp = compressor.compress_child(
            &parts.msp,
            &compress_ctx,
            self.id(),
            DecimalBytePartsSlots::MSP,
            exec_ctx,
        )?;
        let mut lower_parts = Vec::with_capacity(parts.lower_parts.len());
        for (index, part) in parts.lower_parts.iter().enumerate() {
            lower_parts.push(compressor.compress_child(
                part,
                &compress_ctx,
                self.id(),
                DecimalBytePartsSlots::LOWER_PARTS_OFFSET + index,
                exec_ctx,
            )?);
        }
        let values =
            DecimalByteParts::try_new_with_lower_parts(msp, lower_parts, divided.decimal_dtype())?;
        Ok(Scaled::try_new_uniform(values.into_array(), 10, k)?.into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::arrays::decimal::converted_buffer;
    use vortex::array::validity::Validity;
    use vortex::array::VortexSessionExecute;
    use vortex::compressor::BtrBlocksCompressorBuilder;
    use vortex::dtype::{DType, DecimalDType, Nullability};
    use vortex::mask::Mask;

    use super::*;
    use crate::scaled::ScaledArrayExt;
    use crate::testing::{encoding_ids, session};

    fn unscaled(array: &ArrayRef, ctx: &mut ExecutionCtx) -> Vec<i128> {
        let array = array.clone().execute::<DecimalArray>(ctx).unwrap();
        let valid = Mask::new_true(array.len());
        converted_buffer::<i128>(&array, &valid).unwrap().to_vec()
    }

    /// Dollar volumes at a scale of 12, wider than an i64 unscaled but with 8 trailing zeros:
    /// stored scaled, by `10^8` for every row, their integer part one i64 (no lower parts), and
    /// read back as they were.
    #[test]
    fn wide_decimals_with_trailing_zeros_are_scaled() {
        let session = session();
        let mut ctx = session.create_execution_ctx();
        let values: Buffer<i128> = (0..4096i128)
            .map(|i| (700_000_000_000 + 1_234 * i) * 100_000_000)
            .collect();
        let dtype = DecimalDType::new(38, 12);
        let decimal = DecimalArray::new(values.clone(), dtype, Validity::NonNullable);
        assert_eq!(chunk_exponent(&decimal, &mut ctx).unwrap(), 8);
        let compressor = BtrBlocksCompressorBuilder::from_session(&session)
            .exclude_schemes([DecimalScheme::v1().id()])
            .with_new_scheme(&DecimalPow10Scheme)
            .build();
        let compressed = compressor
            .compress(&decimal.into_array(), &mut ctx)
            .unwrap();
        let ids = encoding_ids(&compressed);
        assert_eq!(ids[0], "vortex_ohlc.scaled", "{ids:?}");
        let scaled = compressed.as_opt::<Scaled>().unwrap();
        assert_eq!((scaled.base(), scaled.exponent()), (10, Some(8)));
        let parts = &compressed.children()[0];
        assert_eq!(
            parts.encoding_id().to_string(),
            "vortex.decimal_byte_parts.v2"
        );
        assert_eq!(parts.children().len(), 1, "an integer part only: {ids:?}");
        assert_eq!(
            compressed.dtype(),
            &DType::Decimal(dtype, Nullability::NonNullable)
        );
        assert_eq!(unscaled(&compressed, &mut ctx), values.to_vec());
    }

    /// The exponent ignores nulls, and stops at the scale.
    #[test]
    fn the_exponent_is_the_valid_values_and_at_most_the_scale() {
        let mut ctx = session().create_execution_ctx();
        let values: Buffer<i128> = [1_000_000, 7, 3_000_000].into_iter().collect();
        let validity = Validity::from_iter([true, false, true]);
        let decimal = DecimalArray::new(values, DecimalDType::new(10, 4), validity);
        assert_eq!(chunk_exponent(&decimal, &mut ctx).unwrap(), 4);
    }
}
