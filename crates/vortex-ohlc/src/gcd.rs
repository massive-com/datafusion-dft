//! [`GcdScheme`]: timestamps' seconds into the day stored divided by the greatest common divisor
//! of a chunk's values, as a [`Scaled`] array whose base is the divisor, to the power 1.
//!
//! Vortex stores a timestamp as DateTimeParts: days, seconds into the day and subseconds. A minute
//! bar's seconds are all multiples of 60, which Vortex's compressor packs at 16 bits a row (a
//! block of one ticker's bars spans most of a trading day). Divided by 60 they are minutes into
//! the day, which pack at 10. encoderrr's DateTimeMinutes got there with a copy of DateTimeParts
//! with a fourth part; dividing its seconds by their chunk's divisor keeps Vortex's own. With one
//! divisor for every row, a [`Scaled`] array reads through its quotients, so DateTimeParts'
//! comparisons (on the days, then the seconds) run on them.

use std::sync::Arc;

use num_traits::{NumCast, ToPrimitive};
use vortex::array::arrays::PrimitiveArray;
use vortex::array::vtable::VTable;
use vortex::array::{
    match_each_integer_ptype, ArrayId, ArrayRef, Canonical, ExecutionCtx, IntoArray,
};
use vortex::buffer::Buffer;
use vortex::error::{vortex_bail, vortex_err, VortexResult};
use vortex_btrblocks::schemes::temporal::TemporalScheme;
use vortex_btrblocks::{
    ArrayAndStats, CascadingCompressor, CompressorContext, Scheme, SchemeExt, SchemeId,
};
use vortex_compressor::scheme::{CompressionEstimate, EstimateVerdict};

use crate::scaled::Scaled;

/// `divisor` in `T`, if it fits.
fn divisor_as<T: NumCast>(divisor: u64) -> Option<T> {
    T::from(divisor)
}

/// The greatest common divisor of `values`' valid values, if it is above 1 and fits their type.
pub fn chunk_divisor(values: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<Option<u64>> {
    let valid = values.validity()?.execute_mask(values.len(), ctx)?;
    let divisor = match_each_integer_ptype!(values.ptype(), |T| {
        let mut divisor = 0u64;
        for (i, value) in values.as_slice::<T>().iter().enumerate() {
            if !valid.value(i) {
                continue;
            }
            // Every integer type's magnitudes fit a u64, i64::MIN's too.
            let magnitude = value.to_i128().unwrap_or_default().unsigned_abs() as u64;
            divisor = gcd(divisor, magnitude);
            if divisor == 1 {
                return Ok(None);
            }
        }
        (divisor > 1 && divisor_as::<T>(divisor).is_some()).then_some(divisor)
    });
    Ok(divisor)
}

/// The greatest common divisor of `a` and `b`, by Euclid's algorithm.
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `values` divided by `divisor`, which their valid values are all multiples of.
fn divide(values: &PrimitiveArray, divisor: u64) -> VortexResult<PrimitiveArray> {
    let validity = values.validity()?;
    Ok(match_each_integer_ptype!(values.ptype(), |T| {
        let divisor = divisor_as::<T>(divisor)
            .ok_or_else(|| vortex_err!("Gcd: a divisor of {divisor} for {}", values.ptype()))?;
        let quotients: Buffer<T> = values
            .as_slice::<T>()
            .iter()
            .map(|&v| v / divisor)
            .collect();
        PrimitiveArray::new(quotients, validity)
    }))
}

/// Stores DateTimeParts' seconds as [`Scaled`] by their chunk's divisor when they share one, then
/// compresses the quotients as the seconds would have been. A chunk's quotients span its values'
/// range over the divisor, so they pack narrower.
///
/// Only DateTimeParts' seconds, which share a divisor whenever the timestamps fall on whole minutes
/// or hours. A divisor of other integers (counts, prices' parts, which
/// [`Pow10Scheme`](crate::scaled::Pow10Scheme) divides a block at a time) is mostly 1.
#[derive(Debug)]
pub struct GcdScheme;

impl Scheme for GcdScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex_ohlc.int.gcd"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Scaled.id()]
    }

    /// Children: the quotients.
    fn num_children(&self) -> usize {
        1
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // A sample's divisor can be larger than the array's.
        let eligible = !compress_ctx.is_sample()
            && !compress_ctx.finished_cascading()
            && is_temporal_seconds(compress_ctx.cascade_history());
        let worth_it = eligible && divisor(data, exec_ctx).0.is_some();
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
        let Some(divisor) = divisor(data, exec_ctx).0 else {
            vortex_bail!("Gcd: no divisor");
        };
        let quotients = divide(&data.array_as_primitive().into_owned(), divisor)?.into_array();
        let quotients =
            compressor.compress_child(&quotients, &compress_ctx, self.id(), 0, exec_ctx)?;
        Ok(Scaled::try_new_uniform(quotients, divisor, 1)?.into_array())
    }
}

/// The divisor of a chunk, for both the estimate and [`GcdScheme::compress`], which get the same
/// `data`; `None` if it is 1 or couldn't be computed.
struct Divisor(Option<u64>);

fn divisor(data: &ArrayAndStats, ctx: &mut ExecutionCtx) -> Arc<Divisor> {
    data.get_or_insert_with(|| {
        Divisor(
            chunk_divisor(&data.array_as_primitive().into_owned(), ctx)
                .ok()
                .flatten(),
        )
    })
}

/// Whether the compressor is at DateTimeParts' seconds, under Vortex's temporal scheme.
fn is_temporal_seconds(history: &[(SchemeId, usize)]) -> bool {
    matches!(history, [(parent, 1)] if *parent == TemporalScheme.id())
}

#[cfg(test)]
mod tests {
    use vortex::array::arrays::extension::ExtensionArrayExt;
    use vortex::array::arrays::{ConstantArray, TemporalArray};
    use vortex::array::builtins::ArrayBuiltins;
    use vortex::array::extension::datetime::TimeUnit;
    use vortex::array::scalar::Scalar;
    use vortex::array::validity::Validity;
    use vortex::array::VortexSessionExecute;
    use vortex::compressor::BtrBlocksCompressorBuilder;
    use vortex::dtype::Nullability;
    use vortex::mask::Mask;

    use super::*;
    use crate::scaled::tests::{kernel_compare, OPERATORS};
    use crate::scaled::ScaledArrayExt;
    use crate::testing::{encoding_ids, session};

    /// `array`'s values, decoded and widened to i128s.
    fn decoded(array: &ArrayRef, ctx: &mut ExecutionCtx) -> Vec<i128> {
        let array = array.clone().execute::<PrimitiveArray>(ctx).unwrap();
        match_each_integer_ptype!(array.ptype(), |T| {
            (array.as_slice::<T>().iter())
                .map(|v| v.to_i128().unwrap())
                .collect()
        })
    }

    /// Minute bars' timestamps, in seconds: three days of a ticker's bars from 04:00 to 20:00.
    fn minute_bars() -> Vec<i64> {
        let day = 1_155_600_000 / 86_400 * 86_400;
        (0..3)
            .flat_map(|d| (240..1200).map(move |m| day + d * 86_400 + m * 60))
            .collect()
    }

    /// Minute bars' timestamps are stored as DateTimeParts with their seconds divided by 60, and
    /// read and compared as they were.
    #[test]
    fn minute_bars_seconds_are_divided_by_60() {
        let session = session();
        let mut ctx = session.create_execution_ctx();
        let seconds = minute_bars();
        let timestamps = TemporalArray::new_timestamp(
            PrimitiveArray::from_iter(seconds.iter().copied()).into_array(),
            TimeUnit::Seconds,
            None,
        );
        let compressor = BtrBlocksCompressorBuilder::from_session(&session).build();
        let compressed = compressor
            .compress(&timestamps.into_array(), &mut ctx)
            .unwrap();
        let ids = encoding_ids(&compressed);
        assert_eq!(ids[0], "vortex.datetimeparts", "{ids:?}");
        let parts = compressed.children();
        let scaled = parts[1].as_opt::<Scaled>().expect("the seconds are Scaled");
        assert_eq!((scaled.base(), scaled.exponent()), (60, Some(1)), "{ids:?}");

        let seconds: Vec<i128> = seconds.into_iter().map(<i128 as From<i64>>::from).collect();
        let storage = compressed
            .clone()
            .execute::<vortex::array::arrays::ExtensionArray>(&mut ctx)
            .unwrap();
        assert_eq!(decoded(storage.storage_array(), &mut ctx), seconds);

        // 14:30 on the second day, and a second after it.
        let at = seconds[960 + 630];
        for t in [at, at + 1] {
            let literal = Scalar::extension::<vortex::array::extension::datetime::Timestamp>(
                vortex::array::extension::datetime::TimestampOptions {
                    unit: TimeUnit::Seconds,
                    tz: None,
                },
                Scalar::primitive(i64::try_from(t).unwrap(), Nullability::NonNullable),
            );
            for (operator, expected) in OPERATORS {
                let literal = ConstantArray::new(literal.clone(), compressed.len()).into_array();
                let result = compressed.binary(literal, operator).unwrap();
                let result = result.execute::<Mask>(&mut ctx).unwrap();
                let expected: Vec<bool> = seconds.iter().map(|&v| expected(v, t)).collect();
                let result: Vec<bool> = (0..seconds.len()).map(|i| result.value(i)).collect();
                assert_eq!(result, expected, "{operator:?} {t}");
            }
        }
        // Against 14:30's seconds of the day, a multiple of 60, Scaled's kernel answers every
        // comparison of the seconds part, on the quotients.
        let part = decoded(&parts[1], &mut ctx);
        let t = 14 * 3600 + 30 * 60;
        for (operator, expected) in OPERATORS {
            let expected: Vec<bool> = part.iter().map(|&v| expected(v, t)).collect();
            let kernel = kernel_compare(&parts[1], t, operator, &mut ctx);
            assert_eq!(kernel, Some(expected), "{operator:?} {t}");
        }
    }

    /// Other integers aren't divided, even when they share a divisor.
    #[test]
    fn only_datetime_seconds_are_divided() {
        let session = session();
        let mut ctx = session.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..4096i64).map(|i| i * 60));
        let compressed = BtrBlocksCompressorBuilder::from_session(&session)
            .build()
            .compress(&values.into_array(), &mut ctx)
            .unwrap();
        let ids = encoding_ids(&compressed);
        assert!(!ids.iter().any(|id| id == "vortex_ohlc.scaled"), "{ids:?}");
    }

    /// The divisor ignores nulls and is the magnitudes', negative values' too; there is none if
    /// it is 1, or if every valid value is 0.
    #[test]
    fn the_divisor_is_the_valid_values() {
        let mut ctx = session().create_execution_ctx();
        let divisor = |values: &[i32], valid: &[bool], ctx: &mut ExecutionCtx| {
            let values: Buffer<i32> = values.iter().copied().collect();
            let validity = Validity::from_iter(valid.iter().copied());
            chunk_divisor(&PrimitiveArray::new(values, validity), ctx).unwrap()
        };
        assert_eq!(
            divisor(&[-120, 7, 180], &[true, false, true], &mut ctx),
            Some(60)
        );
        assert_eq!(
            divisor(&[-120, 7, 180], &[true, true, true], &mut ctx),
            None
        );
        assert_eq!(divisor(&[0, 7, 0], &[true, false, true], &mut ctx), None);
    }
}
