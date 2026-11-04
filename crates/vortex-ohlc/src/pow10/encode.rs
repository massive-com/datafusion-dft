//! Choosing each block's exponent, and dividing by it.

use num_traits::{AsPrimitive, CheckedMul, PrimInt};
use vortex::array::arrays::primitive::PrimitiveArrayExt;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::dtype::NativePType;
use vortex::array::patches::Patches;
use vortex::array::validity::Validity;
use vortex::array::{match_each_integer_ptype, ExecutionCtx, IntoArray};
use vortex::buffer::{Buffer, BufferMut};
use vortex::error::VortexResult;
use vortex::mask::Mask;

use super::{Pow10, Pow10Array, BLOCK};

/// `array` as a [`Pow10Array`] with plain children, for the caller to compress; `None` when no
/// block is worth an exponent.
pub fn pow10_encode(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Pow10Array>> {
    let exponents = block_exponents(array, ctx)?;
    pow10_encode_with(array, &exponents, ctx)
}

/// `array` as a [`Pow10Array`] with each block's exponent in `exponents`, as
/// [`block_exponents`] chose them; `None` when they're all 0.
pub(super) fn pow10_encode_with(
    array: &PrimitiveArray,
    exponents: &[u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Pow10Array>> {
    if exponents.iter().all(|&e| e == 0) {
        return Ok(None);
    }
    let valid = array.validity()?.execute_mask(array.len(), ctx)?;
    let patch_validity = match array.dtype().is_nullable() {
        true => Validity::AllValid,
        false => Validity::NonNullable,
    };
    match_each_integer_ptype!(array.ptype(), |T| {
        let values = array.as_slice::<T>();
        let divided = divide(values, &valid, exponents);
        let patches = match divided.patch_rows.is_empty() {
            true => None,
            false => {
                let rows =
                    PrimitiveArray::new(Buffer::from(divided.patch_rows), Validity::NonNullable);
                let patched = PrimitiveArray::new(divided.patch_values, patch_validity);
                Some(Patches::new(
                    values.len(),
                    0,
                    rows.narrow(ctx)?.into_array(),
                    patched.into_array(),
                    None,
                )?)
            }
        };
        let quotients = PrimitiveArray::new(divided.quotients, array.validity()?);
        let exponents = Buffer::copy_from(exponents).into_array();
        Ok(Some(Pow10::try_new(
            quotients.into_array(),
            exponents,
            patches,
        )?))
    })
}

/// Each block's exponent in `array`: see [`exponents`].
pub(super) fn block_exponents(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<u8>> {
    let valid = array.validity()?.execute_mask(array.len(), ctx)?;
    Ok(match_each_integer_ptype!(array.ptype(), |T| {
        exponents(array.as_slice::<T>(), &valid)
    }))
}

/// Each block's exponent, the one that packs it smallest: its kept quotients' range in bits for
/// each of its values, plus a cost per patch (a 32-bit row and the value). 0 for a block without
/// valid values.
fn exponents<T>(values: &[T], valid: &Mask) -> Vec<u8>
where
    T: NativePType + PrimInt + CheckedMul + AsPrimitive<i128>,
{
    let powers = powers::<T>();
    let max = powers.len() - 1;
    let patch_bits = 32 + T::PTYPE.bit_width() as u64;
    // For each count of trailing zeros, the least and greatest value with it, and how many.
    let mut by_zeros = vec![(T::max_value(), T::min_value(), 0u64); max + 1];
    let mut exponents = Vec::with_capacity(values.len().div_ceil(BLOCK));
    for (block, chunk) in values.chunks(BLOCK).enumerate() {
        by_zeros.fill((T::max_value(), T::min_value(), 0));
        let mut total = 0;
        for (i, &v) in chunk.iter().enumerate() {
            if !valid.value(block * BLOCK + i) {
                continue;
            }
            let mut zeros = max;
            if !v.is_zero() {
                zeros = 0;
                while zeros < max && v % powers[zeros + 1] == T::zero() {
                    zeros += 1;
                }
            }
            let (lo, hi, n) = &mut by_zeros[zeros];
            (*lo, *hi, *n) = ((*lo).min(v), (*hi).max(v), *n + 1);
            total += 1;
        }
        // From the largest exponent down, so that a tie goes to the smaller one.
        let (mut lo, mut hi, mut kept) = (T::max_value(), T::min_value(), 0);
        let mut best = (u64::MAX, 0);
        for exponent in (0..=max).rev() {
            let (l, h, n) = by_zeros[exponent];
            (lo, hi, kept) = (lo.min(l), hi.max(h), kept + n);
            if kept == 0 {
                continue;
            }
            let range: i128 = (hi.as_() - lo.as_()) / powers[exponent].as_();
            let width = u64::from(128 - range.leading_zeros());
            let cost = chunk.len() as u64 * width + (total - kept) * patch_bits;
            if cost <= best.0 {
                best = (cost, exponent as u8);
            }
        }
        exponents.push(best.1);
    }
    exponents
}

/// `10^0` up to the largest power of ten `T` holds.
fn powers<T: PrimInt + CheckedMul>() -> Vec<T> {
    let ten = T::from(10).unwrap_or_else(T::one);
    std::iter::successors(Some(T::one()), |p| p.checked_mul(&ten)).collect()
}

/// The quotients of a [`Pow10Array`], and its patches' rows and values.
struct Divided<T> {
    quotients: Buffer<T>,
    patch_rows: Vec<u64>,
    patch_values: Buffer<T>,
}

/// Each value divided by its block's power of ten, or patched where it isn't a multiple. A null
/// or patched row holds its block's lowest quotient, which leaves the block's range as it is.
fn divide<T: NativePType + PrimInt>(values: &[T], valid: &Mask, exponents: &[u8]) -> Divided<T> {
    let mut quotients = BufferMut::<T>::with_capacity(values.len());
    let (mut patch_rows, mut patch_values) = (Vec::new(), BufferMut::<T>::empty());
    for ((block, chunk), &exponent) in values.chunks(BLOCK).enumerate().zip(exponents) {
        let factor = pow10_checked::<T>(exponent);
        let start = block * BLOCK;
        let kept = |i: usize| valid.value(start + i) && chunk[i] % factor == T::zero();
        let filler = (0..chunk.len())
            .filter(|&i| kept(i))
            .map(|i| chunk[i] / factor)
            .min()
            .unwrap_or_else(T::zero);
        for (i, &v) in chunk.iter().enumerate() {
            if kept(i) {
                quotients.push(v / factor);
                continue;
            }
            quotients.push(filler);
            if valid.value(start + i) {
                patch_rows.push((start + i) as u64);
                patch_values.push(v);
            }
        }
    }
    Divided {
        quotients: quotients.freeze(),
        patch_rows,
        patch_values: patch_values.freeze(),
    }
}

/// `10^exponent`, for an exponent [`exponents`] chose, so it fits.
fn pow10_checked<T: PrimInt>(exponent: u8) -> T {
    let ten = T::from(10).unwrap_or_else(T::one);
    (0..exponent).fold(T::one(), |acc, _| acc * ten)
}

#[cfg(test)]
mod tests {
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::VortexSessionExecute;

    use super::*;
    use crate::pow10::Pow10ArraySlotsExt;

    fn session() -> vortex::session::VortexSession {
        use vortex::VortexSessionDefault;
        let session = vortex::session::VortexSession::default();
        crate::pow10::register(&session);
        session
    }

    /// What `array` decodes to, read whole and one row at a time.
    fn assert_round_trip(array: PrimitiveArray, exponents: &[u8]) {
        let mut ctx = session().create_execution_ctx();
        let encoded = pow10_encode(&array, &mut ctx)
            .unwrap()
            .expect("a block divides");
        let got = encoded
            .exponents()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_eq!(got.as_slice::<u8>(), exponents);
        let decoded = encoded
            .clone()
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        let (want, encoded) = (array.into_array(), encoded.into_array());
        for i in 0..want.len() {
            let want = want.execute_scalar(i, &mut ctx).unwrap();
            assert_eq!(
                decoded.execute_scalar(i, &mut ctx).unwrap(),
                want,
                "row {i}"
            );
            assert_eq!(
                encoded.execute_scalar(i, &mut ctx).unwrap(),
                want,
                "row {i}"
            );
        }
    }

    /// Prices at a scale of 4 that move in cents, but for a block of odd values; negative
    /// values; a block of zeros; and a short last block.
    #[test]
    fn exponents_per_block() {
        let values = (0..4500i64).map(|i| match i {
            1024..2048 => 1_000_000 + 7 * i,
            2048..3072 => -2_000_000 + 100 * i,
            3072..4096 => 0,
            _ => 1_000_000 + 100 * (i % 50),
        });
        assert_round_trip(PrimitiveArray::from_iter(values), &[2, 0, 2, 0, 2]);
    }

    /// Cents with a sub-cent price now and then: the block keeps its exponent, the odd prices are
    /// patched, and they read back as they were.
    #[test]
    fn patches_odd_values() {
        let values: Vec<i64> = (0..2500i64)
            .map(|i| match i % 97 {
                0 => 672_505 + i,
                _ => 672_500 + 100 * (i % 40),
            })
            .collect();
        let odd = values.iter().filter(|&&v| v % 100 != 0).count();
        let mut ctx = session().create_execution_ctx();
        let encoded = pow10_encode(&PrimitiveArray::from_iter(values.clone()), &mut ctx)
            .unwrap()
            .unwrap();
        let patches = crate::pow10::Pow10ArrayExt::patches(&encoded).expect("patches");
        assert_eq!(patches.num_patches(), odd);
        assert_round_trip(PrimitiveArray::from_iter(values), &[2, 2, 2]);
    }

    /// Nulls neither limit their block's exponent nor lose their validity.
    #[test]
    fn nulls() {
        let values = (0..3000u32).map(|i| match i % 5 {
            0 => None,
            _ => Some(1000 * (i % 64) + 1000),
        });
        assert_round_trip(PrimitiveArray::from_option_iter(values), &[3, 3, 3]);
    }

    /// A type's largest power of ten, and none past it.
    #[test]
    fn widest_exponent() {
        let values = (0..2048).map(|i| if i < 1024 { [-100i8, 0, 100][i % 3] } else { 1 });
        assert_round_trip(PrimitiveArray::from_iter(values), &[2, 0]);
        assert_eq!(powers::<u64>().len(), 20);
    }
}
