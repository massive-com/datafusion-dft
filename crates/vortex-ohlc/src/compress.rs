//! FoR + FastLanes for decimal and integer chunks, with nothing but Vortex's own encodings:
//! decimals narrowed to the smallest integer that holds the chunk and kept in DecimalByteParts,
//! and the integers under Vortex's blocked FoR + BitPacked or plain FoR + FastLanes, whichever
//! comes out smaller.

use vortex::array::arrays::decimal::narrowed_decimal;
use vortex::array::arrays::{DecimalArray, PrimitiveArray};
use vortex::array::{ArrayRef, ExecutionCtx, IntoArray};
use vortex::compressor::BtrBlocksCompressor;
use vortex::dtype::DecimalType;
use vortex::encodings::decimal_byte_parts::{split_decimal, DecimalByteParts};
use vortex::encodings::fastlanes::bitpack_compress::{
    bitpack_blocked_to_best_bit_widths, bitpack_to_best_bit_width,
};
use vortex::encodings::fastlanes::{FoR, FoRArrayExt, FoRArraySlotsExt, FoRReferences};
use vortex::error::VortexResult;

/// Values per FastLanes block: blocked FoR and bit-packing keep a reference and a bit width for
/// each.
const BLOCK: usize = 1024;

/// Compress one chunk of a decimal or integer column with FoR + FastLanes; anything else goes
/// to `fallback`. Integers are packed as they are. Decimals are narrowed to the smallest integer
/// that holds every value in the chunk and wrapped in DecimalByteParts; a chunk wider than an
/// i64 is cut into an i64 and 64-bit lower parts, each packed.
pub fn compress_chunk(
    chunk: &ArrayRef,
    ctx: &mut ExecutionCtx,
    fallback: &BtrBlocksCompressor,
) -> VortexResult<ArrayRef> {
    let Some(decimal_dtype) = chunk.dtype().as_decimal_opt().copied() else {
        return match chunk.dtype().is_int() {
            true => pack_ints(chunk.clone().execute::<PrimitiveArray>(ctx)?, ctx),
            false => fallback.compress(chunk, ctx),
        };
    };
    let decimal = narrowed_decimal(chunk.clone().execute::<DecimalArray>(ctx)?);
    let validity = decimal.validity()?;
    let ints = match decimal.values_type() {
        DecimalType::I8 => PrimitiveArray::new(decimal.buffer::<i8>(), validity),
        DecimalType::I16 => PrimitiveArray::new(decimal.buffer::<i16>(), validity),
        DecimalType::I32 => PrimitiveArray::new(decimal.buffer::<i32>(), validity),
        DecimalType::I64 => PrimitiveArray::new(decimal.buffer::<i64>(), validity),
        _ => {
            let parts = split_decimal(&decimal, ctx)?;
            let msp = pack_ints(parts.msp.execute::<PrimitiveArray>(ctx)?, ctx)?;
            let lower_parts = parts
                .lower_parts
                .into_iter()
                .map(|part| pack_ints(part.execute::<PrimitiveArray>(ctx)?, ctx))
                .collect::<VortexResult<_>>()?;
            return Ok(DecimalByteParts::try_new_with_lower_parts(
                msp,
                lower_parts,
                decimal.decimal_dtype(),
            )?
            .into_array());
        }
    };
    Ok(DecimalByteParts::try_new(pack_ints(ints, ctx)?, decimal_dtype)?.into_array())
}

/// FoR + FastLanes over an integer array: Vortex's blocked FoR + BitPacked (a reference and a
/// bit width per [`BLOCK`] values) where there is more than one block, or plain FoR + FastLanes
/// (one of each), whichever is smaller.
pub fn pack_ints(ints: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let blocked = match ints.len() > BLOCK {
        true => Some(blocked_for_fastlanes(&ints, ctx)?),
        false => None,
    };
    let plain = for_fastlanes(ints, ctx)?;
    Ok(match blocked {
        Some(blocked) if blocked.nbytes() < plain.nbytes() => blocked,
        _ => plain,
    })
}

/// Vortex's own blocked FoR + FastLanes: a FoR reference per [`BLOCK`] values (each block's
/// minimum) over bit-packing at a width per block that minimizes its bits plus its patches. The
/// references are packed with plain FoR + FastLanes.
fn blocked_for_fastlanes(ints: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let for_array = FoR::encode_chunked(ints.clone(), ctx)?;
    let shifted = for_array.encoded().clone().execute::<PrimitiveArray>(ctx)?;
    let encoded = match bitpack_blocked_to_best_bit_widths(&shifted, ctx) {
        Ok(packed) => packed.into_array(),
        Err(_) => shifted.into_array(),
    };
    let array = match for_array.references() {
        FoRReferences::Global(reference) => FoR::try_new(encoded, reference.clone())?,
        FoRReferences::Blocked(references) => {
            let references = references.clone().execute::<PrimitiveArray>(ctx)?;
            FoR::try_new_chunked(encoded, for_fastlanes(references, ctx)?, for_array.offset())?
        }
    };
    Ok(array.into_array())
}

/// FoR: shift the integers so they're all non-negative, which bit-packing requires. Then
/// FastLanes: pack at the narrowest width that keeps the exception count cheap. If the values
/// need the full width (or the shift wrapped negative), keep them as they are.
fn for_fastlanes(ints: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let for_array = FoR::encode(ints, ctx)?;
    let reference = match for_array.references() {
        FoRReferences::Global(reference) => reference.clone(),
        FoRReferences::Blocked(_) => unreachable!("FoR::encode takes one reference"),
    };
    let shifted = for_array.encoded().clone().execute::<PrimitiveArray>(ctx)?;
    let packed = match bitpack_to_best_bit_width(&shifted, ctx) {
        Ok(packed) => packed.into_array(),
        Err(_) => shifted.into_array(),
    };
    Ok(FoR::try_new(packed, reference)?.into_array())
}
