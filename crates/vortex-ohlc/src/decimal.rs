//! Reading decimal arrays' unscaled values, and the powers of ten between their scales.

use vortex::array::arrays::decimal::converted_buffer;
use vortex::array::arrays::DecimalArray;
use vortex::array::ExecutionCtx;
use vortex::buffer::Buffer;
use vortex::error::VortexResult;
use vortex::mask::Mask;

/// `decimal`'s unscaled values as i128s, and which are valid.
pub(crate) fn unscaled(
    decimal: &DecimalArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Buffer<i128>, Mask)> {
    let valid = decimal.validity()?.execute_mask(decimal.len(), ctx)?;
    Ok((converted_buffer::<i128>(decimal, &valid)?, valid))
}

/// `10^k`; `None` for a negative `k`, or one too big for an i128.
pub(crate) fn checked_pow10(k: impl TryInto<u32>) -> Option<i128> {
    10i128.checked_pow(k.try_into().ok()?)
}
