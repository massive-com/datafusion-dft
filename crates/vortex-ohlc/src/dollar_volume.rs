//! A bar's dollar volume as the [`bars`](crate::bars) layout stores it, a residual against
//! `volume × open`: [`residual`] to write it, and `vortex_ohlc.dollar_volume`, a Vortex scalar
//! function, to rebuild `residual + volume × open` on reading.
//!
//! A bar's dollar volume is the sum of price × size over its trades: `volume × vwap`, with the
//! vwap between the bar's low and high. So `dollar_volume - volume × open` is at most
//! `volume × (high - low)` in size, and 0 when every trade was at the open (a single trade, a flat
//! bar), as many minute bars are. The layout stores that residual in its place.
//! A null volume counts as 0, so where the volume is null the residual is the dollar volume.

use std::fmt;

use num_traits::AsPrimitive;
use vortex::array::arrays::decimal::{converted_buffer, DecimalArrayExt};
use vortex::array::arrays::DecimalArray;
use vortex::array::{ArrayRef, ExecutionCtx, IntoArray};
use vortex::buffer::{Buffer, BufferMut};
use vortex::dtype::{DType, DecimalDType, DecimalType};
use vortex::error::{vortex_bail, vortex_err, VortexResult};
use vortex::expr::Expression;
use vortex::mask::Mask;
use vortex::scalar_fn::{
    Arity, ChildName, ExecutionArgs, ScalarFnId, ScalarFnVTable, ScalarFnVTableExt,
};
use vortex::session::registry::CachedId;
use vortex::session::VortexSession;

use crate::scaled::decoded;

/// The scalar function: its children are the residual, the volume and the open, all decimals,
/// with the residual at the dollar volume's scale and the volume's and the open's adding up to no
/// more than it.
#[derive(Clone, Debug)]
pub struct DollarVolumeFn;

/// The dollar volume's decimal type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DollarVolumeOptions {
    pub decimal: DecimalDType,
}

impl fmt::Display for DollarVolumeOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.decimal)
    }
}

/// The dollar volume, a `decimal`, from expressions for its residual, the volume and the open.
pub fn dollar_volume(
    decimal: DecimalDType,
    residual: Expression,
    volume: Expression,
    open: Expression,
) -> Expression {
    DollarVolumeFn.new_expr(DollarVolumeOptions { decimal }, [residual, volume, open])
}

/// `10^k`, for the `k` that brings `volume × open` to the dollar volume's `scale`; `None` if it
/// can't: the residual is only exact at a scale at least theirs.
pub fn shift(scale: i8, volume: i8, open: i8) -> Option<i128> {
    let k = i32::from(scale) - i32::from(volume) - i32::from(open);
    10i128.checked_pow(u32::try_from(k).ok()?)
}

/// A chunk's dollar volume and what the [`bars`](crate::bars) layout stores for it (see
/// [`residual`]).
pub(crate) struct Residual {
    /// The dollar volume's unscaled values, and which are valid.
    pub dollar_volume: Buffer<i128>,
    pub valid: Mask,
    /// `dollar_volume - volume × open`, at the dollar volume's type and with its validity.
    pub residual: ArrayRef,
}

/// The residual of a chunk's `dollar_volume`, given its `volume` and its unscaled `open` prices,
/// at `open_scale`. A null volume counts as 0.
pub(crate) fn residual(
    dollar_volume: &DecimalArray,
    volume: &DecimalArray,
    open: &[i128],
    open_scale: i8,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Residual> {
    let unscaled = |column: &DecimalArray, ctx: &mut ExecutionCtx| -> VortexResult<_> {
        let valid = column.validity()?.execute_mask(column.len(), ctx)?;
        Ok((converted_buffer::<i128>(column, &valid)?, valid))
    };
    let (dv_values, valid) = unscaled(dollar_volume, ctx)?;
    let (volume_values, volume_valid) = unscaled(volume, ctx)?;
    let factor = shift(dollar_volume.scale(), volume.scale(), open_scale)
        .ok_or_else(|| vortex_err!("the dollar volume's scale is below volume × open's"))?;
    let mut residual = BufferMut::with_capacity(dv_values.len());
    for i in 0..dv_values.len() {
        let product = match volume_valid.value(i) {
            true => (volume_values[i].checked_mul(open[i]))
                .and_then(|p| p.checked_mul(factor))
                .ok_or_else(|| vortex_err!("volume × open is too wide for an i128"))?,
            false => 0,
        };
        residual.push(dv_values[i].wrapping_sub(product));
    }
    let residual = DecimalArray::new(
        residual.freeze(),
        dollar_volume.decimal_dtype(),
        dollar_volume.validity()?,
    );
    Ok(Residual {
        dollar_volume: dv_values,
        valid,
        residual: residual.into_array(),
    })
}

/// `10^k`; `None` for a negative `k`, or one too big for an i128.
fn pow10(k: i32) -> Option<i128> {
    10i128.checked_pow(u32::try_from(k).ok()?)
}

/// Run `$body` with `$values` bound to `$array`'s unscaled values at their width if it's an i32,
/// i64 or i128's, and widened to i128s otherwise.
macro_rules! with_values {
    ($array:expr, |$values:ident| $body:expr) => {
        match $array.values_type() {
            DecimalType::I32 => {
                let $values = $array.buffer::<i32>();
                $body
            }
            DecimalType::I64 => {
                let $values = $array.buffer::<i64>();
                $body
            }
            _ => {
                let valid = Mask::new_true($array.len());
                let $values = converted_buffer::<i128>(&$array, &valid)?;
                $body
            }
        }
    };
}

/// `x × y`, widening 64 × 64 bits into 128: two instructions, where an i128 × i128 takes several.
fn widening(x: i64, y: i64) -> i128 {
    i128::from(x) * i128::from(y)
}

/// `r × r_factor + v × o × product_factor` for each row, `v × o` taken as 0 where `valid`
/// (the volume's) is false.
fn rebuild<R, V, O>(
    r: &[R],
    v: &[V],
    o: &[O],
    valid: &Mask,
    r_factor: i128,
    product_factor: i128,
) -> Buffer<i128>
where
    R: AsPrimitive<i128> + AsPrimitive<i64>,
    V: AsPrimitive<i128> + AsPrimitive<i64>,
    O: AsPrimitive<i128> + AsPrimitive<i64>,
{
    let at = |r: R, v: V, o: O| {
        let (v, o): (i128, i128) = (v.as_(), o.as_());
        let product = v.wrapping_mul(o).wrapping_mul(product_factor);
        AsPrimitive::<i128>::as_(r)
            .wrapping_mul(r_factor)
            .wrapping_add(product)
    };
    // The usual case: the residual fits an i64, the volume and the open an i32 (so `v × o` is
    // exact in an i64), and both factors an i64. Each product then widens, and neither the
    // products nor their sum can overflow.
    let narrow = size_of::<R>() <= 8 && size_of::<V>() <= 4 && size_of::<O>() <= 4;
    if let (true, true, Ok(r_factor), Ok(product_factor)) = (
        narrow,
        valid.all_true(),
        i64::try_from(r_factor),
        i64::try_from(product_factor),
    ) {
        let rows = r.iter().zip(v).zip(o).map(|((&r, &v), &o)| {
            let (r, v, o): (i64, i64, i64) = (r.as_(), v.as_(), o.as_());
            widening(r, r_factor) + widening(v * o, product_factor)
        });
        return Buffer::from_trusted_len_iter(rows);
    }
    if valid.all_true() {
        let rows = r.iter().zip(v).zip(o).map(|((&r, &v), &o)| at(r, v, o));
        return Buffer::from_trusted_len_iter(rows);
    }
    Buffer::from_trusted_len_iter((0..r.len()).map(|i| match valid.value(i) {
        true => at(r[i], v[i], o[i]),
        false => AsPrimitive::<i128>::as_(r[i]).wrapping_mul(r_factor),
    }))
}

impl ScalarFnVTable for DollarVolumeFn {
    type Options = DollarVolumeOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex_ohlc.dollar_volume");
        *ID
    }

    fn serialize(&self, options: &DollarVolumeOptions) -> VortexResult<Option<Vec<u8>>> {
        let decimal = options.decimal;
        Ok(Some(vec![decimal.precision(), decimal.scale() as u8]))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<DollarVolumeOptions> {
        let &[precision, scale] = metadata else {
            vortex_bail!(
                "vortex_ohlc.dollar_volume takes 2 bytes of options, not {}",
                metadata.len()
            );
        };
        let decimal = DecimalDType::new(precision, scale as i8);
        Ok(DollarVolumeOptions { decimal })
    }

    fn arity(&self, _options: &DollarVolumeOptions) -> Arity {
        Arity::Exact(3)
    }

    fn child_name(&self, _options: &DollarVolumeOptions, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("residual"),
            1 => ChildName::from("volume"),
            2 => ChildName::from("open"),
            _ => unreachable!("vortex_ohlc.dollar_volume has three children"),
        }
    }

    fn return_dtype(&self, options: &DollarVolumeOptions, args: &[DType]) -> VortexResult<DType> {
        match &args[0] {
            DType::Decimal(_, nullability) => Ok(DType::Decimal(options.decimal, *nullability)),
            other => vortex_bail!("vortex_ohlc.dollar_volume's residual is a decimal, not {other}"),
        }
    }

    fn execute(
        &self,
        options: &DollarVolumeOptions,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let residual = decoded(args.get(0)?, ctx)?;
        let volume = decoded(args.get(1)?, ctx)?;
        let open = decoded(args.get(2)?, ctx)?;
        let scale = i32::from(options.decimal.scale());
        let below = || vortex_err!("the dollar volume's scale is below its parts'");
        let r_factor = pow10(scale - i32::from(residual.scale())).ok_or_else(below)?;
        let product_factor =
            pow10(scale - i32::from(volume.scale()) - i32::from(open.scale())).ok_or_else(below)?;
        let validity = residual.validity()?;
        let volume_valid = volume.validity()?.execute_mask(volume.len(), ctx)?;
        let rebuilt = with_values!(residual, |r| {
            with_values!(volume, |v| {
                with_values!(open, |o| {
                    rebuild(&r, &v, &o, &volume_valid, r_factor, product_factor)
                })
            })
        });
        Ok(DecimalArray::new(rebuilt, options.decimal, validity).into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::validity::Validity;
    use vortex::array::VortexSessionExecute;
    use vortex::expr::{get_item, root};
    use vortex::session::VortexSession;
    use vortex::VortexSessionDefault;

    use super::*;
    use crate::scaled::Scaled;

    /// `residual + volume × open` at the dollar volume's scale, a null volume counting as 0.
    #[test]
    fn rebuilds_the_dollar_volume() {
        let session = VortexSession::default();
        crate::register(&session);
        let mut ctx = session.create_execution_ctx();
        let decimal = DecimalDType::new(38, 12);
        // $1.50 × 100 shares = $150, plus a residual of $0.25, and a bar with a null volume.
        let residual = DecimalArray::new(
            Buffer::from_iter([250_000_000_000i128, 7]),
            decimal,
            Validity::NonNullable,
        );
        let volume = DecimalArray::new(
            Buffer::from_iter([100_000_000i128, 5]),
            DecimalDType::new(38, 6),
            Validity::from_iter([true, false]),
        );
        let open = DecimalArray::new(
            Buffer::from_iter([15_000i128, 3]),
            DecimalDType::new(38, 4),
            Validity::NonNullable,
        );
        let table = vortex::array::arrays::StructArray::from_fields(&[
            ("residual", residual.into_array()),
            ("volume", volume.into_array()),
            ("open", open.into_array()),
        ])
        .unwrap()
        .into_array();
        let expr = dollar_volume(
            decimal,
            get_item("residual", root()),
            get_item("volume", root()),
            get_item("open", root()),
        );
        let rebuilt = table.apply(&expr).unwrap();
        let rebuilt = rebuilt.execute::<DecimalArray>(&mut ctx).unwrap();
        let valid = Mask::new_true(2);
        let values = converted_buffer::<i128>(&rebuilt, &valid).unwrap();
        assert_eq!(values.as_slice(), &[150_250_000_000_000, 7]);
        assert_eq!(shift(12, 6, 4), Some(100));
        assert_eq!(shift(4, 6, 4), None);
    }

    /// The same, from [`Scaled`] children, read at their values' scales and widths.
    #[test]
    fn rebuilds_from_scaled_children() {
        let session = VortexSession::default();
        crate::register(&session);
        let mut ctx = session.create_execution_ctx();
        fn scaled<T: vortex::dtype::NativeDecimalType>(
            values: Vec<T>,
            scale: i8,
            k: u8,
        ) -> ArrayRef {
            // The quotients, at the scale of the values they stand for.
            let values = DecimalArray::new(
                Buffer::from_iter(values),
                DecimalDType::new(38, scale + k as i8),
                Validity::NonNullable,
            );
            Scaled::try_new_uniform(values.into_array(), 10, k)
                .unwrap()
                .into_array()
        }
        // $0.25, 100 shares and $1.50 at scales 4, 0 and 2, stored 8, 6 and 2 digits short, at
        // the widths the month's are: the residual an i64, the volume and the open i32s.
        let table = vortex::array::arrays::StructArray::from_fields(&[
            ("residual", scaled(vec![2_500i64, -1], 4, 8)),
            ("volume", scaled(vec![100i32, 3], 0, 6)),
            ("open", scaled(vec![150i32, 7], 2, 2)),
        ])
        .unwrap()
        .into_array();
        let expr = dollar_volume(
            DecimalDType::new(38, 12),
            get_item("residual", root()),
            get_item("volume", root()),
            get_item("open", root()),
        );
        let rebuilt = table.apply(&expr).unwrap();
        let rebuilt = rebuilt.execute::<DecimalArray>(&mut ctx).unwrap();
        let values = converted_buffer::<i128>(&rebuilt, &Mask::new_true(2)).unwrap();
        // $150.25, and 3 × $0.07 − $0.0001 = $0.2099.
        assert_eq!(values.as_slice(), &[150_250_000_000_000, 209_900_000_000]);
    }
}
