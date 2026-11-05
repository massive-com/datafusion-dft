//! A decimal column stored as its residual against the product of two others: [`residual`] to
//! write it, and `vortex_ohlc.product_residual`, a Vortex scalar function, to rebuild
//! `residual + a × b` on reading.
//!
//! The [`bars`](crate::bars) layout stores a bar's dollar volume this way, against
//! `volume × open`. A bar's dollar volume is the sum of price × size over its trades:
//! `volume × vwap`, with the vwap between the bar's low and high. So `dollar_volume - volume ×
//! open` is at most `volume × (high - low)` in size, and 0 when every trade was at the open (a
//! single trade, a flat bar): 43% of the bars of a month of minute bars. Where `a` or `b` is
//! null, `a × b` counts as 0, so the residual is the column itself.

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

use crate::decimal::{checked_pow10, unscaled};
use crate::scaled::decoded;

/// The scalar function: its children are the residual, `a` and `b`, all decimals, with the
/// residual at the column's scale and `a`'s and `b`'s adding up to no more than it.
#[derive(Clone, Debug)]
pub struct ProductResidualFn;

/// The column's decimal type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProductResidualOptions {
    pub decimal: DecimalDType,
}

impl fmt::Display for ProductResidualOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.decimal)
    }
}

/// The column, a `decimal`, from expressions for its residual, `a` and `b`.
pub fn product_residual(
    decimal: DecimalDType,
    residual: Expression,
    a: Expression,
    b: Expression,
) -> Expression {
    ProductResidualFn.new_expr(ProductResidualOptions { decimal }, [residual, a, b])
}

/// `10^k`, for the `k` that brings `a × b`, at scales `a` and `b`, to the column's `scale`;
/// `None` if it can't: the residual is only exact at a scale at least theirs.
pub fn shift(scale: i8, a: i8, b: i8) -> Option<i128> {
    checked_pow10(i32::from(scale) - i32::from(a) - i32::from(b))
}

/// A chunk of a column and what the [`bars`](crate::bars) layout stores for it (see
/// [`residual`]).
pub(crate) struct Residual {
    /// The column's unscaled values, and which are valid.
    pub values: Buffer<i128>,
    pub valid: Mask,
    /// `column - a × b`, at the column's type and with its validity.
    pub residual: ArrayRef,
}

/// The residual of a chunk of `column` against `a × b`, which counts as 0 where either is null.
pub(crate) fn residual(
    column: &DecimalArray,
    a: &DecimalArray,
    b: &DecimalArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Residual> {
    let (values, valid) = unscaled(column, ctx)?;
    let (a_values, a_valid) = unscaled(a, ctx)?;
    let (b_values, b_valid) = unscaled(b, ctx)?;
    let both_valid = &a_valid & &b_valid;
    let factor = shift(column.scale(), a.scale(), b.scale())
        .ok_or_else(|| vortex_err!("the column's scale is below a × b's"))?;
    let mut residual = BufferMut::with_capacity(values.len());
    for i in 0..values.len() {
        let product = match both_valid.value(i) {
            true => (a_values[i].checked_mul(b_values[i]))
                .and_then(|p| p.checked_mul(factor))
                .ok_or_else(|| vortex_err!("a × b is too wide for an i128"))?,
            false => 0,
        };
        residual.push(values[i].wrapping_sub(product));
    }
    let residual = DecimalArray::new(
        residual.freeze(),
        column.decimal_dtype(),
        column.validity()?,
    );
    Ok(Residual {
        values,
        valid,
        residual: residual.into_array(),
    })
}

/// Run `$body` with `$values` bound to `$array`'s unscaled values at their width if it's an i32,
/// i64 or i128's, and widened to i128s otherwise.
///
/// *Upstream*: after vortex#10701 a decoded decimal always has its precision's width, and the
/// stored width is a `Widen`'s child, which [`decoded`] should return (see there).
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

/// `r × r_factor + a × b × product_factor` for each row, `a × b` taken as 0 where `valid` (both
/// `a`'s and `b`'s) is false.
fn rebuild<R, A, B>(
    r: &[R],
    a: &[A],
    b: &[B],
    valid: &Mask,
    r_factor: i128,
    product_factor: i128,
) -> Buffer<i128>
where
    R: AsPrimitive<i128> + AsPrimitive<i64>,
    A: AsPrimitive<i128> + AsPrimitive<i64>,
    B: AsPrimitive<i128> + AsPrimitive<i64>,
{
    let at = |r: R, a: A, b: B| {
        let (a, b): (i128, i128) = (a.as_(), b.as_());
        let product = a.wrapping_mul(b).wrapping_mul(product_factor);
        AsPrimitive::<i128>::as_(r)
            .wrapping_mul(r_factor)
            .wrapping_add(product)
    };
    // The usual case: the residual fits an i64, `a` and `b` an i32 (so `a × b` is exact in an
    // i64), and both factors an i64. Each product then widens, and neither the products nor
    // their sum can overflow.
    let narrow = size_of::<R>() <= 8 && size_of::<A>() <= 4 && size_of::<B>() <= 4;
    if let (true, true, Ok(r_factor), Ok(product_factor)) = (
        narrow,
        valid.all_true(),
        i64::try_from(r_factor),
        i64::try_from(product_factor),
    ) {
        let rows = r.iter().zip(a).zip(b).map(|((&r, &a), &b)| {
            let (r, a, b): (i64, i64, i64) = (r.as_(), a.as_(), b.as_());
            widening(r, r_factor) + widening(a * b, product_factor)
        });
        return Buffer::from_trusted_len_iter(rows);
    }
    if valid.all_true() {
        let rows = r.iter().zip(a).zip(b).map(|((&r, &a), &b)| at(r, a, b));
        return Buffer::from_trusted_len_iter(rows);
    }
    Buffer::from_trusted_len_iter((0..r.len()).map(|i| match valid.value(i) {
        true => at(r[i], a[i], b[i]),
        false => AsPrimitive::<i128>::as_(r[i]).wrapping_mul(r_factor),
    }))
}

impl ScalarFnVTable for ProductResidualFn {
    type Options = ProductResidualOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex_ohlc.product_residual");
        *ID
    }

    fn serialize(&self, options: &ProductResidualOptions) -> VortexResult<Option<Vec<u8>>> {
        let decimal = options.decimal;
        Ok(Some(vec![decimal.precision(), decimal.scale() as u8]))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<ProductResidualOptions> {
        let &[precision, scale] = metadata else {
            vortex_bail!(
                "vortex_ohlc.product_residual takes 2 bytes of options, not {}",
                metadata.len()
            );
        };
        let decimal = DecimalDType::new(precision, scale as i8);
        Ok(ProductResidualOptions { decimal })
    }

    fn arity(&self, _options: &ProductResidualOptions) -> Arity {
        Arity::Exact(3)
    }

    fn child_name(&self, _options: &ProductResidualOptions, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("residual"),
            1 => ChildName::from("a"),
            2 => ChildName::from("b"),
            _ => unreachable!("vortex_ohlc.product_residual has three children"),
        }
    }

    fn return_dtype(
        &self,
        options: &ProductResidualOptions,
        args: &[DType],
    ) -> VortexResult<DType> {
        match &args[0] {
            DType::Decimal(_, nullability) => Ok(DType::Decimal(options.decimal, *nullability)),
            other => {
                vortex_bail!("vortex_ohlc.product_residual's residual is a decimal, not {other}")
            }
        }
    }

    fn execute(
        &self,
        options: &ProductResidualOptions,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let residual = decoded(args.get(0)?, ctx)?;
        let a = decoded(args.get(1)?, ctx)?;
        let b = decoded(args.get(2)?, ctx)?;
        let scale = i32::from(options.decimal.scale());
        let below = || vortex_err!("the column's scale is below its residual's or a × b's");
        let r_factor = checked_pow10(scale - i32::from(residual.scale())).ok_or_else(below)?;
        let product_factor =
            checked_pow10(scale - i32::from(a.scale()) - i32::from(b.scale())).ok_or_else(below)?;
        let validity = residual.validity()?;
        let a_valid = a.validity()?.execute_mask(a.len(), ctx)?;
        let both_valid = &a_valid & &b.validity()?.execute_mask(b.len(), ctx)?;
        let rebuilt = with_values!(residual, |r| {
            with_values!(a, |a| {
                with_values!(b, |b| rebuild(
                    &r,
                    &a,
                    &b,
                    &both_valid,
                    r_factor,
                    product_factor
                ))
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

    /// `residual + a × b` at the column's scale, a null factor counting as 0: a dollar volume from
    /// its residual, the volume and the open.
    #[test]
    fn rebuilds_the_column() {
        let session = VortexSession::default();
        crate::register(&session);
        let mut ctx = session.create_execution_ctx();
        let decimal = DecimalDType::new(38, 12);
        // $1.50 × 100 shares = $150, plus a residual of $0.25, a bar with a null volume and one
        // with a null open.
        let residual = DecimalArray::new(
            Buffer::from_iter([250_000_000_000i128, 7, 9]),
            decimal,
            Validity::NonNullable,
        );
        let volume = DecimalArray::new(
            Buffer::from_iter([100_000_000i128, 5, 2]),
            DecimalDType::new(38, 6),
            Validity::from_iter([true, false, true]),
        );
        let open = DecimalArray::new(
            Buffer::from_iter([15_000i128, 3, 4]),
            DecimalDType::new(38, 4),
            Validity::from_iter([true, true, false]),
        );
        let table = vortex::array::arrays::StructArray::from_fields(&[
            ("residual", residual.into_array()),
            ("volume", volume.into_array()),
            ("open", open.into_array()),
        ])
        .unwrap()
        .into_array();
        let expr = product_residual(
            decimal,
            get_item("residual", root()),
            get_item("volume", root()),
            get_item("open", root()),
        );
        let rebuilt = table.apply(&expr).unwrap();
        let rebuilt = rebuilt.execute::<DecimalArray>(&mut ctx).unwrap();
        let valid = Mask::new_true(3);
        let values = converted_buffer::<i128>(&rebuilt, &valid).unwrap();
        assert_eq!(values.as_slice(), &[150_250_000_000_000, 7, 9]);
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
        let expr = product_residual(
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
