//! The four prices as Vortex expressions over the [`OHLC`] struct's parts, for a scan to rebuild
//! them ([`price`]) or filter on them ([`compare_price`]) without leaving Vortex.
//!
//! Vortex only adds and compares decimals of one type, and a sum is one digit wider than its
//! operands, so each operand is cast to the precision its place needs: the parts start at their
//! own precision `p` ([`OHLC_PRECISION`](crate::plan::OHLC_PRECISION)), a sum of two at `p + 1`
//! and of three at `p + 2`, at most 18 digits, so every value is an i64.

use vortex::dtype::{DType, DecimalDType, Nullability};
use vortex::expr::{
    and, binary, cast, checked_add, fill_null, get_item, gt_eq, lit, lt, mask, or, root, Expression,
};
use vortex::scalar::{DecimalValue, Scalar};
use vortex::scalar_fn::fns::operators::Operator;

use crate::plan::{OHLC, OHLC_PARTS};

/// A price as an expression over the [`OHLC`] struct's parts (in a table of `table`), with only
/// Vortex's own functions; `None` for anything but the four prices, or a table without the
/// struct. close = open + body, high = (open + upper) + max(body, 0) and low = (open - lower) +
/// min(body, 0): the larger and smaller of open and close are open plus the body where its sign
/// is right. Keeping the body only there (`mask`, then `fill_null` with 0) runs vectorized, where
/// a `zip` picking open or close slices both at every run of its mask, and the body's sign
/// changes from one bar to the next.
pub fn price(table: &DType, price: &str) -> Option<Expression> {
    let parts = OhlcParts::new(table)?;
    let p = parts.precision;
    // The body where `sign` holds, 0 elsewhere, at p + 1 digits.
    let body_where = |sign: Expression| {
        let body = fill_null(mask(parts.raw("body"), sign), parts.zero());
        let decimal = DecimalDType::new(p + 1, parts.scale);
        cast(body, DType::Decimal(decimal, Nullability::Nullable))
    };
    Some(match price {
        "open" => parts.part("open", p),
        "close" => parts.close(),
        // p + 2 digits.
        "high" => {
            let base = checked_add(parts.part("open", p), parts.part("upper", p));
            checked_add(base, body_where(parts.up()))
        }
        "low" => {
            let base = binary(Operator::Sub, parts.part("open", p), parts.part("lower", p));
            checked_add(base, body_where(parts.down()))
        }
        _ => return None,
    })
}

/// `price op literal`, where the literal is the decimal `unscaled / 10^scale`, as a filter over
/// the [`OHLC`] struct's parts (in a table of `table`); `op` is a comparison. `None` for
/// anything but the four prices, a table without the struct, or a literal the price's type can't
/// hold exactly.
///
/// Open and close are compared as [`price`] rebuilds them. High and low are compared one branch
/// of the body's sign at a time: `high op x` is `(body >= 0 AND close + upper op x) OR (body < 0
/// AND open + upper op x)`, and low the same with open and close swapped. Each side is compared
/// at its own type, as a sum of parts.
pub fn compare_price(
    table: &DType,
    price: &str,
    op: Operator,
    unscaled: i128,
    scale: i8,
) -> Option<Expression> {
    let compare = |value: Expression| -> Option<Expression> {
        let DType::Decimal(decimal, _) = value.bind(table).ok()?.dtype().clone() else {
            return None;
        };
        Some(binary(
            op,
            value,
            decimal_literal(unscaled, scale, decimal)?,
        ))
    };
    let parts = OhlcParts::new(table)?;
    let p = parts.precision;
    let (when_up, when_down) = match price {
        "high" => (
            checked_add(parts.close(), parts.part("upper", p + 1)),
            checked_add(parts.part("open", p), parts.part("upper", p)),
        ),
        "low" => (
            binary(Operator::Sub, parts.part("open", p), parts.part("lower", p)),
            binary(Operator::Sub, parts.close(), parts.part("lower", p + 1)),
        ),
        _ => return compare(self::price(table, price)?),
    };
    Some(or(
        and(parts.up(), compare(when_up)?),
        and(parts.down(), compare(when_down)?),
    ))
}

/// The decimal `unscaled / 10^scale` as a literal of type `decimal`, or `None` if that type
/// can't hold it exactly.
pub(crate) fn decimal_literal(
    unscaled: i128,
    scale: i8,
    decimal: DecimalDType,
) -> Option<Expression> {
    let shift = decimal.scale() as i32 - scale as i32;
    let factor = 10i128.checked_pow(shift.unsigned_abs())?;
    let value = if shift >= 0 {
        unscaled.checked_mul(factor)?
    } else {
        (unscaled % factor == 0).then(|| unscaled / factor)?
    };
    let value = DecimalValue::I128(value);
    value
        .fits_in_precision(decimal)
        .then(|| lit(Scalar::decimal(value, decimal, Nullability::NonNullable)))
}

/// The [`OHLC`] struct's parts in a table, as the expressions over them are built.
struct OhlcParts {
    precision: u8,
    scale: i8,
}

impl OhlcParts {
    /// `None` if the table of `dtype` has no [`OHLC`] struct of decimal parts.
    fn new(dtype: &DType) -> Option<Self> {
        let parts = dtype.as_struct_fields_opt()?.field(OHLC)?;
        let DType::Decimal(decimal, _) = parts.as_struct_fields_opt()?.field(OHLC_PARTS[0])? else {
            return None;
        };
        Some(Self {
            precision: decimal.precision(),
            scale: decimal.scale(),
        })
    }

    fn raw(&self, name: &str) -> Expression {
        get_item(name, get_item(OHLC, root()))
    }

    /// The part `name`, cast to `precision`.
    fn part(&self, name: &str, precision: u8) -> Expression {
        let decimal = DecimalDType::new(precision, self.scale);
        cast(
            self.raw(name),
            DType::Decimal(decimal, Nullability::Nullable),
        )
    }

    /// open + body, at `p + 1` digits.
    fn close(&self) -> Expression {
        let p = self.precision;
        checked_add(self.part("open", p), self.part("body", p))
    }

    fn zero(&self) -> Expression {
        let decimal = DecimalDType::new(self.precision, self.scale);
        lit(Scalar::decimal(
            DecimalValue::I128(0),
            decimal,
            Nullability::NonNullable,
        ))
    }

    /// Where close is at least open.
    fn up(&self) -> Expression {
        gt_eq(self.raw("body"), self.zero())
    }

    /// Where close is below open.
    fn down(&self) -> Expression {
        lt(self.raw("body"), self.zero())
    }
}
