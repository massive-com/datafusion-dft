//! `vortex_ohlc.price`: one of the four prices, rebuilt from the [`OHLC`](crate::bars::OHLC)
//! struct's parts, as a Vortex scalar function. Its kernel is [`Price::rebuild`].

use std::fmt;

use num_traits::{AsPrimitive, PrimInt, WrappingMul};

use vortex::array::arrays::decimal::DecimalArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::arrays::{DecimalArray, StructArray};
use vortex::array::{ArrayRef, ExecutionCtx, IntoArray};
use vortex::buffer::Buffer;
use vortex::dtype::{DType, DecimalDType, DecimalType, NativeDecimalType};
use vortex::error::{vortex_bail, VortexResult};
use vortex::expr::Expression;
use vortex::scalar_fn::{
    Arity, ChildName, ExecutionArgs, ScalarFnId, ScalarFnVTable, ScalarFnVTableExt,
};
use vortex::session::registry::CachedId;
use vortex::session::VortexSession;

use crate::ohlc::{Part, Price};
use crate::scaled::decoded;

/// Which price, and the decimal type it's rebuilt into: the source's, at a scale at least the
/// parts'.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PriceOptions {
    pub price: Price,
    pub decimal: DecimalDType,
}

impl fmt::Display for PriceOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}", self.price.name(), self.decimal)
    }
}

/// The scalar function: its one child is the [`OHLC`](crate::bars::OHLC) struct.
#[derive(Clone, Debug)]
pub struct OhlcPriceFn;

/// `price` rebuilt from `ohlc`, an expression for the struct of parts, as `decimal`.
pub fn ohlc_price(price: Price, decimal: DecimalDType, ohlc: Expression) -> Expression {
    OhlcPriceFn.new_expr(PriceOptions { price, decimal }, [ohlc])
}

impl ScalarFnVTable for OhlcPriceFn {
    type Options = PriceOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex_ohlc.price");
        *ID
    }

    fn serialize(&self, options: &PriceOptions) -> VortexResult<Option<Vec<u8>>> {
        let price = Price::ALL.iter().position(|p| *p == options.price).unwrap() as u8;
        let decimal = options.decimal;
        Ok(Some(vec![
            price,
            decimal.precision(),
            decimal.scale() as u8,
        ]))
    }

    fn deserialize(&self, metadata: &[u8], _session: &VortexSession) -> VortexResult<PriceOptions> {
        let &[price, precision, scale] = metadata else {
            vortex_bail!(
                "vortex_ohlc.price takes 3 bytes of options, not {}",
                metadata.len()
            );
        };
        let Some(&price) = Price::ALL.get(price as usize) else {
            vortex_bail!("no price {price}");
        };
        let decimal = DecimalDType::new(precision, scale as i8);
        Ok(PriceOptions { price, decimal })
    }

    fn arity(&self, _options: &PriceOptions) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &PriceOptions, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("ohlc"),
            _ => unreachable!("vortex_ohlc.price has one child"),
        }
    }

    fn return_dtype(&self, options: &PriceOptions, args: &[DType]) -> VortexResult<DType> {
        match &args[0] {
            DType::Struct(_, nullability) => Ok(DType::Decimal(options.decimal, *nullability)),
            other => vortex_bail!("vortex_ohlc.price takes the struct of parts, not {other}"),
        }
    }

    fn execute(
        &self,
        options: &PriceOptions,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let ohlc = args.get(0)?.execute::<StructArray>(ctx)?;
        // The parts the price uses, at the scale and width they're stored at (through `scaled`).
        let mut parts: [Option<DecimalArray>; 4] = Default::default();
        for (part, values) in Part::ALL.into_iter().zip(&mut parts) {
            if options.price.uses(part) {
                let field = ohlc.unmasked_field_by_name(part.name())?.clone();
                *values = Some(decoded(field, ctx)?);
            }
        }
        // The prices fit their precision, so i64 holds them if that's at most 18 digits.
        let validity = ohlc.struct_validity();
        let array = if options.decimal.precision() <= 18 {
            let rebuilt = rebuilt::<i64>(options, &parts)?;
            DecimalArray::new(rebuilt, options.decimal, validity)
        } else {
            let rebuilt = rebuilt::<i128>(options, &parts)?;
            DecimalArray::new(rebuilt, options.decimal, validity)
        };
        Ok(array.into_array())
    }
}

/// The price of every bar, in `W`, from `parts` (those it uses) at their own scales: each is
/// brought to the finest of them, and their sum to the price's scale.
fn rebuilt<W>(options: &PriceOptions, parts: &[Option<DecimalArray>; 4]) -> VortexResult<Buffer<W>>
where
    W: PrimInt + WrappingMul + NativeDecimalType + 'static,
    i8: AsPrimitive<W>,
    i16: AsPrimitive<W>,
    i32: AsPrimitive<W>,
    i64: AsPrimitive<W>,
    i128: AsPrimitive<W>,
{
    let finest = (parts.iter().flatten())
        .map(|part| part.scale())
        .max()
        .expect("every price uses open");
    let Ok(shift) = u8::try_from(options.decimal.scale() - finest) else {
        vortex_bail!("the price's scale is below the parts'");
    };
    let mut values: [Option<Buffer<W>>; 4] = Default::default();
    for (part, values) in parts.iter().zip(&mut values) {
        if let Some(part) = part {
            *values = Some(widened(part, pow10((finest - part.scale()) as u8))?);
        }
    }
    let values = values.each_ref().map(|v| v.as_deref().unwrap_or(&[]));
    Ok(Buffer::from(options.price.rebuild(values, pow10(shift))))
}

/// `10^exponent` in `W`, wrapping where it doesn't fit (a price's scale is at most `W`'s digits).
fn pow10<W: PrimInt + WrappingMul>(exponent: u8) -> W {
    let ten = W::from(10).unwrap_or_else(W::one);
    (0..exponent).fold(W::one(), |acc, _| acc.wrapping_mul(&ten))
}

/// A decimal array's unscaled values times `factor`, as `W`s, whatever width they're stored at.
fn widened<W>(array: &DecimalArray, factor: W) -> VortexResult<Buffer<W>>
where
    W: PrimInt + NativeDecimalType + 'static,
    i8: AsPrimitive<W>,
    i16: AsPrimitive<W>,
    i32: AsPrimitive<W>,
    i64: AsPrimitive<W>,
    i128: AsPrimitive<W>,
{
    fn widen<T, W>(values: &[T], factor: W) -> Buffer<W>
    where
        T: NativeDecimalType + AsPrimitive<W>,
        W: PrimInt + 'static,
    {
        // A multiply of 64-bit lanes doesn't vectorize on NEON or AVX2, so only where needed.
        match factor.is_one() {
            true => values.iter().map(|&v| v.as_()).collect(),
            false => values.iter().map(|&v| v.as_() * factor).collect(),
        }
    }
    Ok(match array.values_type() {
        DecimalType::I8 => widen(&array.buffer::<i8>(), factor),
        DecimalType::I16 => widen(&array.buffer::<i16>(), factor),
        DecimalType::I32 => widen(&array.buffer::<i32>(), factor),
        DecimalType::I64 => widen(&array.buffer::<i64>(), factor),
        DecimalType::I128 => widen(&array.buffer::<i128>(), factor),
        other => vortex_bail!("parts stored as {other:?}"),
    })
}

#[cfg(test)]
mod tests {
    use vortex::array::arrays::{DecimalArray, StructArray};
    use vortex::array::validity::Validity;
    use vortex::array::{IntoArray, VortexSessionExecute};
    use vortex::buffer::Buffer;
    use vortex::dtype::DecimalDType;
    use vortex::expr::root;
    use vortex::session::VortexSession;
    use vortex::VortexSessionDefault;

    use vortex::dtype::DecimalType;

    use super::ohlc_price;
    use crate::ohlc::{Part, Price};
    use crate::scaled::Scaled;

    /// Parts stored scaled by different powers of ten are brought to one scale before they're
    /// summed, and prices of at most 18 digits come back as i64s: a bar opening at 12.34, closing
    /// 0.5 higher, with a high 0.06 above that, in a `Decimal(18, 4)` source.
    #[test]
    fn scaled_parts_rebuild_in_i64() {
        let session = VortexSession::default();
        crate::register(&session);
        let mut ctx = session.create_execution_ctx();
        let part = |value: i32, scale: i8, k: u8| {
            let values = Buffer::from(vec![value]);
            // The quotients, at the scale of the values they stand for.
            let dtype = DecimalDType::new(16, scale + k as i8);
            let values = DecimalArray::new(values, dtype, Validity::NonNullable);
            let scaled = Scaled::try_new_uniform(values.into_array(), 10, k).unwrap();
            scaled.into_array()
        };
        let fields = [
            ("open", part(1234, 2, 2)),
            ("body", part(5, 1, 3)),
            ("upper", part(6, 2, 2)),
            ("lower", part(0, 4, 0)),
        ];
        let ohlc = StructArray::from_fields(&fields).unwrap().into_array();
        let decimal = DecimalDType::new(18, 4);
        let expected = [123_400i64, 129_000, 123_400, 128_400];
        for (price, expected) in Price::ALL.into_iter().zip(expected) {
            let expr = ohlc_price(price, decimal, root());
            let rebuilt = ohlc.clone().apply_bound(&expr.bind(ohlc.dtype()).unwrap());
            let rebuilt = rebuilt.unwrap().execute::<DecimalArray>(&mut ctx).unwrap();
            assert_eq!(rebuilt.values_type(), DecimalType::I64, "{price:?}");
            assert_eq!(rebuilt.buffer::<i64>()[0], expected, "{price:?}");
        }
    }

    /// Prices wider than an i64 come back whole: 60000.5 in a `Decimal(38, 18)` source, its
    /// parts rescaled by 17, takes 23 digits at the source's scale.
    #[test]
    fn wide_prices_rebuild_at_the_source_scale() {
        let session = VortexSession::default();
        crate::register(&session);
        let mut ctx = session.create_execution_ctx();
        let parts = DecimalDType::new(16, 1);
        let fields = Part::ALL.map(|part| {
            let value = if part == Part::Open { 600_005i64 } else { 0 };
            let values = DecimalArray::new(Buffer::from(vec![value]), parts, Validity::NonNullable);
            (part.name(), values.into_array())
        });
        let ohlc = StructArray::from_fields(&fields).unwrap().into_array();
        let decimal = DecimalDType::new(38, 18);
        for price in Price::ALL {
            let expr = ohlc_price(price, decimal, root());
            let rebuilt = ohlc.clone().apply_bound(&expr.bind(ohlc.dtype()).unwrap());
            let rebuilt = rebuilt.unwrap().execute::<DecimalArray>(&mut ctx).unwrap();
            assert_eq!(
                rebuilt.buffer::<i128>()[0],
                600_005 * 10i128.pow(17),
                "{price:?}"
            );
        }
    }
}
