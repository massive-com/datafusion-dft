//! `vortex_ohlc.price`: one of the four prices, rebuilt from the [`OHLC`](crate::plan::OHLC)
//! struct's parts, as a Vortex scalar function. Its kernel is [`Price::rebuild`].

use std::fmt;

use vortex::array::arrays::decimal::DecimalArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::arrays::{DecimalArray, StructArray};
use vortex::array::{ArrayRef, ExecutionCtx, IntoArray};
use vortex::buffer::Buffer;
use vortex::dtype::{DType, DecimalDType, DecimalType};
use vortex::error::{vortex_bail, VortexResult};
use vortex::expr::Expression;
use vortex::scalar_fn::{
    Arity, ChildName, ExecutionArgs, ScalarFnId, ScalarFnVTable, ScalarFnVTableExt,
};
use vortex::session::registry::CachedId;
use vortex::session::VortexSession;

use crate::ohlc::{Part, Price};

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

/// The scalar function: its one child is the [`OHLC`](crate::plan::OHLC) struct.
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
        let mut values: [Option<Buffer<i128>>; 4] = Default::default();
        let mut parts_scale = None;
        for (part, values) in Part::ALL.into_iter().zip(&mut values) {
            if !options.price.uses(part) {
                continue;
            }
            let field = ohlc.unmasked_field_by_name(part.name())?.clone();
            let field = field.execute::<DecimalArray>(ctx)?;
            parts_scale = Some(field.scale());
            *values = Some(widened(&field)?);
        }
        let shift = options.decimal.scale() - parts_scale.expect("every price uses open");
        if shift < 0 {
            vortex_bail!("the price's scale is below the parts'");
        }
        let parts = values.each_ref().map(|v| v.as_deref().unwrap_or(&[]));
        // The prices fit their precision, so i64 holds them if that's at most 18 digits.
        let rebuilt = if options.decimal.precision() <= 18 {
            options.price.rebuild(parts, 10i64.pow(shift as u32))
        } else {
            options.price.rebuild(parts, 10i128.pow(shift as u32))
        };
        let array = DecimalArray::new(
            Buffer::from(rebuilt),
            options.decimal,
            ohlc.struct_validity(),
        );
        Ok(array.into_array())
    }
}

/// A decimal array's unscaled values as i128s, whatever width they're stored at.
fn widened(array: &DecimalArray) -> VortexResult<Buffer<i128>> {
    Ok(match array.values_type() {
        DecimalType::I128 => array.buffer::<i128>(),
        DecimalType::I64 => array.buffer::<i64>().iter().map(|&v| v as i128).collect(),
        DecimalType::I32 => array.buffer::<i32>().iter().map(|&v| v as i128).collect(),
        DecimalType::I16 => array.buffer::<i16>().iter().map(|&v| v as i128).collect(),
        DecimalType::I8 => array.buffer::<i8>().iter().map(|&v| v as i128).collect(),
        other => vortex_bail!("parts stored as {other:?}"),
    })
}
