//! Column plans: rescale decimals, or store the four prices against each other, before writing.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::cast::AsArray;
use arrow::array::types::Decimal128Type;
use arrow::array::{Array, ArrayRef as ArrowArrayRef, Decimal128Array, RecordBatch, StructArray};
use arrow::datatypes::{DataType, Field, Fields, Schema};

use crate::ohlc::{ohlc_to_parts, Part, Price};

/// How a column is stored. Columns missing from a [`Plan`] are kept as they are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    /// Drop `k` trailing zero digits: `Decimal(p, s)` becomes `Decimal(p, s - k)`, each
    /// unscaled value divided by `10^k`. Lossless only because every value has them.
    Rescale(u8),
    /// One of the four [`Price`]s, rescaled by `k` as above and stored with the other three as
    /// one struct column, [`OHLC`]: open as it is, plus small offsets that are mostly zero (see
    /// [`Part`]). All four prices carry this transform; the struct takes open's place.
    Ohlc(u8),
}

pub type Plan = BTreeMap<String, Transform>;

/// The struct column holding the four [`Transform::Ohlc`] prices, as its [`Part`]s.
pub const OHLC: &str = "ohlc";

/// The [`OHLC`] parts' precision: 16 digits, so a price filter's sum of up to three parts has at
/// most 18 and is an i64 decimal too. [`apply_plan`] checks that every part fits.
pub const OHLC_PRECISION: u8 = 16;

/// The [`OHLC`] struct's fields: every part a `Decimal(OHLC_PRECISION, scale)`.
pub fn ohlc_fields(scale: i8) -> Fields {
    Part::ALL
        .iter()
        .map(|part| {
            Field::new(
                part.name(),
                DataType::Decimal128(OHLC_PRECISION, scale),
                false,
            )
        })
        .collect()
}

fn decimal_params(field: &Field) -> (u8, i8) {
    match field.data_type() {
        DataType::Decimal128(precision, scale) => (*precision, *scale),
        other => panic!("{} is {other}, not Decimal128", field.name()),
    }
}

/// The schema as written, with the plan applied.
pub fn planned_schema(plan: &Plan, schema: &Schema) -> Arc<Schema> {
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .filter_map(|field| {
            let field = field.as_ref().clone();
            Some(match plan.get(field.name()) {
                None => field,
                Some(Transform::Rescale(k)) => {
                    let (precision, scale) = decimal_params(&field);
                    field.with_data_type(DataType::Decimal128(precision, scale - *k as i8))
                }
                // The struct goes where open was; the other prices are in it.
                Some(Transform::Ohlc(k)) if field.name() == Price::Open.name() => {
                    let (_, scale) = decimal_params(&field);
                    let fields = ohlc_fields(scale - *k as i8);
                    field
                        .with_name(OHLC)
                        .with_data_type(DataType::Struct(fields))
                }
                Some(Transform::Ohlc(_)) => return None,
            })
        })
        .collect();
    Arc::new(Schema::new(fields))
}

/// Rewrite the planned columns of a source batch into their stored form, for `schema` (see
/// [`planned_schema`]).
///
/// # Panics
///
/// If the batch doesn't fit the plan: a null price, an [`OHLC`] part that doesn't fit
/// [`OHLC_PRECISION`], or (in debug builds only) a rescale that would drop digits. A plan from
/// [`crate::profile`] of the same data fits it.
pub fn apply_plan(plan: &Plan, schema: &Arc<Schema>, batch: RecordBatch) -> RecordBatch {
    if plan.is_empty() {
        return batch;
    }
    let columns: Vec<ArrowArrayRef> = batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .filter_map(|(field, column)| {
            let transform = match plan.get(field.name()) {
                None => return Some(Arc::clone(column)),
                Some(transform) => transform,
            };
            let (precision, scale) = decimal_params(field);
            let values = column.as_primitive::<Decimal128Type>();
            Some(match transform {
                Transform::Ohlc(k) if field.name() == Price::Open.name() => {
                    ohlc_struct(&batch, *k, scale - *k as i8)
                }
                Transform::Ohlc(_) => return None,
                Transform::Rescale(k) => {
                    let factor = 10i128.pow(*k as u32);
                    let rescaled = values.unary::<_, Decimal128Type>(|v| {
                        debug_assert_eq!(v % factor, 0, "rescale would drop digits");
                        v / factor
                    });
                    Arc::new(
                        rescaled
                            .with_precision_and_scale(precision, scale - *k as i8)
                            .unwrap(),
                    ) as ArrowArrayRef
                }
            })
        })
        .collect();
    RecordBatch::try_new(Arc::clone(schema), columns).unwrap()
}

/// A batch's four prices as the [`OHLC`] struct, rescaled by `k` to parts at `scale`.
fn ohlc_struct(batch: &RecordBatch, k: u8, scale: i8) -> ArrowArrayRef {
    let factor = 10i128.pow(k as u32);
    let [open, high, low, close] = Price::ALL.map(|price| {
        let name = price.name();
        let column = batch.column_by_name(name).unwrap();
        assert_eq!(column.null_count(), 0, "{name} must be non-null");
        column.as_primitive::<Decimal128Type>().values()
    });
    let mut parts: [Vec<i128>; 4] = Default::default();
    for i in 0..batch.num_rows() {
        let prices = [open[i], high[i], low[i], close[i]];
        debug_assert!(
            prices.iter().all(|v| v % factor == 0),
            "rescale would drop digits"
        );
        let [o, h, l, c] = prices.map(|v| v / factor);
        for (part, v) in parts.iter_mut().zip(ohlc_to_parts(o, h, l, c)) {
            part.push(v);
        }
    }
    let columns = parts
        .map(|values| {
            let array = Decimal128Array::from(values)
                .with_precision_and_scale(OHLC_PRECISION, scale)
                .unwrap();
            array
                .validate_decimal_precision(OHLC_PRECISION)
                .expect("OHLC parts fit their precision");
            Arc::new(array) as ArrowArrayRef
        })
        .to_vec();
    Arc::new(StructArray::new(ohlc_fields(scale), columns, None))
}
