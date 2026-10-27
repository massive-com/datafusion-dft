//! Column plans: rescale decimals, or store the four prices against each other, before writing.
//! The file records its plan (see [`transforms_json`]), so a reader can undo it.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::cast::AsArray;
use arrow::array::types::Decimal128Type;
use arrow::array::{Array, ArrayRef as ArrowArrayRef, Decimal128Array, RecordBatch, StructArray};
use arrow::datatypes::{DataType, Field, Fields, Schema};

use crate::ohlc::ohlc_to_parts;

/// How a column is stored. Columns missing from a [`Plan`] are kept as they are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    /// Drop `k` trailing zero digits: `Decimal(p, s)` becomes `Decimal(p, s - k)`, each
    /// unscaled value divided by `10^k`. Lossless only because every value has them.
    Rescale(u8),
    /// One of [`OHLC_PRICES`], rescaled by `k` as above and stored with the other three as one
    /// struct column, [`OHLC`]: open as it is, plus small offsets that are mostly zero (see
    /// [`OHLC_PARTS`]). All four prices carry this transform; the struct takes open's place.
    Ohlc(u8),
    /// An integer column, left as it is but compressed with FoR + FastLanes (see
    /// [`crate::compress`]) rather than BtrBlocks, without a dictionary: a bit width per 1024
    /// values, so the blocks of quiet tickers pack narrower than the chunk's busiest.
    Packed,
}

pub type Plan = BTreeMap<String, Transform>;

/// The struct column holding the four [`Transform::Ohlc`] prices.
pub const OHLC: &str = "ohlc";
/// The prices it holds, in the source's column order.
pub const OHLC_PRICES: [&str; 4] = ["open", "high", "low", "close"];
/// Its fields: `open`, `body = close - open` (signed), `upper = high - max(open, close)` and
/// `lower = min(open, close) - low` (both never negative, since `low <= open, close <= high`).
/// The wicks are zero in most bars and the body in about half.
pub const OHLC_PARTS: [&str; 4] = ["open", "body", "upper", "lower"];

/// The [`OHLC`] parts' precision: 16 digits, so a price filter's sum of up to three parts has at
/// most 18 and is an i64 decimal too. [`apply_plan`] checks that every part fits.
pub const OHLC_PRECISION: u8 = 16;

/// The [`OHLC`] struct's fields: every part a `Decimal(OHLC_PRECISION, scale)`.
pub fn ohlc_fields(scale: i8) -> Fields {
    OHLC_PARTS
        .iter()
        .map(|&part| Field::new(part, DataType::Decimal128(OHLC_PRECISION, scale), false))
        .collect()
}

/// The stored column that holds `column`: [`OHLC`] for a price in it, and `column` itself
/// otherwise.
pub fn stored_name<'a>(plan: &Plan, column: &'a str) -> &'a str {
    match plan.get(column) {
        Some(Transform::Ohlc(_)) => OHLC,
        _ => column,
    }
}

/// The key of the file's metadata segment that records its plan (see [`transforms_json`]).
pub const TRANSFORMS_KEY: &str = "vortex_ohlc.transforms";
/// Bumped whenever [`transforms_json`] changes what it records.
const TRANSFORMS_VERSION: u32 = 1;

/// How to rebuild each planned column of `source` from the file, recorded in the file so a
/// reader that doesn't know the plan can tell what it holds: `{"version", "columns": {name:
/// {"transform", "stored", "expr", "type", "k"}}}`. Each `expr` gives the column's values from
/// stored columns (`ohlc.open` is a field of the struct), to be cast to the source's `type`; `k`
/// is how many trailing zero digits the stored values drop. A column only compressed its own way
/// has itself as its `expr`.
pub fn transforms_json(plan: &Plan, source: &Schema) -> serde_json::Value {
    let columns: serde_json::Map<String, serde_json::Value> = plan
        .iter()
        .map(|(name, transform)| {
            let field = source.field_with_name(name).unwrap();
            let (kind, expr, k) = match *transform {
                Transform::Rescale(k) => ("rescale", name.clone(), Some(k)),
                Transform::Ohlc(k) => {
                    let (open, close) = ("ohlc.open", "(ohlc.open + ohlc.body)");
                    let expr = match name.as_str() {
                        "open" => open.to_string(),
                        "high" => format!("max({open}, {close}) + ohlc.upper"),
                        "low" => format!("min({open}, {close}) - ohlc.lower"),
                        _ => close.to_string(),
                    };
                    ("ohlc", expr, Some(k))
                }
                Transform::Packed => ("packed", name.clone(), None),
            };
            let mut entry = serde_json::json!({
                "transform": kind,
                "stored": stored_name(plan, name),
                "expr": expr,
                "type": field.data_type().to_string(),
            });
            if let Some(k) = k {
                entry["k"] = k.into();
            }
            (name.clone(), entry)
        })
        .collect();
    serde_json::json!({"version": TRANSFORMS_VERSION, "columns": columns})
}

/// The plan a file records (see [`transforms_json`]), or `None` if `json` isn't a record of this
/// version.
pub fn plan_from_json(json: &serde_json::Value) -> Option<Plan> {
    if json["version"].as_u64()? != TRANSFORMS_VERSION as u64 {
        return None;
    }
    json["columns"]
        .as_object()?
        .iter()
        .map(|(name, entry)| {
            let k = || u8::try_from(entry["k"].as_u64()?).ok();
            let transform = match entry["transform"].as_str()? {
                "rescale" => Transform::Rescale(k()?),
                "ohlc" => Transform::Ohlc(k()?),
                "packed" => Transform::Packed,
                _ => return None,
            };
            Some((name.clone(), transform))
        })
        .collect()
}

pub fn decimal_params(field: &Field) -> (u8, i8) {
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
                None | Some(Transform::Packed) => field,
                Some(Transform::Rescale(k)) => {
                    let (precision, scale) = decimal_params(&field);
                    field.with_data_type(DataType::Decimal128(precision, scale - *k as i8))
                }
                // The struct goes where open was; the other prices are in it.
                Some(Transform::Ohlc(k)) if field.name() == OHLC_PRICES[0] => {
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
                None | Some(Transform::Packed) => return Some(Arc::clone(column)),
                Some(transform) => transform,
            };
            let (precision, scale) = decimal_params(field);
            let values = column.as_primitive::<Decimal128Type>();
            Some(match transform {
                Transform::Packed => unreachable!("returned unchanged above"),
                Transform::Ohlc(k) if field.name() == OHLC_PRICES[0] => {
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
    let [open, high, low, close] = OHLC_PRICES.map(|name| {
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
