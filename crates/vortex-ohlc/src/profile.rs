//! Measuring a source's decimal columns (trailing zero digits, the largest unscaled value,
//! nulls), and the plan that follows from that: how far each can be rescaled, and whether the
//! four prices can be stored as the [`OHLC`](crate::plan::OHLC) struct.

use arrow::array::cast::AsArray;
use arrow::array::types::Decimal128Type;
use arrow::array::{Array, RecordBatch};
use arrow::datatypes::{DataType, Schema};

use crate::plan::{Plan, Transform, OHLC_PRECISION, OHLC_PRICES};

/// What one decimal column's unscaled integers look like.
#[derive(Clone, Debug)]
pub struct DecimalProfile {
    pub name: String,
    pub precision: u8,
    pub scale: i8,
    pub nulls: u64,
    pub nonzero: u64,
    /// `at_least[j]`: non-zero values with at least `j` trailing zero digits, for `j` in
    /// `0..=scale`. Zeros are left out - they divide by any power of ten.
    pub at_least: Vec<u64>,
    pub max_abs: u128,
}

impl DecimalProfile {
    /// The largest `k` such that every value is a multiple of `10^k` - how far the scale can
    /// drop without losing anything.
    pub fn lossless_k(&self) -> u8 {
        (0..self.at_least.len())
            .rev()
            .find(|&j| self.at_least[j] == self.nonzero)
            .unwrap_or(0) as u8
    }
}

/// Profiles each `Decimal128` column of a schema, one batch at a time.
pub struct DecimalProfiler {
    /// The profiled columns' indices in the schema.
    columns: Vec<usize>,
    /// While counting, `at_least[j]` holds the values with exactly `j` trailing zero digits (or
    /// at least `scale`); [`Self::finish`] makes it cumulative.
    profiles: Vec<DecimalProfile>,
}

impl DecimalProfiler {
    pub fn new(schema: &Schema) -> Self {
        let (columns, profiles) = schema
            .fields()
            .iter()
            .enumerate()
            .filter_map(|(i, f)| match f.data_type() {
                DataType::Decimal128(precision, scale) => Some((
                    i,
                    DecimalProfile {
                        name: f.name().clone(),
                        precision: *precision,
                        scale: *scale,
                        nulls: 0,
                        nonzero: 0,
                        at_least: vec![0; (*scale).max(0) as usize + 1],
                        max_abs: 0,
                    },
                )),
                _ => None,
            })
            .unzip();
        Self { columns, profiles }
    }

    /// Count `batch`'s values. Its schema must be the one the profiler was made for.
    pub fn add(&mut self, batch: &RecordBatch) {
        for (profile, &i) in self.profiles.iter_mut().zip(&self.columns) {
            let max_k = profile.at_least.len() - 1;
            let mut record = |v: i128| {
                let v = v.unsigned_abs();
                profile.max_abs = profile.max_abs.max(v);
                if v != 0 {
                    profile.nonzero += 1;
                    // Counted cumulatively in `finish`: a value with 3 zeros has at least 0, 1,
                    // 2 and 3.
                    profile.at_least[trailing_zero_digits(v, max_k)] += 1;
                }
            };
            let values = batch.column(i).as_primitive::<Decimal128Type>();
            if values.null_count() == 0 {
                values.values().iter().for_each(|&v| record(v));
            } else {
                values.iter().flatten().for_each(&mut record);
                profile.nulls += values.null_count() as u64;
            }
        }
    }

    pub fn finish(mut self) -> Vec<DecimalProfile> {
        for profile in &mut self.profiles {
            for j in (0..profile.at_least.len() - 1).rev() {
                profile.at_least[j] += profile.at_least[j + 1];
            }
        }
        self.profiles
    }
}

/// The plan for a source whose decimal columns measured `profiles`: each column rescaled by its
/// [`lossless_k`](DecimalProfile::lossless_k), and the four prices, where the source has them,
/// as the [`OHLC`](crate::plan::OHLC) struct, at the `k` all four allow. That takes no nulls,
/// one scale, and parts that fit [`OHLC_PRECISION`]; otherwise the prices are only rescaled.
///
/// Integer columns aren't profiled; mark the ones to pack with [`Transform::Packed`].
pub fn plan(profiles: &[DecimalProfile]) -> Plan {
    let mut plan: Plan = profiles
        .iter()
        .filter(|p| p.lossless_k() > 0)
        .map(|p| (p.name.clone(), Transform::Rescale(p.lossless_k())))
        .collect();
    let prices: Option<Vec<&DecimalProfile>> = OHLC_PRICES
        .iter()
        .map(|&price| profiles.iter().find(|p| p.name == price))
        .collect();
    if let Some(k) = prices.and_then(|prices| ohlc_k(&prices)) {
        for price in OHLC_PRICES {
            plan.insert(price.to_string(), Transform::Ohlc(k));
        }
    }
    plan
}

/// The `k` the four `prices` can be stored as the OHLC struct at, if they can.
fn ohlc_k(prices: &[&DecimalProfile]) -> Option<u8> {
    let scale = prices[0].scale;
    if prices.iter().any(|p| p.nulls > 0 || p.scale != scale) {
        return None;
    }
    let k = prices.iter().map(|p| p.lossless_k()).min()?;
    // Each part is a price or the difference of two, so at most twice the largest price.
    let max_abs = prices.iter().map(|p| p.max_abs).max()?;
    let part_max = max_abs / 10u128.pow(k as u32) * 2;
    (part_max < 10u128.pow(OHLC_PRECISION as u32)).then_some(k)
}

/// How many decimal zeros `v` ends in, up to `max`.
///
/// 128-bit division is a slow library call, so it is only used to peel digits off values too
/// big for a `u64`; everything else (nearly all of them) is counted with 64-bit arithmetic.
fn trailing_zero_digits(v: u128, max: usize) -> usize {
    let (mut v, mut zeros) = (v, 0);
    while v > u64::MAX as u128 {
        if zeros == max || !v.is_multiple_of(10) {
            return zeros;
        }
        v /= 10;
        zeros += 1;
    }
    let mut v = v as u64;
    while zeros < max && v.is_multiple_of(10) {
        v /= 10;
        zeros += 1;
    }
    zeros
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Decimal128Array};
    use arrow::datatypes::Field;

    use super::*;

    fn decimal(values: Vec<Option<i128>>, scale: i8) -> ArrayRef {
        Arc::new(
            Decimal128Array::from(values)
                .with_precision_and_scale(38, scale)
                .unwrap(),
        )
    }

    /// Profile one batch of `columns` and plan from it.
    fn plan_of(columns: Vec<(&str, ArrayRef)>) -> Plan {
        let fields: Vec<Field> = columns
            .iter()
            .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            columns.into_iter().map(|c| c.1).collect(),
        )
        .unwrap();
        let mut profiler = DecimalProfiler::new(&schema);
        profiler.add(&batch);
        plan(&profiler.finish())
    }

    /// Two bars at scale 4 in whole dollars (so `k` = 4), with `odd_low` one low of 98.5 (so
    /// `k` = 3), and with `null_close` one close missing.
    fn prices(odd_low: bool, null_close: bool) -> Vec<(&'static str, ArrayRef)> {
        let low = if odd_low { 985_000 } else { 980_000 };
        let close = if null_close { None } else { Some(1_010_000) };
        vec![
            ("open", decimal(vec![Some(1_000_000), Some(990_000)], 4)),
            ("high", decimal(vec![Some(1_020_000), Some(1_000_000)], 4)),
            ("low", decimal(vec![Some(990_000), Some(low)], 4)),
            ("close", decimal(vec![close, Some(990_000)], 4)),
        ]
    }

    #[test]
    fn rescales_by_the_trailing_zeros_every_value_has() {
        let plan = plan_of(vec![
            (
                "volume",
                decimal(vec![Some(5_000_000), Some(0), Some(-120_000_000)], 6),
            ),
            ("odd", decimal(vec![Some(10), Some(7), None], 2)),
        ]);
        assert_eq!(plan.get("volume"), Some(&Transform::Rescale(6)));
        assert_eq!(plan.get("odd"), None);
    }

    #[test]
    fn prices_become_the_ohlc_struct_at_the_smallest_k() {
        let plan = plan_of(prices(false, false));
        assert!(OHLC_PRICES
            .iter()
            .all(|p| plan.get(*p) == Some(&Transform::Ohlc(4))));
        let plan = plan_of(prices(true, false));
        assert!(OHLC_PRICES
            .iter()
            .all(|p| plan.get(*p) == Some(&Transform::Ohlc(3))));
    }

    #[test]
    fn null_prices_are_only_rescaled() {
        let plan = plan_of(prices(false, true));
        assert_eq!(plan.get("open"), Some(&Transform::Rescale(4)));
        assert_eq!(plan.get("close"), Some(&Transform::Rescale(4)));
    }

    #[test]
    fn prices_too_wide_for_the_parts_are_only_rescaled() {
        let wide = decimal(vec![Some(10i128.pow(20))], 0);
        let plan = plan_of(vec![
            ("open", Arc::clone(&wide)),
            ("high", Arc::clone(&wide)),
            ("low", Arc::clone(&wide)),
            ("close", wide),
        ]);
        assert_eq!(plan.get("open"), None);
    }
}
