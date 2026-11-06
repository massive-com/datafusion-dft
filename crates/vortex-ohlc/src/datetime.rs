//! A `between` kernel for Vortex's DateTimeParts ([`DateTimePartsBetween`]), which
//! `vortex-datetime-parts` 0.91 has none of: a window of time (`t >= a AND t < b`, which reaches
//! the scan as one `between`) otherwise decodes every timestamp of a chunk to compare it.
//!
//! DateTimeParts stores a timestamp as days, seconds into the day and subseconds, each with the
//! timestamp's sign, so a row is decided by its day unless that is one of the bounds': a day
//! strictly between theirs is inside, any other outside. A row on a bound's day is decided by its
//! time into the day, `seconds × units a second + subseconds`. Only those rows have their seconds
//! read, through their encodings' filter kernels: in a chunk of a month's minute bars, a window
//! within one day reads about one row in 23. Subseconds are read too unless they are a constant,
//! as minute bars' are.
//!
//! `t = c` is left to Vortex's compare kernel for DateTimeParts, which answers it before any kernel
//! a session registers can. Since Vortex 0.91 it evaluates each part once (vortex#10684), and
//! passing `=` to this kernel instead, as a `between`, measured no faster.
//!
//! *Upstream:* the kernel belongs in `vortex-datetime-parts` (a draft in this repo's
//! `site/upstream/datetime-parts-between/`).

use std::cmp::Ordering;
use std::ops::RangeInclusive;

use num_traits::AsPrimitive;
use vortex::array::arrays::scalar_fn::{ExactScalarFn, ScalarFnArrayView};
use vortex::array::arrays::{BoolArray, ConstantArray, PrimitiveArray};
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::extension::datetime::{TimeUnit, Timestamp};
use vortex::array::kernel::ExecuteParentKernel;
use vortex::array::optimizer::kernels::ArrayKernelsExt;
use vortex::array::scalar::Scalar;
use vortex::array::scalar_fn::fns::between::{Between, StrictComparison};
use vortex::array::scalar_fn::fns::operators::Operator;
use vortex::array::scalar_fn::ScalarFnVTable;
use vortex::array::{match_each_integer_ptype, ArrayRef, ArrayView, ExecutionCtx, IntoArray};
use vortex::buffer::BitBuffer;
use vortex::dtype::{DType, Nullability, PType};
use vortex::encodings::datetime_parts::{DateTimeParts, DateTimePartsArraySlotsExt};
use vortex::error::VortexResult;
use vortex::mask::Mask;
use vortex::session::VortexSession;

/// Register the kernel on `session`.
pub fn register(session: &VortexSession) {
    session.kernels().register_execute_parent_kernel(
        Between.id(),
        DateTimeParts,
        DateTimePartsBetween,
    );
}

/// `lower < t < upper` (or `<=` on either side) on a DateTimeParts array, against constant bounds:
/// decided on the days, and on the other parts only on the bounds' days.
#[derive(Debug)]
pub struct DateTimePartsBetween;

impl ExecuteParentKernel<DateTimeParts> for DateTimePartsBetween {
    type Parent = ExactScalarFn<Between>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, DateTimeParts>,
        parent: ScalarFnArrayView<'_, Between>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // The bounds are the parent's other two children.
        if child_idx != 0 {
            return Ok(None);
        }
        let children = parent.children();
        let (Some(per_second), Some(lower), Some(upper)) = (
            per_second(array.dtype()),
            bound(&children[1], array.dtype()),
            bound(&children[2], array.dtype()),
        ) else {
            return Ok(None);
        };
        // The bounds on a time into the day, made inclusive.
        let after = match parent.options.lower_strict {
            StrictComparison::Strict => lower.time + 1,
            StrictComparison::NonStrict => lower.time,
        };
        let before = match parent.options.upper_strict {
            StrictComparison::Strict => upper.time - 1,
            StrictComparison::NonStrict => upper.time,
        };

        let days = array.days();
        let mut result = match upper.day - lower.day >= 2 {
            true => {
                &days_where(days, Operator::Gt, lower.day, ctx)?
                    & &days_where(days, Operator::Lt, upper.day, ctx)?
            }
            false => Mask::new_false(array.len()),
        };
        // Each bound's day, and the times into the day its rows need.
        let bound_days = match lower.day.cmp(&upper.day) {
            Ordering::Less => vec![
                (lower.day, after..=i64::MAX),
                (upper.day, i64::MIN..=before),
            ],
            Ordering::Equal => vec![(lower.day, after..=before)],
            Ordering::Greater => vec![],
        };
        for (day, times) in bound_days {
            let on_day = days_where(days, Operator::Eq, day, ctx)?;
            if on_day.all_false() {
                continue;
            }
            let within = times_within(array, &on_day, per_second, times, ctx)?;
            result = result | &on_day.intersect_by_rank(&within);
        }
        // The days carry the array's nulls; a null row's result is null.
        let validity = array
            .validity()?
            .union_nullability(parent.dtype().nullability());
        Ok(Some(
            BoolArray::new(result.into_bit_buffer(), validity).into_array(),
        ))
    }
}

/// A bound, split as DateTimeParts splits a timestamp (`vortex-datetime-parts`' `split`, which it
/// doesn't export): its day, truncated toward zero, and its time into that day in the
/// timestamp's unit, which has the timestamp's sign.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Bound {
    day: i64,
    time: i64,
}

/// `bound` split, if it is a non-null constant timestamp of `dtype`.
fn bound(bound: &ArrayRef, dtype: &DType) -> Option<Bound> {
    let scalar = bound.as_constant()?;
    if !scalar.dtype().eq_ignore_nullability(dtype) {
        return None;
    }
    let day = 86_400 * per_second(dtype)?;
    let timestamp = (scalar.as_extension().to_storage_scalar().as_primitive()).as_::<i64>()?;
    Some(Bound {
        day: timestamp / day,
        time: timestamp % day,
    })
}

/// A timestamp dtype's units a second; `None` for a unit of days, which DateTimeParts doesn't
/// take, or another dtype.
fn per_second(dtype: &DType) -> Option<i64> {
    let DType::Extension(extension) = dtype else {
        return None;
    };
    match extension.metadata_opt::<Timestamp>()?.unit {
        TimeUnit::Nanoseconds => Some(1_000_000_000),
        TimeUnit::Microseconds => Some(1_000_000),
        TimeUnit::Milliseconds => Some(1_000),
        TimeUnit::Seconds => Some(1),
        TimeUnit::Days => None,
    }
}

/// Which of the rows of `on_day` have a time into the day within `times`, as a mask over those
/// rows. Only their seconds are read, and their subseconds unless those are a constant.
fn times_within(
    array: ArrayView<'_, DateTimeParts>,
    on_day: &Mask,
    per_second: i64,
    times: RangeInclusive<i64>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Mask> {
    let (mut first, mut last) = times.into_inner();
    let seconds = (array.seconds().filter(on_day.clone())?).execute::<PrimitiveArray>(ctx)?;
    let constant = (array.subseconds().as_constant()).and_then(|c| c.as_primitive().as_::<i64>());
    let subseconds = match constant {
        // A constant moves the bounds instead.
        Some(subsecond) => {
            (first, last) = (
                first.saturating_sub(subsecond),
                last.saturating_sub(subsecond),
            );
            None
        }
        None => Some(
            (array.subseconds().filter(on_day.clone())?)
                .cast(DType::Primitive(PType::I64, Nullability::NonNullable))?
                .execute::<PrimitiveArray>(ctx)?,
        ),
    };
    let within = match_each_integer_ptype!(seconds.ptype(), |S| {
        let seconds = seconds.as_slice::<S>();
        let time = |i: usize| -> i64 {
            let second: i64 = seconds[i].as_();
            second * per_second
        };
        match &subseconds {
            None => BitBuffer::collect_bool(seconds.len(), |i| (first..=last).contains(&time(i))),
            Some(subseconds) => {
                let subseconds = subseconds.as_slice::<i64>();
                BitBuffer::collect_bool(seconds.len(), |i| {
                    (first..=last).contains(&(time(i) + subseconds[i]))
                })
            }
        }
    });
    Ok(Mask::from_buffer(within))
}

/// The rows whose day is `op` `value`; a null day is not among them.
fn days_where(
    days: &ArrayRef,
    op: Operator,
    value: i64,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Mask> {
    let nullability = days.dtype().nullability();
    let wide = days.cast(DType::Primitive(PType::I64, nullability))?;
    let value = ConstantArray::new(Scalar::primitive(value, nullability), days.len());
    let result = wide.binary(value.into_array(), op)?;
    match nullability {
        Nullability::NonNullable => result,
        Nullability::Nullable => result.fill_null(false)?,
    }
    .execute::<Mask>(ctx)
}

#[cfg(test)]
mod tests {
    use vortex::array::arrays::bool::BoolArrayExt;
    use vortex::array::arrays::TemporalArray;
    use vortex::array::extension::datetime::TimestampOptions;
    use vortex::array::matcher::Matcher;
    use vortex::array::scalar_fn::fns::between::BetweenOptions;
    use vortex::array::validity::Validity;
    use vortex::array::VortexSessionExecute;
    use vortex::buffer::Buffer;
    use vortex::compressor::BtrBlocksCompressorBuilder;
    use vortex::encodings::datetime_parts::split_temporal;

    use super::*;
    use crate::testing::{encoding_ids, session};

    const STRICTNESS: [StrictComparison; 2] =
        [StrictComparison::Strict, StrictComparison::NonStrict];

    /// What the kernel gives for `array` between `lower` and `upper`, called directly, as
    /// `Some(value)` a row and `None` for a null row; `None` if it leaves the bounds to Vortex.
    fn kernel_between(
        array: &ArrayRef,
        [lower, upper]: [i64; 2],
        options: BetweenOptions,
        ctx: &mut ExecutionCtx,
    ) -> Option<Vec<Option<bool>>> {
        let DType::Extension(extension) = array.dtype() else {
            panic!("a timestamp array");
        };
        let unit = extension.metadata_opt::<Timestamp>().unwrap().unit;
        let bound = |t: i64| {
            let t = Scalar::primitive(t, Nullability::NonNullable);
            let t = Scalar::extension::<Timestamp>(TimestampOptions { unit, tz: None }, t);
            ConstantArray::new(t, array.len()).into_array()
        };
        let parent = Between::try_new(array.clone(), bound(lower), bound(upper), options)
            .unwrap()
            .into_array();
        let between = ExactScalarFn::<Between>::try_match(&parent).unwrap();
        let child = array
            .as_opt::<DateTimeParts>()
            .expect("a DateTimeParts array");
        let result = (DateTimePartsBetween.execute_parent(child, between, 0, ctx)).unwrap()?;
        let result = result.execute::<BoolArray>(ctx).unwrap();
        let valid = (result.validity().unwrap())
            .execute_mask(result.len(), ctx)
            .unwrap();
        let values = result.to_bit_buffer();
        Some(
            (0..result.len())
                .map(|i| valid.value(i).then(|| values.value(i)))
                .collect(),
        )
    }

    /// For every pair of `bounds` and every strictness, the kernel answers, as the timestamps do.
    fn between_matches_timestamps(
        array: &ArrayRef,
        timestamps: &[Option<i64>],
        bounds: &[i64],
        ctx: &mut ExecutionCtx,
    ) {
        for &lower in bounds {
            for &upper in bounds {
                for lower_strict in STRICTNESS {
                    for upper_strict in STRICTNESS {
                        let options = BetweenOptions {
                            lower_strict,
                            upper_strict,
                        };
                        let expected: Vec<Option<bool>> = (timestamps.iter())
                            .map(|t| {
                                t.map(|t| {
                                    let above = match lower_strict {
                                        StrictComparison::Strict => t > lower,
                                        StrictComparison::NonStrict => t >= lower,
                                    };
                                    let below = match upper_strict {
                                        StrictComparison::Strict => t < upper,
                                        StrictComparison::NonStrict => t <= upper,
                                    };
                                    above && below
                                })
                            })
                            .collect();
                        let result = kernel_between(array, [lower, upper], options.clone(), ctx);
                        assert_eq!(result, Some(expected), "{lower} {upper} {options:?}");
                    }
                }
            }
        }
    }

    /// `timestamps` in `unit`, compressed as the writer compresses them, and their subseconds'
    /// constant, if they are one.
    fn compressed(
        timestamps: &[i64],
        unit: TimeUnit,
        session: &VortexSession,
        ctx: &mut ExecutionCtx,
    ) -> (ArrayRef, Option<i64>) {
        let array = TemporalArray::new_timestamp(
            PrimitiveArray::from_iter(timestamps.iter().copied()).into_array(),
            unit,
            None,
        );
        let compressor = BtrBlocksCompressorBuilder::from_session(session).build();
        let array = compressor.compress(&array.into_array(), ctx).unwrap();
        let ids = encoding_ids(&array);
        assert_eq!(ids[0], "vortex.datetimeparts", "{ids:?}");
        let subseconds = array
            .as_opt::<DateTimeParts>()
            .unwrap()
            .subseconds()
            .clone();
        let constant = (subseconds.as_constant()).and_then(|c| c.as_primitive().as_::<i64>());
        (array, constant)
    }

    /// Minute bars' timestamps as the writer stores them (the days run-end encoded, the seconds
    /// divided by 60, the subseconds a constant zero), against windows within a day, across days,
    /// and on a bar's minute.
    #[test]
    fn minute_bars_between() {
        let session = session();
        let mut ctx = session.create_execution_ctx();
        // Five days of a ticker's bars from 04:00 to 20:00, in seconds.
        let day = 1_155_600_000 / 86_400 * 86_400;
        let timestamps: Vec<i64> = (0..5)
            .flat_map(|d| (240..1200).map(move |m| day + d * 86_400 + m * 60))
            .collect();
        let (array, constant) = compressed(&timestamps, TimeUnit::Seconds, &session, &mut ctx);
        assert_eq!(constant, Some(0));

        let at = |d: i64, h: i64, m: i64| day + d * 86_400 + h * 3600 + m * 60;
        let bounds = [
            at(1, 14, 30),
            at(1, 14, 35),
            at(1, 14, 30) + 1,
            at(2, 9, 30),
            at(4, 16, 0),
            day - 1,
            at(5, 0, 0),
        ];
        let timestamps: Vec<Option<i64>> = timestamps.into_iter().map(Some).collect();
        between_matches_timestamps(&array, &timestamps, &bounds, &mut ctx);
    }

    /// A constant subsecond other than zero (every bar a quarter past its second) moves the
    /// bounds, which then fall either side of a bar or on it.
    #[test]
    fn constant_subseconds_between() {
        let session = session();
        let mut ctx = session.create_execution_ctx();
        let day = 1_155_600_000_000 / 86_400_000 * 86_400_000;
        let timestamps: Vec<i64> = (0..3)
            .flat_map(|d| (240..1200).map(move |m| day + d * 86_400_000 + m * 60_000 + 250))
            .collect();
        let (array, constant) = compressed(&timestamps, TimeUnit::Milliseconds, &session, &mut ctx);
        assert_eq!(constant, Some(250));

        let bounds: Vec<i64> = [timestamps[700], timestamps[1500]]
            .into_iter()
            .flat_map(|t| [t - 251, t - 250, t - 1, t, t + 1])
            .collect();
        let timestamps: Vec<Option<i64>> = timestamps.into_iter().map(Some).collect();
        between_matches_timestamps(&array, &timestamps, &bounds, &mut ctx);
    }

    /// Nanoseconds either side of the epoch, where the parts are negative, with nulls. The
    /// subseconds vary, so they are read.
    #[test]
    fn nanoseconds_around_the_epoch_between() {
        let session = session();
        let mut ctx = session.create_execution_ctx();
        const SECOND: i64 = 1_000_000_000;
        const DAY: i64 = 86_400 * SECOND;
        let values: Vec<i64> = [-2 * DAY, -DAY - 1, -DAY, -SECOND / 2, -1, 0, 1, SECOND / 2]
            .into_iter()
            .chain([DAY - 1, DAY, DAY + 3 * SECOND + 7, 3 * DAY + 17])
            .collect();
        let valid: Vec<bool> = (0..values.len()).map(|i| i % 5 != 3).collect();
        let timestamps: Vec<Option<i64>> = (values.iter().zip(&valid))
            .map(|(&t, &valid)| valid.then_some(t))
            .collect();
        let storage = PrimitiveArray::new(
            values.iter().copied().collect::<Buffer<i64>>(),
            Validity::from_iter(valid),
        );
        let array = TemporalArray::new_timestamp(storage.into_array(), TimeUnit::Nanoseconds, None);
        let dtype = array.dtype().clone();
        let parts = split_temporal(array, &mut ctx).unwrap();
        assert!(parts.subseconds.as_constant().is_none());
        let array = DateTimeParts::try_new(dtype, parts.days, parts.seconds, parts.subseconds)
            .unwrap()
            .into_array();

        let bounds: Vec<i64> = (values.iter())
            .flat_map(|&t| [t - 1, t, t + 1])
            .chain([i64::MIN / 2, i64::MAX / 2])
            .collect();
        between_matches_timestamps(&array, &timestamps, &bounds, &mut ctx);
    }

    /// Bounds the kernel can't read are left to Vortex: not constants, or not timestamps of the
    /// array's unit.
    #[test]
    fn other_bounds_are_left_to_vortex() {
        let mut ctx = session().create_execution_ctx();
        let array = TemporalArray::new_timestamp(
            PrimitiveArray::from_iter([0i64, 60, 120]).into_array(),
            TimeUnit::Seconds,
            None,
        );
        let dtype = array.dtype().clone();
        let parts = split_temporal(array, &mut ctx).unwrap();
        let array = DateTimeParts::try_new(dtype, parts.days, parts.seconds, parts.subseconds)
            .unwrap()
            .into_array();
        let millis = Scalar::extension::<Timestamp>(
            TimestampOptions {
                unit: TimeUnit::Milliseconds,
                tz: None,
            },
            Scalar::primitive(60_000i64, Nullability::NonNullable),
        );
        assert_eq!(
            bound(&ConstantArray::new(millis, 3).into_array(), array.dtype()),
            None
        );
        assert_eq!(bound(&array, array.dtype()), None);
    }
}
