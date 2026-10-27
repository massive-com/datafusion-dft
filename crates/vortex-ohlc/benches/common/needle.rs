//! Needle-in-a-haystack lookups: a handful of selective filters, run against Parquet (row group
//! statistics + a row filter) and against the Vortex file (a scan filter, pruned on zone maps).
//! Every needle returns all columns, so the results can be compared value for value across
//! formats.

use std::fs::File;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef as ArrowArrayRef, BooleanArray, Datum, Decimal128Array, Int64Array,
    RecordBatch, Scalar, StringArray,
};
use arrow::compute::kernels::cmp;
use arrow::compute::{cast, filter_record_batch};
use arrow::datatypes::DataType;
use arrow::error::ArrowError;
use parquet::arrow::arrow_reader::{
    ArrowPredicate, ArrowPredicateFn, ParquetRecordBatchReaderBuilder, RowFilter,
};
use parquet::arrow::ProjectionMask;
use parquet::file::statistics::{Statistics, ValueStatistics};
use vortex::dtype::{DType, DecimalDType, Nullability};
use vortex::expr::{
    and_collect, binary, cast as vortex_cast, checked_add, eq, get_item, gt, gt_eq, lit, lt, root,
    zip_expr, BoundExpression, Expression,
};
use vortex::file::VortexFile;
use vortex::scalar::{DecimalValue, Scalar as VortexScalar};
use vortex::scalar_fn::fns::operators::Operator;
use vortex_ohlc::plan::{Plan, Transform, OHLC, OHLC_PARTS};

/// How a term compares its column with its value.
#[derive(Clone, Copy, Debug)]
pub enum Op {
    Eq,
    Gt,
    /// `>=` and `<`: the two ends of a window of time or of prices.
    Ge,
    Lt,
}

impl Op {
    /// `column op literal`, row by row, in Arrow.
    fn arrow(self, column: &dyn Array, literal: &dyn Datum) -> Result<BooleanArray, ArrowError> {
        match self {
            Op::Eq => cmp::eq(&column, literal),
            Op::Gt => cmp::gt(&column, literal),
            Op::Ge => cmp::gt_eq(&column, literal),
            Op::Lt => cmp::lt(&column, literal),
        }
    }

    fn vortex(self, lhs: Expression, rhs: Expression) -> Expression {
        match self {
            Op::Eq => eq(lhs, rhs),
            Op::Gt => gt(lhs, rhs),
            Op::Ge => gt_eq(lhs, rhs),
            Op::Lt => lt(lhs, rhs),
        }
    }
}

/// A literal, in a form that doesn't depend on how either format stores the column.
#[derive(Clone, Copy, Debug)]
pub enum Value {
    Str(&'static str),
    /// Nanoseconds since the epoch.
    TimestampNs(i64),
    /// An exact decimal: `unscaled / 10^scale`.
    Decimal {
        unscaled: i128,
        scale: i8,
    },
}

impl Value {
    /// A decimal literal's unscaled value at `scale`, or `None` if that would round it.
    fn unscaled_at(&self, scale: i8) -> Option<i128> {
        let Value::Decimal {
            unscaled,
            scale: own,
        } = *self
        else {
            panic!("{self:?} is not a decimal");
        };
        (scale >= own).then(|| unscaled * 10i128.pow((scale - own) as u32))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Term {
    pub column: &'static str,
    pub op: Op,
    pub value: Value,
}

pub struct Needle {
    pub name: &'static str,
    /// All of them must hold.
    pub terms: Vec<Term>,
}

/// The lookups, picked from the default source (August 2006, every ticker from A to ZZ, sorted
/// by ticker):
///
/// - `AAPL`: one ticker, 11,922 bars.
/// - `AAPL @ 14:30`: one bar, 2006-08-15 14:30 America/New_York.
/// - `AAPL 14:30-15:30` and `AAPL 08-15 session` (09:30-16:00): one ticker over a window of
///   time on that day, the shape of most production queries.
/// - `all @ 14:30` and `all 14:30-14:35`: every ticker's bars in that minute or those five, one
///   per ticker and so spread through every chunk.
/// - `AAPM (miss)`: a ticker that sorts between two real ones (`AAPL`, `AAR`) but doesn't
///   exist, so only min/max statistics can rule data out.
/// - `high > 1000`: a range predicate on a decimal column, 6,540 bars (mostly BRK.A / BRK.B).
/// - `low > 1000` and `close > 400.5`: the same on the other prices, 6,540 and 22,948 bars.
///   With the prices stored against each other ([`OHLC`]), each price filters through a
///   different expression over the parts (see [`ohlc_price`]).
pub fn needles() -> Vec<Needle> {
    const NS: i64 = 1_000_000_000;
    // 2006-08-15 14:30 America/New_York, in seconds since the epoch.
    const AUG_15_1430: i64 = 1_155_666_600;
    let ticker = |t| Term {
        column: "ticker",
        op: Op::Eq,
        value: Value::Str(t),
    };
    let above = |column, unscaled, scale| Term {
        column,
        op: Op::Gt,
        value: Value::Decimal { unscaled, scale },
    };
    let at = |column, unscaled, scale| Term {
        column,
        op: Op::Eq,
        value: Value::Decimal { unscaled, scale },
    };
    // `from <= column < to`, in whole units.
    let range = |column, from, to| {
        vec![
            Term {
                column,
                op: Op::Ge,
                value: Value::Decimal {
                    unscaled: from,
                    scale: 0,
                },
            },
            Term {
                column,
                op: Op::Lt,
                value: Value::Decimal {
                    unscaled: to,
                    scale: 0,
                },
            },
        ]
    };
    // `from <= window_start < to`, in seconds since the epoch.
    let window = |from: i64, to: i64| {
        vec![
            Term {
                column: "window_start",
                op: Op::Ge,
                value: Value::TimestampNs(from * NS),
            },
            Term {
                column: "window_start",
                op: Op::Lt,
                value: Value::TimestampNs(to * NS),
            },
        ]
    };
    vec![
        Needle {
            name: "AAPL",
            terms: vec![ticker("AAPL")],
        },
        Needle {
            name: "AAPL @ 14:30",
            terms: vec![
                ticker("AAPL"),
                Term {
                    column: "window_start",
                    op: Op::Eq,
                    value: Value::TimestampNs(AUG_15_1430 * NS),
                },
            ],
        },
        // A ticker over a window of time, as production queries mostly are, and windows over
        // every ticker: one bar each, scattered through every chunk.
        Needle {
            name: "AAPL 14:30-15:30",
            terms: [
                vec![ticker("AAPL")],
                window(AUG_15_1430, AUG_15_1430 + 3600),
            ]
            .concat(),
        },
        Needle {
            name: "AAPL 08-15 session",
            terms: [
                vec![ticker("AAPL")],
                window(AUG_15_1430 - 5 * 3600, AUG_15_1430 + 5400),
            ]
            .concat(),
        },
        Needle {
            name: "all @ 14:30",
            terms: vec![Term {
                column: "window_start",
                op: Op::Eq,
                value: Value::TimestampNs(AUG_15_1430 * NS),
            }],
        },
        Needle {
            name: "all 14:30-14:35",
            terms: window(AUG_15_1430, AUG_15_1430 + 300),
        },
        Needle {
            name: "AAPM (miss)",
            terms: vec![ticker("AAPM")],
        },
        Needle {
            name: "high > 1000",
            terms: vec![above("high", 1000, 0)],
        },
        Needle {
            name: "low > 1000",
            terms: vec![above("low", 1000, 0)],
        },
        Needle {
            name: "close > 400.5",
            terms: vec![above("close", 4005, 1)],
        },
        // Ranges of prices, `from <= price < to`. Vortex's scan merges open's two ends into one
        // `between` (open is a field of the OHLC struct); close's are on a sum of parts, and
        // stay two comparisons.
        Needle {
            name: "open 400-401",
            terms: range("open", 400, 401),
        },
        Needle {
            name: "close 400-401",
            terms: range("close", 400, 401),
        },
        // Equalities on high and low, which need the body's sign to pick open or close.
        Needle {
            name: "high = 50",
            terms: vec![at("high", 50, 0)],
        },
        Needle {
            name: "low = 50",
            terms: vec![at("low", 50, 0)],
        },
        // A ticker and a price together: the price term only needs deciding where the ticker's
        // rows are.
        Needle {
            name: "AAPL & close > 67",
            terms: vec![ticker("AAPL"), above("close", 67, 0)],
        },
        // dollar_volume above its 99.99th, 99.9th, 99th and 90th percentiles (2,526, 25,275,
        // 252,767 and 2,527,683 bars), and above AAPL's own 99th (118 bars).
        Needle {
            name: "dv > 32041624",
            terms: vec![above("dollar_volume", 32_041_624, 0)],
        },
        Needle {
            name: "dv > 7618557",
            terms: vec![above("dollar_volume", 7_618_557, 0)],
        },
        Needle {
            name: "dv > 1529338",
            terms: vec![above("dollar_volume", 1_529_338, 0)],
        },
        Needle {
            name: "dv > 205884",
            terms: vec![above("dollar_volume", 205_884, 0)],
        },
        Needle {
            name: "AAPL & dv > 19656066",
            terms: vec![ticker("AAPL"), above("dollar_volume", 19_656_066, 0)],
        },
    ]
}

// ---------------------------------------------------------------------------------------------
// Parquet
// ---------------------------------------------------------------------------------------------

/// Restrict a reader to `needle`: drop the row groups whose statistics rule it out, and filter
/// the rest row by row. Each term's column is decoded first and the other columns only for the
/// rows that pass.
pub fn parquet_needle(
    builder: ParquetRecordBatchReaderBuilder<File>,
    needle: &Needle,
) -> ParquetRecordBatchReaderBuilder<File> {
    let row_groups = candidate_row_groups(&builder, &needle.terms);
    let filter = row_filter(&builder, &needle.terms);
    builder.with_row_groups(row_groups).with_row_filter(filter)
}

fn leaf(builder: &ParquetRecordBatchReaderBuilder<File>, column: &str) -> usize {
    builder
        .parquet_schema()
        .columns()
        .iter()
        .position(|c| c.path().parts() == [column])
        .unwrap_or_else(|| panic!("no {column} column"))
}

/// The decimal scale of `data_type`.
fn decimal_scale(data_type: &DataType) -> i8 {
    match data_type {
        DataType::Decimal32(_, s)
        | DataType::Decimal64(_, s)
        | DataType::Decimal128(_, s)
        | DataType::Decimal256(_, s) => *s,
        other => panic!("{other} is not a decimal"),
    }
}

/// Could a row group whose column has these statistics hold a row matching `term`? Anything the
/// statistics can't answer counts as a yes.
fn may_hold(stats: Option<&Statistics>, term: &Term, scale: Option<i8>) -> bool {
    let Some(stats) = stats else {
        return true;
    };
    match (stats, term.value) {
        (_, Value::Str(s)) => {
            let s = s.as_bytes();
            let above_min = stats.min_bytes_opt().is_none_or(|min| s >= min);
            let below_max = stats.max_bytes_opt().is_none_or(|max| s <= max);
            match term.op {
                Op::Eq => above_min && below_max,
                Op::Gt => stats.max_bytes_opt().is_none_or(|max| max > s),
                Op::Ge => stats.max_bytes_opt().is_none_or(|max| max >= s),
                Op::Lt => stats.min_bytes_opt().is_none_or(|min| min < s),
            }
        }
        (Statistics::Int64(stats), Value::TimestampNs(v)) => in_range(stats, v, term.op),
        // INT64-backed decimals: the statistics are unscaled, at the column's scale.
        (Statistics::Int64(stats), value @ Value::Decimal { .. }) => {
            let v = value.unscaled_at(scale.expect("decimal column has a scale"));
            v.and_then(|v| i64::try_from(v).ok())
                .is_none_or(|v| in_range(stats, v, term.op))
        }
        _ => true,
    }
}

fn in_range(stats: &ValueStatistics<i64>, v: i64, op: Op) -> bool {
    match op {
        Op::Eq => {
            stats.min_opt().is_none_or(|min| v >= *min)
                && stats.max_opt().is_none_or(|max| v <= *max)
        }
        Op::Gt => stats.max_opt().is_none_or(|max| *max > v),
        Op::Ge => stats.max_opt().is_none_or(|max| *max >= v),
        Op::Lt => stats.min_opt().is_none_or(|min| *min < v),
    }
}

pub fn candidate_row_groups(
    builder: &ParquetRecordBatchReaderBuilder<File>,
    terms: &[Term],
) -> Vec<usize> {
    let leaves: Vec<(usize, &Term, Option<i8>)> = terms
        .iter()
        .map(|term| {
            let scale = matches!(term.value, Value::Decimal { .. }).then(|| {
                let field = builder.schema().field_with_name(term.column).unwrap();
                decimal_scale(field.data_type())
            });
            (leaf(builder, term.column), term, scale)
        })
        .collect();
    builder
        .metadata()
        .row_groups()
        .iter()
        .enumerate()
        .filter(|(_, rg)| {
            leaves
                .iter()
                .all(|(leaf, term, scale)| may_hold(rg.column(*leaf).statistics(), term, *scale))
        })
        .map(|(idx, _)| idx)
        .collect()
}

fn row_filter(builder: &ParquetRecordBatchReaderBuilder<File>, terms: &[Term]) -> RowFilter {
    let predicates = terms
        .iter()
        .map(|term| {
            // Compare in whatever type the reader decodes the column to (Utf8View, a timestamp
            // with its time zone, or a decimal).
            let data_type = builder
                .schema()
                .field_with_name(term.column)
                .unwrap()
                .data_type()
                .clone();
            let literal = arrow_literal(term.value, &data_type);
            let op = term.op;
            let predicate = ArrowPredicateFn::new(
                ProjectionMask::leaves(builder.parquet_schema(), [leaf(builder, term.column)]),
                move |batch: RecordBatch| -> Result<BooleanArray, _> {
                    op.arrow(batch.column(0), &literal)
                },
            );
            Box::new(predicate) as Box<dyn ArrowPredicate>
        })
        .collect();
    RowFilter::new(predicates)
}

/// `value` as a one-element Arrow scalar of `data_type` (Utf8View, a timestamp with its time
/// zone, or a decimal at that type's scale), for comparing against arrays of that type.
fn arrow_literal(value: Value, data_type: &DataType) -> Scalar<ArrowArrayRef> {
    let literal: ArrowArrayRef = match value {
        Value::Str(s) => Arc::new(StringArray::from(vec![s])),
        Value::TimestampNs(ns) => Arc::new(Int64Array::from(vec![ns])),
        value @ Value::Decimal { .. } => {
            let scale = decimal_scale(data_type);
            let unscaled = value
                .unscaled_at(scale)
                .unwrap_or_else(|| panic!("{value:?} doesn't fit a scale of {scale}"));
            Arc::new(
                Decimal128Array::from(vec![unscaled])
                    .with_precision_and_scale(38, scale)
                    .unwrap(),
            )
        }
    };
    Scalar::new(cast(&literal, data_type).unwrap())
}

/// The rows of a decoded batch that match every one of `terms`, compared in Arrow in each
/// column's own type.
pub fn filter_rows(mut batch: RecordBatch, terms: &[Term]) -> RecordBatch {
    for term in terms {
        let column: &dyn Array = batch
            .column_by_name(term.column)
            .unwrap_or_else(|| panic!("no {} column to filter on", term.column))
            .as_ref();
        let literal = arrow_literal(term.value, column.data_type());
        let mask = term.op.arrow(column, &literal).unwrap();
        batch = filter_record_batch(&batch, &mask).unwrap();
    }
    batch
}

// ---------------------------------------------------------------------------------------------
// Vortex
// ---------------------------------------------------------------------------------------------

/// `needle` as one Vortex filter expression over `file`, and the terms it leaves out, to check
/// with [`filter_rows`] after decoding: those whose literal its column's stored type can't hold
/// exactly. `None` when no term can go into the filter. Literals are built in each column's own
/// stored type, so a rescaled decimal column gets a literal at its rescaled scale, and a price
/// stored in the [`OHLC`] struct is compared as [`ohlc_price`] rebuilds it.
pub fn vortex_filter(
    file: &VortexFile,
    needle: &Needle,
    plan: &Plan,
) -> (Option<BoundExpression>, Vec<Term>) {
    let fields = file.dtype().as_struct_fields();
    let mut after = Vec::new();
    let mut conjuncts = Vec::new();
    for term in &needle.terms {
        let pushed = match plan.get(term.column) {
            Some(Transform::Ohlc(_)) => ohlc_term(file.dtype(), term),
            _ => {
                let dtype = fields.field(term.column).unwrap();
                vortex_literal(term.value, &dtype)
                    .map(|literal| term.op.vortex(get_item(term.column, root()), literal))
            }
        };
        match pushed {
            Some(expr) => conjuncts.push(expr),
            None => after.push(*term),
        }
    }
    let filter = and_collect(conjuncts).map(|filter| {
        filter
            .bind(file.dtype())
            .unwrap()
            .optimize_recursive()
            .unwrap()
    });
    (filter, after)
}

/// `value` as a literal of `dtype`, or `None` if `dtype` can't hold it exactly.
fn vortex_literal(value: Value, dtype: &DType) -> Option<Expression> {
    Some(match value {
        Value::Str(s) => lit(s),
        // Timestamps are an extension type over i64; the literal has to be one too.
        Value::TimestampNs(ns) => {
            let ext = dtype.as_extension();
            let storage = VortexScalar::primitive(ns, ext.storage_dtype().nullability());
            lit(VortexScalar::extension_ref(ext.clone(), storage))
        }
        value @ Value::Decimal { .. } => {
            let DType::Decimal(decimal, nullability) = dtype else {
                panic!("{dtype} is not a decimal");
            };
            let v = DecimalValue::I128(value.unscaled_at(decimal.scale())?);
            if !v.fits_in_precision(*decimal) {
                return None;
            }
            lit(VortexScalar::decimal(v, *decimal, *nullability))
        }
    })
}

/// A term on a price stored in the [`OHLC`] struct, as one comparison of the price
/// [`ohlc_price`] rebuilds from the parts; `None` if the literal doesn't fit its type.
fn ohlc_term(dtype: &DType, term: &Term) -> Option<Expression> {
    let (price, precision, scale) = ohlc_price(dtype, term.column)?;
    let decimal = DType::Decimal(
        DecimalDType::new(precision, scale),
        Nullability::NonNullable,
    );
    let literal = vortex_literal(term.value, &decimal)?;
    Some(term.op.vortex(price, literal))
}

/// A price as an expression over the [`OHLC`] struct's parts (in a table of `dtype`), with only
/// Vortex's own functions, and that expression's precision and scale; `None` for anything but
/// the four prices. close = open + body, high = max(open, close) + upper and low = min(open,
/// close) - lower, where the body's sign picks the larger of open and close (`zip`).
///
/// Vortex only adds and compares decimals of one type, and a sum is one digit wider than its
/// operands, so each operand is cast to the precision its place needs: the parts start at their
/// own precision `p` ([`vortex_ohlc::plan::OHLC_PRECISION`]), a sum of two at `p + 1` and of
/// three at `p + 2`, at most 18 digits, so every value is an i64.
pub fn ohlc_price(dtype: &DType, price: &str) -> Option<(Expression, u8, i8)> {
    let fields = dtype.as_struct_fields();
    let parts = fields.field(OHLC)?;
    let DType::Decimal(decimal, _) = parts.as_struct_fields().field(OHLC_PARTS[0]).unwrap() else {
        panic!("{OHLC} parts must be decimals");
    };
    let (p, scale) = (decimal.precision(), decimal.scale());
    let at = |precision: u8| DecimalDType::new(precision, scale);
    let raw = |name: &str| get_item(name, get_item(OHLC, root()));
    let part = |name: &str, precision: u8| {
        vortex_cast(
            raw(name),
            DType::Decimal(at(precision), Nullability::Nullable),
        )
    };
    // p + 1 digits.
    let close = || checked_add(part("open", p), part("body", p));
    let zero = VortexScalar::decimal(DecimalValue::I128(0), at(p), Nullability::NonNullable);
    let up = || gt_eq(raw("body"), lit(zero.clone()));
    Some(match price {
        "open" => (part("open", p), p, scale),
        "close" => (close(), p + 1, scale),
        // p + 2 digits.
        "high" => {
            let top = zip_expr(up(), close(), part("open", p + 1));
            (checked_add(top, part("upper", p + 1)), p + 2, scale)
        }
        "low" => {
            let bottom = zip_expr(up(), part("open", p + 1), close());
            let low = binary(Operator::Sub, bottom, part("lower", p + 1));
            (low, p + 2, scale)
        }
        _ => return None,
    })
}
