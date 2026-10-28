//! Needle-in-a-haystack lookups: a handful of selective filters, each returning every column of
//! the rows it matches, and the rows of the Vortex file that match one (see [`matching_rows`]).

use std::path::Path;

use vortex::array::arrays::PrimitiveArray;
use vortex::array::VortexSessionExecute;
use vortex::dtype::DType;
use vortex::expr::{and_collect, binary, get_item, lit, root, BoundExpression, Expression};
use vortex::file::{OpenOptionsSessionExt, VortexFile};
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::layouts::row_idx::row_idx;
use vortex::scalar::{DecimalValue, Scalar as VortexScalar};
use vortex::scalar_fn::fns::operators::Operator;
use vortex_ohlc::plan::{Plan, Transform};
use vortex_ohlc::prices::compare_price;
use vortex_ohlc::write::file_plan;

use crate::common::data::new_session;

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
    fn operator(self) -> Operator {
        match self {
            Op::Eq => Operator::Eq,
            Op::Gt => Operator::Gt,
            Op::Ge => Operator::Gte,
            Op::Lt => Operator::Lt,
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
///   With the prices stored against each other ([`OHLC`](vortex_ohlc::plan::OHLC)), each price filters through a
///   different expression over the parts (see [`vortex_ohlc::prices`]).
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
// Vortex
// ---------------------------------------------------------------------------------------------

/// The rows of the Vortex file at `path` that match `needle`, ascending, and the file's row
/// count: a filtered scan with only the row index projected. `None` if a term doesn't go into a
/// Vortex filter (see [`vortex_filter`]).
pub fn matching_rows(path: &Path, needle: &Needle) -> Option<(Vec<u64>, u64)> {
    let runtime = CurrentThreadRuntime::new();
    let session = new_session(&runtime);
    let open = session.open_options().include_metadata().open_path(path);
    let file = runtime.block_on(open).unwrap();
    let plan = file_plan(&file).expect("the file records a plan of this version");
    let filter = vortex_filter(&file, needle, &plan)?;
    let mut ctx = session.create_execution_ctx();
    let hits = file
        .scan()
        .unwrap()
        .with_filter(filter)
        .with_projection(row_idx().bind(file.dtype()).unwrap())
        .into_array_iter(&runtime)
        .unwrap();
    let mut rows: Vec<u64> = Vec::new();
    for chunk in hits {
        let chunk = chunk.unwrap().execute::<PrimitiveArray>(&mut ctx).unwrap();
        rows.extend_from_slice(chunk.as_slice::<u64>());
    }
    Some((rows, file.row_count()))
}

/// `needle` as one Vortex filter expression over `file`, its literals in each column's stored
/// type: a rescaled decimal column gets a literal at its rescaled scale, and a price stored in
/// the [`OHLC`](vortex_ohlc::plan::OHLC) struct is compared over its parts (see [`ohlc_term`]).
/// `None` if a term's literal doesn't fit its column's stored type exactly.
fn vortex_filter(file: &VortexFile, needle: &Needle, plan: &Plan) -> Option<BoundExpression> {
    let fields = file.dtype().as_struct_fields();
    let conjuncts = needle
        .terms
        .iter()
        .map(|term| match plan.get(term.column) {
            Some(Transform::Ohlc(_)) => ohlc_term(file.dtype(), term),
            _ => {
                let dtype = fields.field(term.column).unwrap();
                vortex_literal(term.value, &dtype).map(|literal| {
                    binary(term.op.operator(), get_item(term.column, root()), literal)
                })
            }
        })
        .collect::<Option<Vec<_>>>()?;
    let filter = and_collect(conjuncts)?.bind(file.dtype()).unwrap();
    Some(filter.optimize_recursive().unwrap())
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

/// A term on a price stored in the [`OHLC`](vortex_ohlc::plan::OHLC) struct (in a table of `dtype`), over its parts (see
/// [`compare_price`]); `None` if the literal doesn't fit the type it's compared in.
fn ohlc_term(dtype: &DType, term: &Term) -> Option<Expression> {
    let Value::Decimal { unscaled, scale } = term.value else {
        panic!("{}: a price compares with a decimal", term.column);
    };
    compare_price(dtype, term.column, term.op.operator(), unscaled, scale)
}
