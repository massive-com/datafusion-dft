//! The table of bars vortex-ohlc's tests and datafusion-vortex-ohlc's write and read back: its
//! source batch, its plan and the file written with it.

use std::sync::Arc;

use arrow::array::{
    ArrayRef as ArrowArrayRef, Decimal128Array, Int64Array, RecordBatch, StringViewArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_ohlc::plan::Plan;
use vortex_ohlc::profile::{plan, DecimalProfiler};
use vortex_ohlc::write::{stored_arrays, write_options};

pub const ROWS: usize = 5_000;
pub const ROW_BLOCK: usize = 2_048;
/// Rows from here on (all in the last chunk) have a volume too wide for an i64.
const WIDE_FROM: usize = 4_500;

/// A small deterministic generator, so the test needs no dependency.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, below: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) % below
    }
}

fn decimal(values: Vec<i128>, scale: i8) -> ArrowArrayRef {
    Arc::new(
        Decimal128Array::from(values)
            .with_precision_and_scale(38, scale)
            .unwrap(),
    )
}

/// Bars for 10 tickers in a row each, ticker `t` trading around `10 + 40t` dollars: prices at
/// scale 4 in whole cents, volumes at scale 6 in whole shares.
pub fn source() -> RecordBatch {
    let mut rng = Lcg(7);
    let mut columns: [Vec<i128>; 5] = Default::default();
    let (mut tickers, mut transactions) = (Vec::new(), Vec::new());
    for row in 0..ROWS {
        let ticker = row * 10 / ROWS;
        tickers.push(format!("T{ticker}"));
        let base = (10 + 40 * ticker as i128) * 100;
        let open = base + rng.next(50) as i128;
        let close = base + rng.next(50) as i128;
        let high = open.max(close) + [0, 0, 0, 3][rng.next(4) as usize];
        let low = open.min(close) - [0, 0, 2][rng.next(3) as usize];
        let volume = match row >= WIDE_FROM {
            // Wider than an i64 even once rescaled to whole shares.
            true => i64::MAX as i128 + 1 + rng.next(1_000) as i128,
            false => rng.next(100_000) as i128,
        };
        for (column, cents) in columns.iter_mut().zip([open, high, low, close]) {
            column.push(cents * 100);
        }
        columns[4].push(volume * 1_000_000);
        transactions.push(rng.next(500) as i64);
    }
    let [open, high, low, close, volume] = columns;
    // Utf8View, as Vortex reads strings back.
    let schema = Schema::new(vec![
        Field::new("ticker", DataType::Utf8View, false),
        Field::new("open", DataType::Decimal128(38, 4), false),
        Field::new("high", DataType::Decimal128(38, 4), false),
        Field::new("low", DataType::Decimal128(38, 4), false),
        Field::new("close", DataType::Decimal128(38, 4), false),
        Field::new("volume", DataType::Decimal128(38, 6), false),
        Field::new("transactions", DataType::Int64, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringViewArray::from(tickers)),
            decimal(open, 4),
            decimal(high, 4),
            decimal(low, 4),
            decimal(close, 4),
            decimal(volume, 6),
            Arc::new(Int64Array::from(transactions)),
        ],
    )
    .unwrap()
}

/// `source`'s plan, from its profile.
pub fn source_plan(source: &RecordBatch) -> Plan {
    let mut profiler = DecimalProfiler::new(&source.schema());
    profiler.add(source);
    plan(&profiler.finish())
}

/// `source` written with `plan`, in 1,000-row batches and `ROW_BLOCK`-row chunks.
pub fn write(source: &RecordBatch, plan: &Plan) -> Vec<u8> {
    let runtime = CurrentThreadRuntime::new();
    let session = VortexSession::default().with_handle(runtime.handle());
    vortex_ohlc::edition::enable(&session).unwrap();
    let batches: Vec<RecordBatch> = (0..ROWS)
        .step_by(1_000)
        .map(|start| source.slice(start, 1_000.min(ROWS - start)))
        .collect();
    let arrays = stored_arrays(&session, &source.schema(), plan, batches).unwrap();
    let mut bytes = Vec::new();
    write_options(&session, ROW_BLOCK)
        .blocking(&runtime)
        .write(&mut bytes, arrays)
        .unwrap();
    bytes
}
