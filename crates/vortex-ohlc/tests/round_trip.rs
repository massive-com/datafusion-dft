//! Profile a table of bars, plan from it, write it through `write_strategy` and read it back:
//! the file holds exactly the planned batch, the prices put back together from the OHLC parts are
//! the source's, and the decimal and packed columns are stored as the strategy chose.

use std::sync::Arc;

use arrow::array::cast::AsArray;
use arrow::array::types::Decimal128Type;
use arrow::array::{
    ArrayRef as ArrowArrayRef, Decimal128Array, Int64Array, RecordBatch, StringArray,
};
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Schema};
use vortex::array::iter::ArrayIteratorAdapter;
use vortex::array::{ArrayRef, VortexSessionExecute};
use vortex::arrow::ArrowSessionExt;
use vortex::buffer::ByteBuffer;
use vortex::file::{OpenOptionsSessionExt, WriteOptionsSessionExt};
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_ohlc::ohlc::ohlc_from_parts;
use vortex_ohlc::plan::{
    apply_plan, plan_from_json, planned_schema, transforms_json, Plan, Transform, OHLC,
    OHLC_PRICES, TRANSFORMS_KEY,
};
use vortex_ohlc::profile::{plan, DecimalProfiler};
use vortex_ohlc::strategy::write_strategy;

const ROWS: usize = 5_000;
const ROW_BLOCK: usize = 2_048;
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

/// Bars for 10 tickers in a row each, prices at scale 4 in whole cents, volumes at scale 6 in
/// whole shares.
fn source() -> RecordBatch {
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
    let schema = Schema::new(vec![
        Field::new("ticker", DataType::Utf8, false),
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
            Arc::new(StringArray::from(tickers)),
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

/// `source`'s plan, from its profile, with `transactions` packed.
fn source_plan(source: &RecordBatch) -> Plan {
    let mut profiler = DecimalProfiler::new(&source.schema());
    profiler.add(source);
    let mut plan = plan(&profiler.finish());
    plan.insert("transactions".to_string(), Transform::Packed);
    plan
}

/// Write `source` with `plan` as `ROW_BLOCK`-row chunks, and check the file records the plan;
/// return the chunks read back, as Vortex arrays, and as one Arrow batch.
fn write_and_read(source: &RecordBatch, plan: &Plan) -> (Vec<ArrayRef>, RecordBatch) {
    let runtime = CurrentThreadRuntime::new();
    let session = VortexSession::default().with_handle(runtime.handle());
    vortex_ohlc::edition::enable(&session).unwrap();

    let schema = planned_schema(plan, &source.schema());
    let dtype = session.arrow().from_arrow_schema(&schema).unwrap();
    let transforms = serde_json::to_vec(&transforms_json(plan, &source.schema())).unwrap();
    let chunks: Vec<_> = (0..ROWS)
        .step_by(1_000)
        .map(|start| {
            let batch = source.slice(start, 1_000.min(ROWS - start));
            let batch = apply_plan(plan, &schema, batch);
            session.arrow().from_arrow_record_batch(batch, &schema)
        })
        .collect();
    let mut bytes = Vec::new();
    session
        .write_options()
        .with_strategy(write_strategy(&source.schema(), plan, ROW_BLOCK))
        .with_metadata_segment(TRANSFORMS_KEY, ByteBuffer::copy_from(transforms))
        .blocking(&runtime)
        .write(
            &mut bytes,
            ArrayIteratorAdapter::new(dtype, chunks.into_iter()),
        )
        .unwrap();

    // Metadata segments are only loaded when asked for.
    let file = session
        .open_options()
        .include_metadata()
        .open_buffer(bytes)
        .unwrap();
    let recorded = file.metadata_segment(TRANSFORMS_KEY).unwrap();
    let recorded = serde_json::from_slice(recorded.as_ref()).unwrap();
    assert_eq!(plan_from_json(&recorded).as_ref(), Some(plan));
    let chunks: Vec<ArrayRef> = file
        .scan()
        .unwrap()
        .into_array_iter(&runtime)
        .unwrap()
        .map(|chunk| chunk.unwrap())
        .collect();
    let mut ctx = session.create_execution_ctx();
    let target = Field::new("", DataType::Struct(schema.fields().clone()), false);
    let batches: Vec<RecordBatch> = chunks
        .iter()
        .map(|chunk| {
            let array = session
                .arrow()
                .execute_arrow(chunk.clone(), Some(&target), &mut ctx)
                .unwrap();
            RecordBatch::from(array.as_struct().clone())
        })
        .collect();
    (chunks, concat_batches(&schema, &batches).unwrap())
}

fn child(array: &ArrayRef, name: &str) -> ArrayRef {
    array
        .named_children()
        .into_iter()
        .find(|(child, _)| child == name)
        .map(|(_, child)| child)
        .unwrap_or_else(|| panic!("{} has no child {name}", array.encoding_id()))
}

/// The encoding IDs of `array` and everything under it.
fn encoding_ids(array: &ArrayRef) -> Vec<String> {
    let mut ids = vec![array.encoding_id().to_string()];
    for (_, child) in array.named_children() {
        ids.extend(encoding_ids(&child));
    }
    ids
}

#[test]
fn round_trip() {
    let source = source();
    let plan = source_plan(&source);
    assert!(OHLC_PRICES
        .iter()
        .all(|p| plan.get(*p) == Some(&Transform::Ohlc(2))));
    assert_eq!(plan.get("volume"), Some(&Transform::Rescale(6)));

    let (chunks, read) = write_and_read(&source, &plan);
    assert_eq!(chunks.len(), ROWS.div_ceil(ROW_BLOCK));
    let schema = planned_schema(&plan, &source.schema());
    let planned = apply_plan(&plan, &schema, source.clone());
    assert_eq!(read.columns(), planned.columns());

    // The prices put back together from the parts, at the source's scale.
    let ohlc = read.column_by_name(OHLC).unwrap().as_struct();
    let parts: Vec<&[i128]> = (0..4)
        .map(|i| {
            ohlc.column(i)
                .as_primitive::<Decimal128Type>()
                .values()
                .as_ref()
        })
        .collect();
    let prices: Vec<&[i128]> = OHLC_PRICES
        .iter()
        .map(|p| {
            let column = source.column_by_name(p).unwrap();
            column.as_primitive::<Decimal128Type>().values().as_ref()
        })
        .collect();
    for row in 0..ROWS {
        let rebuilt = ohlc_from_parts([0, 1, 2, 3].map(|i| parts[i][row])).map(|v| v * 100);
        assert_eq!(rebuilt, [0, 1, 2, 3].map(|i| prices[i][row]), "row {row}");
    }

    for chunk in &chunks {
        let ohlc = child(chunk, OHLC);
        for part in ["open", "body", "upper", "lower"] {
            let ids = encoding_ids(&child(&ohlc, part));
            assert!(ids[0].contains("decimal_byte_parts"), "{part}: {ids:?}");
            assert!(
                ids.iter().any(|id| id.starts_with("fastlanes.for")),
                "{part}: {ids:?}"
            );
        }
        let ids = encoding_ids(&child(chunk, "transactions"));
        assert!(ids[0].starts_with("fastlanes.for"), "transactions: {ids:?}");
    }
    // Only the last chunk has volumes wider than an i64, cut into an i64 and lower parts.
    let volume_children = |chunk: &ArrayRef| child(chunk, "volume").named_children().len();
    let last = chunks.last().unwrap();
    assert!(volume_children(last) > volume_children(&chunks[0]));
}
