//! A table of bars written under the bars layout reads back with the source's prices, through
//! Vortex's own scan: the whole row, one price, a price filter, pruning on prices and the dollar
//! volume, and filters on a timestamp.

use std::sync::Arc;

use arrow::array::cast::AsArray;
use arrow::array::types::{Decimal128Type, Int64Type, TimestampNanosecondType};
use arrow::array::{Array, RecordBatch, TimestampNanosecondArray};
use arrow::datatypes::{DataType, Field, Schema};
use vortex::array::extension::datetime::{TimeUnit, Timestamp, TimestampOptions};
use vortex::array::{ArrayRef, VortexSessionExecute};
use vortex::arrow::ArrowSessionExt;
use vortex::dtype::{DType, DecimalDType, Nullability};
use vortex::expr::{and, eq, get_item, gt, gt_eq, lit, lt, lt_eq, or, root, Expression};
use vortex::file::{OpenOptionsSessionExt, VortexFile};
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::mask::Mask;
use vortex::scalar::{DecimalValue, Scalar};
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_ohlc::ohlc::Price;

use test_util::{source, write, ROWS, ROW_BLOCK};

/// test-util's prices: `Decimal(38, 4)`.
fn price_dtype() -> DecimalDType {
    DecimalDType::new(38, 4)
}

fn session(runtime: &CurrentThreadRuntime) -> VortexSession {
    let session = VortexSession::default().with_handle(runtime.handle());
    vortex_ohlc::edition::enable(&session).unwrap();
    vortex_ohlc::register(&session);
    session
}

/// `value` cents as a literal of the prices' type.
fn cents(value: i128) -> Expression {
    let unscaled = DecimalValue::I128(value * 100);
    lit(Scalar::decimal(
        unscaled,
        price_dtype(),
        Nullability::NonNullable,
    ))
}

/// The unscaled values of every chunk `expr` projects, `filter` keeps.
fn scan(
    runtime: &CurrentThreadRuntime,
    file: &VortexFile,
    expr: Expression,
    filter: Option<Expression>,
) -> Vec<ArrayRef> {
    let mut scan = file
        .scan()
        .unwrap()
        .with_projection(expr.bind(file.dtype()).unwrap());
    if let Some(filter) = filter {
        scan = scan.with_filter(filter.bind(file.dtype()).unwrap());
    }
    scan.into_array_iter(runtime)
        .unwrap()
        .map(|chunk| chunk.unwrap())
        .collect()
}

fn decimals(session: &VortexSession, chunks: Vec<ArrayRef>) -> Vec<i128> {
    let mut ctx = session.create_execution_ctx();
    chunks
        .into_iter()
        .flat_map(|chunk| {
            let array = session
                .arrow()
                .execute_arrow(chunk, None, &mut ctx)
                .unwrap();
            assert_eq!(array.data_type(), &DataType::Decimal128(38, 4));
            array.as_primitive::<Decimal128Type>().values().to_vec()
        })
        .collect()
}

fn source_prices(source: &RecordBatch, price: Price) -> Vec<i128> {
    let column = source.column_by_name(price.name()).unwrap();
    column.as_primitive::<Decimal128Type>().values().to_vec()
}

#[test]
fn the_file_has_the_source_prices() {
    let runtime = CurrentThreadRuntime::new();
    let session = session(&runtime);
    let source = source();
    let file = session.open_options().open_buffer(write(&source)).unwrap();
    let fields = file.dtype().as_struct_fields_opt().unwrap();
    for price in Price::ALL {
        assert_eq!(
            fields.field(price.name()),
            Some(DType::Decimal(price_dtype(), Nullability::NonNullable)),
            "{price:?}"
        );
    }
    // One price, then the whole row's.
    for price in Price::ALL {
        let chunks = scan(&runtime, &file, get_item(price.name(), root()), None);
        assert_eq!(decimals(&session, chunks), source_prices(&source, price));
    }
    let mut ctx = session.create_execution_ctx();
    let rows = scan(&runtime, &file, root(), None);
    let mut high: Vec<i128> = Vec::new();
    for chunk in rows {
        let array = session
            .arrow()
            .execute_arrow(chunk, None, &mut ctx)
            .unwrap();
        let column = Arc::clone(array.as_struct().column_by_name("high").unwrap());
        high.extend(column.as_primitive::<Decimal128Type>().values().iter());
    }
    assert_eq!(high, source_prices(&source, Price::High));
}

#[test]
fn price_filters_keep_the_source_rows() {
    let runtime = CurrentThreadRuntime::new();
    let session = session(&runtime);
    let source = source();
    let file = session.open_options().open_buffer(write(&source)).unwrap();
    let high = source_prices(&source, Price::High);
    let low = source_prices(&source, Price::Low);
    let expected: Vec<i128> = high
        .iter()
        .zip(&low)
        .filter(|(&h, &l)| h > 37_040 * 100 && l <= 37_010 * 100)
        .map(|(&h, _)| h)
        .collect();
    assert!(!expected.is_empty());
    let filter = and(
        gt(get_item("high", root()), cents(37_040)),
        lt_eq(get_item("low", root()), cents(37_010)),
    );
    let chunks = scan(&runtime, &file, get_item("high", root()), Some(filter));
    assert_eq!(decimals(&session, chunks), expected);
}

#[test]
fn the_root_prunes_on_a_price() {
    let runtime = CurrentThreadRuntime::new();
    let session = session(&runtime);
    let source = source();
    let file = session.open_options().open_buffer(write(&source)).unwrap();
    let [high, low] = [Price::High, Price::Low].map(|price| source_prices(&source, price));
    // Per zone, whether a high is above $360, and whether a low is below $20.
    let zones = |prices: &[i128], hit: fn(i128) -> bool| -> Vec<bool> {
        prices
            .chunks(ROW_BLOCK)
            .map(|zone| zone.iter().any(|&p| hit(p)))
            .collect()
    };
    let high_zones = zones(&high, |h| h > 360 * 10_000);
    let low_zones = zones(&low, |l| l < 20 * 10_000);
    assert!(high_zones.iter().any(|k| !k), "nothing to prune");
    let either: Vec<bool> = (high_zones.iter().zip(&low_zones))
        .map(|(h, l)| h | l)
        .collect();
    let both: Vec<bool> = (high_zones.iter().zip(&low_zones))
        .map(|(h, l)| h & l)
        .collect();
    // Only the last zone has volumes wider than an i64, and dollar volumes above 10^13.
    let dollar_volume = source.column_by_name("dollar_volume").unwrap();
    let dollar_volume = dollar_volume
        .as_primitive::<Decimal128Type>()
        .values()
        .to_vec();
    let big = 10i128.pow(25);
    let big_zones = zones(&dollar_volume, |dv| dv > 10i128.pow(25));
    assert!(big_zones.iter().any(|k| !k), "nothing to prune");
    let big_dv = lit(Scalar::decimal(
        DecimalValue::I128(big),
        DecimalDType::new(38, 12),
        Nullability::NonNullable,
    ));
    let high_gt = || gt(get_item("high", root()), cents(36_000));
    let low_lt = || lt(get_item("low", root()), cents(2_000));
    // A filter on prices alone, or on prices and other columns: the prices' part prunes.
    let some_transactions = gt(get_item("transactions", root()), lit(10i64));
    let cases = [
        (high_gt(), &high_zones),
        (or(high_gt(), low_lt()), &either),
        (and(high_gt(), low_lt()), &both),
        (and(high_gt(), some_transactions), &high_zones),
        (gt(get_item("dollar_volume", root()), big_dv), &big_zones),
    ];
    let reader = file.layout_reader().unwrap();
    for (expr, kept) in cases {
        let name = expr.to_string();
        let mask = reader
            .pruning_evaluation(
                &(0..ROWS as u64),
                &expr.bind(file.dtype()).unwrap(),
                Mask::new_true(ROWS),
            )
            .unwrap();
        let mask = runtime.block_on(mask).unwrap();
        for (zone, kept) in kept.iter().enumerate() {
            let start = zone * ROW_BLOCK;
            let rows = mask.slice(start..(start + ROW_BLOCK).min(ROWS));
            assert_eq!(rows.true_count() > 0, *kept, "{name}: zone {zone}");
        }
    }
}

/// Seconds since the epoch at 09:30 New York (13:30 UTC) on the first day of
/// [`source_with_times`].
const FIRST_OPEN: i64 = 1_155_600_000 / 86_400 * 86_400 + 13 * 3600 + 30 * 60;

/// test-util's source with a `window_start` of minute bars: each ticker's 500 bars are two days
/// of 250 minutes from 09:30.
fn source_with_times() -> RecordBatch {
    let source = source();
    let starts: Vec<i64> = (0..ROWS as i64)
        .map(|row| {
            let bar = row % (ROWS as i64 / 10);
            (FIRST_OPEN + bar / 250 * 86_400 + bar % 250 * 60) * 1_000_000_000
        })
        .collect();
    let starts = TimestampNanosecondArray::from(starts).with_timezone("America/New_York");
    let mut fields: Vec<Field> = (source.schema().fields().iter())
        .map(|field| field.as_ref().clone())
        .collect();
    fields.push(Field::new(
        "window_start",
        starts.data_type().clone(),
        false,
    ));
    let mut columns = source.columns().to_vec();
    columns.push(Arc::new(starts));
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

/// `seconds` since the epoch as a literal of `window_start`'s type.
fn at(seconds: i64) -> Expression {
    let options = TimestampOptions {
        unit: TimeUnit::Nanoseconds,
        tz: Some("America/New_York".into()),
    };
    let nanos = Scalar::primitive(seconds * 1_000_000_000, Nullability::NonNullable);
    lit(Scalar::extension::<Timestamp>(options, nanos))
}

/// A timestamp's `=`, which the bars layout passes on as a `between`, and a window across two
/// days keep the source's rows; an `=` before every bar prunes every zone.
#[test]
fn time_filters_keep_the_source_rows() {
    let runtime = CurrentThreadRuntime::new();
    let session = session(&runtime);
    let source = source_with_times();
    let file = session.open_options().open_buffer(write(&source)).unwrap();
    let starts = source.column_by_name("window_start").unwrap();
    let starts = starts.as_primitive::<TimestampNanosecondType>().values();
    let transactions = source.column_by_name("transactions").unwrap();
    let transactions = transactions.as_primitive::<Int64Type>().values();
    let ws = || get_item("window_start", root());
    let day = 86_400;
    // 10:00 on the second day, and 13:30 on the first to 09:35 on the second.
    let ten = FIRST_OPEN + day + 30 * 60;
    let (from, to) = (FIRST_OPEN + 4 * 3600, FIRST_OPEN + day + 5 * 60);
    // Each filter, and the seconds it keeps.
    let cases = [
        (eq(ws(), at(ten)), ten..ten + 1),
        (and(gt_eq(ws(), at(from)), lt(ws(), at(to))), from..to),
    ];
    let mut ctx = session.create_execution_ctx();
    for (filter, kept) in cases {
        let name = filter.to_string();
        let expected: Vec<i64> = (starts.iter().zip(transactions.iter()))
            .filter(|(&t, _)| kept.contains(&(t / 1_000_000_000)))
            .map(|(_, &n)| n)
            .collect();
        assert!(!expected.is_empty(), "{name}");
        let chunks = scan(
            &runtime,
            &file,
            get_item("transactions", root()),
            Some(filter),
        );
        let actual: Vec<i64> = (chunks.into_iter())
            .flat_map(|chunk| {
                let array = session
                    .arrow()
                    .execute_arrow(chunk, None, &mut ctx)
                    .unwrap();
                array.as_primitive::<Int64Type>().values().to_vec()
            })
            .collect();
        assert_eq!(actual, expected, "{name}");
    }
    let before = eq(ws(), at(FIRST_OPEN - day)).bind(file.dtype()).unwrap();
    let reader = file.layout_reader().unwrap();
    let mask = reader
        .pruning_evaluation(&(0..ROWS as u64), &before, Mask::new_true(ROWS))
        .unwrap();
    assert_eq!(runtime.block_on(mask).unwrap().true_count(), 0);
}
