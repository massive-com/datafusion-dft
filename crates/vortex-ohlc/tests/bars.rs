//! A table of bars written under the bars layout reads back with the source's prices, through
//! Vortex's own scan: the whole row, one price, a price filter, and pruning on a price.

use std::sync::Arc;

use arrow::array::cast::AsArray;
use arrow::array::types::Decimal128Type;
use arrow::array::RecordBatch;
use arrow::datatypes::DataType;
use vortex::array::{ArrayRef, VortexSessionExecute};
use vortex::arrow::ArrowSessionExt;
use vortex::dtype::{DType, DecimalDType, Nullability};
use vortex::expr::{and, get_item, gt, lit, lt_eq, root, Expression};
use vortex::file::{OpenOptionsSessionExt, VortexFile};
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::mask::Mask;
use vortex::scalar::{DecimalValue, Scalar};
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_ohlc::ohlc::Price;

use test_util::{source, source_plan, ROWS, ROW_BLOCK};

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

/// test-util's source, planned as usual and written, so under the bars layout.
fn write(source: &RecordBatch) -> Vec<u8> {
    test_util::write(source, &source_plan(source))
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
    let high = source_prices(&source, Price::High);
    // The zones with a high above $360.
    let kept: Vec<bool> = high
        .chunks(ROW_BLOCK)
        .map(|zone| zone.iter().any(|&h| h > 360 * 10_000))
        .collect();
    assert!(kept.iter().any(|k| !k), "nothing to prune");
    let reader = file.layout_reader().unwrap();
    let expr = gt(get_item("high", root()), cents(36_000))
        .bind(file.dtype())
        .unwrap();
    let mask = reader
        .pruning_evaluation(&(0..ROWS as u64), &expr, Mask::new_true(ROWS))
        .unwrap();
    let mask = runtime.block_on(mask).unwrap();
    for (zone, kept) in kept.iter().enumerate() {
        let start = zone * ROW_BLOCK;
        let rows = mask.slice(start..(start + ROW_BLOCK).min(ROWS));
        assert_eq!(rows.true_count() > 0, *kept, "zone {zone}");
    }
}
