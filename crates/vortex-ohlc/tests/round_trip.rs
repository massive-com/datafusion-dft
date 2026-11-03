//! Profile a table of bars, plan from it, write it with `write_options` and read it back:
//! the file holds exactly the planned batch, the prices put back together from the OHLC parts are
//! the source's, and the parts and integers are bit-packed in the encodings the edition allows.

use arrow::array::cast::AsArray;
use arrow::array::types::Decimal128Type;
use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field};
use vortex::array::{ArrayRef, VortexSessionExecute};
use vortex::arrow::ArrowSessionExt;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_ohlc::ohlc::ohlc_from_parts;
use vortex_ohlc::plan::{
    apply_plan, planned_schema, Plan, Transform, OHLC, OHLC_PARTS, OHLC_PRICES,
};
use vortex_ohlc::write::{file_plan, file_source_schema};

use test_util::{source, source_plan, write, ROWS, ROW_BLOCK};

/// Write `source` with `plan`, and check the file records the plan, and the source's schema
/// from it, given the file's schema or the source's own; return the chunks read back, as Vortex
/// arrays, and as one Arrow batch.
fn write_and_read(source: &RecordBatch, plan: &Plan) -> (Vec<ArrayRef>, RecordBatch) {
    let runtime = CurrentThreadRuntime::new();
    let session = VortexSession::default().with_handle(runtime.handle());
    // Metadata segments are only loaded when asked for.
    let file = session
        .open_options()
        .include_metadata()
        .open_buffer(write(source, plan))
        .unwrap();
    assert_eq!(file_plan(&file).as_ref(), Some(plan));
    let stored = session.arrow().to_arrow_schema(file.dtype()).unwrap();
    for table in [&stored, source.schema().as_ref()] {
        assert_eq!(
            file_source_schema(&file, table).as_ref(),
            Some(source.schema().as_ref())
        );
    }
    let schema = planned_schema(plan, &source.schema());
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

    // The parts and transactions are bit-packed, in the encodings the edition allows.
    let mut v2 = false;
    for chunk in &chunks {
        let ohlc = child(chunk, OHLC);
        for part in OHLC_PARTS {
            let ids = encoding_ids(&child(&ohlc, part));
            assert!(ids[0].contains("decimal_byte_parts"), "{part}: {ids:?}");
            assert!(
                ids.iter().any(|id| id.starts_with("fastlanes.")),
                "{part}: {ids:?}"
            );
            v2 |= ids.iter().any(|id| id.ends_with(".v2"));
        }
        let ids = encoding_ids(&child(chunk, "transactions"));
        assert!(
            ids.iter().any(|id| id.starts_with("fastlanes.")),
            "transactions: {ids:?}"
        );
    }
    assert!(v2, "no part is in an encoding only the edition allows");
    // Only the last chunk has volumes wider than an i64, cut into an i64 and lower parts.
    let volume_children = |chunk: &ArrayRef| child(chunk, "volume").named_children().len();
    let last = chunks.last().unwrap();
    assert!(volume_children(last) > volume_children(&chunks[0]));
}
