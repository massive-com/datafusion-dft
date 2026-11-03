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
use vortex_ohlc::ohlc::{Part, Price};
use vortex_ohlc::plan::{apply_plan, planned_schema, Plan, Transform, OHLC};

use test_util::{source, source_plan, write, ROWS, ROW_BLOCK};

/// Write `source` with `plan`; return the chunks read back, as Vortex arrays, and as one Arrow
/// batch.
fn write_and_read(source: &RecordBatch, plan: &Plan) -> (Vec<ArrayRef>, RecordBatch) {
    let runtime = CurrentThreadRuntime::new();
    let session = VortexSession::default().with_handle(runtime.handle());
    let file = session
        .open_options()
        .open_buffer(write(source, plan))
        .unwrap();
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
    assert!(Price::ALL
        .iter()
        .all(|p| plan.get(p.name()) == Some(&Transform::Ohlc(2))));
    assert_eq!(plan.get("volume"), Some(&Transform::Rescale(6)));

    let (chunks, read) = write_and_read(&source, &plan);
    assert_eq!(chunks.len(), ROWS.div_ceil(ROW_BLOCK));
    let schema = planned_schema(&plan, &source.schema());
    let planned = apply_plan(&plan, &schema, source.clone());
    assert_eq!(read.columns(), planned.columns());

    // The prices rebuilt from the parts, at the source's scale.
    let ohlc = read.column_by_name(OHLC).unwrap().as_struct();
    let parts = Part::ALL.map(|part| {
        let column = ohlc.column_by_name(part.name()).unwrap();
        column.as_primitive::<Decimal128Type>().values().as_ref()
    });
    for price in Price::ALL {
        let column = source.column_by_name(price.name()).unwrap();
        let source = column.as_primitive::<Decimal128Type>().values().as_ref();
        assert_eq!(price.rebuild(parts, 100), source, "{price:?}");
    }

    // The parts and transactions are bit-packed, in the encodings the edition allows.
    let mut v2 = false;
    for chunk in &chunks {
        let ohlc = child(chunk, OHLC);
        for part in Part::ALL.map(Part::name) {
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
