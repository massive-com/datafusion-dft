//! Profile a table of bars, plan from it, write it with `write_options` and read it back: the
//! file reads back as the source, with the source's prices (the bars layout rebuilds them from
//! their parts) and its other columns as planned, and under the layout the parts and integers are
//! bit-packed in the encodings the edition allows.

use std::sync::Arc;

use arrow::array::cast::AsArray;
use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Schema};
use vortex::array::{ArrayRef, VortexSessionExecute};
use vortex::arrow::ArrowSessionExt;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::layout::LayoutReader;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_ohlc::bars::data_child;
use vortex_ohlc::ohlc::{Part, Price};
use vortex_ohlc::plan::{apply_plan, planned_schema, Plan, Transform, OHLC};

use test_util::{source, source_plan, write, ROWS, ROW_BLOCK};

/// Write `source` with `plan`, and read it back: its chunks as stored, under the bars layout, as
/// Vortex arrays, and its rows through the layout, with `schema`, as one Arrow batch.
fn write_and_read(
    source: &RecordBatch,
    plan: &Plan,
    schema: &Schema,
) -> (Vec<ArrayRef>, RecordBatch) {
    let runtime = CurrentThreadRuntime::new();
    let session = VortexSession::default().with_handle(runtime.handle());
    vortex_ohlc::register(&session);
    let file = session
        .open_options()
        .open_buffer(write(source, plan))
        .unwrap();
    let scan = |reader: Arc<dyn LayoutReader>| -> Vec<ArrayRef> {
        ScanBuilder::new(session.clone(), reader)
            .into_array_iter(&runtime)
            .unwrap()
            .map(|chunk| chunk.unwrap())
            .collect()
    };
    let stored = data_child(file.footer().layout()).unwrap();
    let stored = stored
        .new_reader(
            "".into(),
            file.segment_source(),
            &session,
            &Default::default(),
        )
        .unwrap();
    let chunks = scan(stored);
    let mut ctx = session.create_execution_ctx();
    let target = Field::new("", DataType::Struct(schema.fields().clone()), false);
    let batches: Vec<RecordBatch> = scan(file.layout_reader().unwrap())
        .into_iter()
        .map(|rows| {
            let array = session
                .arrow()
                .execute_arrow(rows, Some(&target), &mut ctx)
                .unwrap();
            RecordBatch::from(array.as_struct().clone())
        })
        .collect();
    (
        chunks,
        concat_batches(&Arc::new(schema.clone()), &batches).unwrap(),
    )
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

    // What the file reads back as: the source, rescaled but for the prices.
    let rescale: Plan = (plan.iter())
        .filter(|(_, transform)| matches!(transform, Transform::Rescale(_)))
        .map(|(name, transform)| (name.clone(), *transform))
        .collect();
    let schema = planned_schema(&rescale, &source.schema());
    let (chunks, read) = write_and_read(&source, &plan, &schema);
    assert_eq!(chunks.len(), ROWS.div_ceil(ROW_BLOCK));
    let expected = apply_plan(&rescale, &schema, source.clone());
    assert_eq!(read.columns(), expected.columns());

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
