//! Write a table of bars with `write_options` and read it back: the file reads back as the
//! source (the bars layout rebuilds the prices from their parts), and under the layout the parts
//! and integers are bit-packed in the encodings the edition allows.

use std::sync::Arc;

use arrow::array::cast::AsArray;
use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field};
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
use vortex_ohlc::bars::{data_child, ProductResidual, OHLC};
use vortex_ohlc::ohlc::Part;

use test_util::{source, write_with_residuals, ROWS, ROW_BLOCK};

/// Write `source` with `residuals`, and read it back: its chunks as stored, under the bars layout,
/// as Vortex arrays, and its rows through the layout, with `source`'s schema, as one Arrow batch.
fn write_and_read(
    source: &RecordBatch,
    residuals: Vec<ProductResidual>,
) -> (Vec<ArrayRef>, RecordBatch) {
    let schema = source.schema();
    let runtime = CurrentThreadRuntime::new();
    let session = VortexSession::default().with_handle(runtime.handle());
    vortex_ohlc::register(&session);
    let file = session
        .open_options()
        .open_buffer(write_with_residuals(source, residuals))
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

/// The names of a stored chunk's columns.
fn names(chunk: &ArrayRef) -> Vec<String> {
    let fields = chunk.dtype().as_struct_fields_opt().unwrap();
    fields.names().iter().map(|name| name.to_string()).collect()
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
    let (chunks, read) = write_and_read(&source, vec![ProductResidual::dollar_volume()]);
    assert_eq!(chunks.len(), ROWS.div_ceil(ROW_BLOCK));
    assert_eq!(read, source);
    assert!(names(&chunks[0]).contains(&"dollar_volume_residual".to_string()));
    assert!(!names(&chunks[0]).contains(&"dollar_volume".to_string()));

    // The parts and transactions are bit-packed, in the encodings the edition allows.
    let mut v2 = false;
    for chunk in &chunks {
        let ohlc = child(chunk, OHLC);
        for part in Part::ALL.map(Part::name) {
            let ids = encoding_ids(&child(&ohlc, part));
            // Under `vortex_ohlc.scaled` where a chunk's values share trailing zeros.
            let parts = match ids[0].as_str() {
                "vortex_ohlc.scaled" => &ids[1],
                _ => &ids[0],
            };
            assert!(parts.contains("decimal_byte_parts"), "{part}: {ids:?}");
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
    // Only the last chunk has volumes wider than an i64, even without their trailing zeros, cut
    // into an i64 and lower parts.
    let volume_children = |chunk: &ArrayRef| {
        let volume = child(chunk, "volume");
        assert_eq!(volume.encoding_id().to_string(), "vortex_ohlc.scaled");
        child(&volume, "values").named_children().len()
    };
    let last = chunks.last().unwrap();
    assert!(volume_children(last) > volume_children(&chunks[0]));
}

/// Written with only a residual whose columns the table hasn't all, the dollar volume is stored
/// as it is, and reads back the same.
#[test]
fn round_trip_without_residual() {
    let source = source();
    let residuals = vec![ProductResidual::new("dollar_volume", "volume", "vwap")];
    let (chunks, read) = write_and_read(&source, residuals);
    assert_eq!(read, source);
    assert!(names(&chunks[0]).contains(&"dollar_volume".to_string()));
    assert!(!names(&chunks[0]).contains(&"dollar_volume_residual".to_string()));
}

/// A residual against any two columns reads back the same: the dollar volume against the volume
/// and the close, a price the layout rebuilds from its parts.
#[test]
fn round_trip_against_other_factors() {
    let source = source();
    let residuals = vec![ProductResidual::new("dollar_volume", "volume", "close")];
    let (chunks, read) = write_and_read(&source, residuals);
    assert_eq!(read, source);
    assert!(names(&chunks[0]).contains(&"dollar_volume_residual".to_string()));
}

/// A residual whose columns the table has but can't store fails the write: a price can't be one.
#[test]
#[should_panic(expected = "open is a price")]
fn residual_of_a_price_fails() {
    write_with_residuals(
        &source(),
        vec![ProductResidual::new("open", "volume", "close")],
    );
}
