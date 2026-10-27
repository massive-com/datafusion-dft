//! Reading Parquet: strings as `Utf8View`, like Vortex's scan returns them, whole files and
//! needles.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use parquet::arrow::ProjectionMask;
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};

use crate::common::needle::{parquet_needle, Needle};

/// Rows per batch in a Parquet scan: one row group.
pub const SCAN_BATCH_SIZE: usize = 65_536;

/// Reader metadata that decodes every string column as `Utf8View` (with `views`), like Vortex's
/// scan does, instead of the reader's default `Utf8`.
fn view_metadata(metadata: Arc<ParquetMetaData>, views: bool) -> ArrowReaderMetadata {
    let natural =
        ArrowReaderMetadata::try_new(Arc::clone(&metadata), ArrowReaderOptions::new()).unwrap();
    if !views {
        return natural;
    }
    let fields: Vec<Field> = natural
        .schema()
        .fields()
        .iter()
        .map(|field| match field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 => {
                field.as_ref().clone().with_data_type(DataType::Utf8View)
            }
            _ => field.as_ref().clone(),
        })
        .collect();
    let schema = Schema::new_with_metadata(fields, natural.schema().metadata().clone());
    let options = ArrowReaderOptions::new().with_schema(Arc::new(schema));
    ArrowReaderMetadata::try_new(metadata, options).unwrap()
}

/// The footer of `path`, parsed and set up to read strings as `Utf8View`. Parsing it once and
/// reusing it is the Parquet side of a lookup against a file that is already open.
pub fn scan_metadata(path: &Path) -> ArrowReaderMetadata {
    let file = File::open(path).unwrap();
    let metadata = ParquetMetaDataReader::new()
        .parse_and_finish(&file)
        .unwrap();
    view_metadata(Arc::new(metadata), true)
}

/// Open a Parquet file for reading, strings as `Utf8View` with `views`, or exactly as stored.
pub fn open_parquet(path: &Path, views: bool) -> ParquetRecordBatchReaderBuilder<File> {
    let file = File::open(path).unwrap();
    let metadata = ParquetMetaDataReader::new()
        .parse_and_finish(&file)
        .unwrap();
    ParquetRecordBatchReaderBuilder::new_with_metadata(
        file,
        view_metadata(Arc::new(metadata), views),
    )
}

/// Read `path`, optionally just `columns`, strings as `Utf8View`.
pub fn parquet_batches(
    path: &Path,
    columns: Option<&[&str]>,
) -> impl Iterator<Item = RecordBatch> + use<> {
    let mut builder = open_parquet(path, true).with_batch_size(SCAN_BATCH_SIZE);
    if let Some(columns) = columns {
        let mask = ProjectionMask::columns(builder.parquet_schema(), columns.iter().copied());
        builder = builder.with_projection(mask);
    }
    builder.build().unwrap().map(Result::unwrap)
}

/// Every column of the rows of `path` that match `needle`, with the footer already parsed (see
/// [`scan_metadata`]); only the file handle is opened.
pub fn parquet_needle_batches(
    path: &Path,
    metadata: &ArrowReaderMetadata,
    needle: &Needle,
) -> impl Iterator<Item = RecordBatch> + use<> {
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(
        File::open(path).unwrap(),
        metadata.clone(),
    )
    .with_batch_size(SCAN_BATCH_SIZE);
    parquet_needle(builder, needle)
        .build()
        .unwrap()
        .map(Result::unwrap)
}
