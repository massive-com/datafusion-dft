//! Write a month of bars as the benches do (`benches/common/data.rs`), also letting the compressor
//! emit the Vortex array encodings named after the paths, which no enabled edition lists, to see
//! what they would change:
//!
//! ```text
//! cargo run --release -p datafusion-vortex-ohlc --example write_bars -- \
//!     <source.parquet> <dest.vortex> [<array id>...]
//! ```
//!
//! For example `fastlanes.delta` lets the compressor choose FastLanes Delta.

use std::fs::File;
use std::io::BufWriter;

use arrow::array::RecordBatchReader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::editions::{Edition, EditionFamily, EditionId, EditionInclusion, EditionSessionExt};
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_ohlc::bars::ProductResidual;
use vortex_ohlc::write::{arrays, write_options, DEFAULT_ROW_BLOCK};

/// A family of its own, since enabling an edition replaces the enabled one of the same family.
static FAMILY: EditionFamily = EditionFamily {
    name: "write_bars",
    origin: "datafusion-vortex-ohlc examples",
    doc: "The array encodings `write_bars` was asked to allow.",
};

const EXTRA: EditionId = EditionId::new("write_bars", 2026, 11, 0);

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [source, dest, extra @ ..] = &args[..] else {
        panic!("usage: write_bars <source.parquet> <dest.vortex> [<array id>...]");
    };

    let runtime = CurrentThreadRuntime::new();
    let session = VortexSession::default().with_handle(runtime.handle());
    vortex_ohlc::edition::enable(&session).unwrap();
    vortex_ohlc::register(&session);
    if !extra.is_empty() {
        let editions = session.editions();
        editions.declare_family(&FAMILY).unwrap();
        editions
            .declare_edition(Edition {
                id: EXTRA,
                immutable: false,
                min_library_version: None,
            })
            .unwrap();
        for id in extra {
            editions
                .declare_inclusion(EditionInclusion::array(id.as_str(), EXTRA))
                .unwrap();
        }
        drop(editions);
        session.enable_edition(EXTRA).unwrap();
    }

    let batches = ParquetRecordBatchReaderBuilder::try_new(File::open(source).unwrap())
        .unwrap()
        .with_batch_size(65_536)
        .build()
        .unwrap();
    let schema = batches.schema();
    let arrays = arrays(&session, &schema, batches.map(|batch| batch.unwrap())).unwrap();
    write_options(
        &session,
        DEFAULT_ROW_BLOCK,
        vec![ProductResidual::dollar_volume()],
    )
    .blocking(&runtime)
    .write(BufWriter::new(File::create(dest).unwrap()), arrays)
    .unwrap();
}
