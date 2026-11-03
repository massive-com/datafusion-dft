//! The files the benches read, in [`DATA_DIR`], written once from [`SOURCE`] and reused after
//! that:
//!
//! - `v2.parquet`: the source rewritten as Parquet v2 (zstd level 1, the source's row groups),
//!   the Parquet baseline;
//! - `upstream-<VORTEX_VERSION>.vortex`: the source with the plan [`vortex_ohlc::profile`] picks
//!   for it, written with [`vortex_ohlc::write`]. Bump [`VORTEX_VERSION`] whenever what the crate
//!   writes changes, so the file is written again.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use arrow::array::RecordBatchReader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::{WriterProperties, WriterVersion};
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_ohlc::profile::{plan, DecimalProfiler};
use vortex_ohlc::write::{stored_arrays, write_options, DEFAULT_ROW_BLOCK};

/// The month of minute bars the benches read: August 2006, every ticker from A to ZZ,
/// sorted by ticker.
const SOURCE: &str = "/Users/techpizza/parquet/stag/atlas-serve/us_stocks_sip/\
    minute_aggs_by_ticker_monthly_v3/year_month=2006-08/_v=1771756535542/data.01.parquet";

/// Where the files are written, under the workspace's `target`.
const DATA_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/vortex-ohlc-bench"
);

/// The version of what the crate writes, in the Vortex file's name: 2 since the OHLC layout, 3
/// since the bars layout at the root, 4 since it's gone, 5 since Vortex's compressor packs the
/// planned columns, 6 since Vortex's default strategy writes every column.
const VORTEX_VERSION: u32 = 6;

/// Rows per batch when converting the source.
const CONVERT_BATCH_SIZE: usize = 65_536;

pub struct Files {
    pub parquet: PathBuf,
    pub vortex: PathBuf,
}

/// The files written from the source (see the module docs), writing those that are missing.
pub fn files() -> Files {
    let source = Path::new(SOURCE);
    assert!(source.exists(), "no source at {}", source.display());
    let dir = Path::new(DATA_DIR);
    std::fs::create_dir_all(dir).unwrap();
    let files = Files {
        parquet: dir.join("v2.parquet"),
        vortex: dir.join(format!("upstream-{VORTEX_VERSION}.vortex")),
    };
    // Each file is written under a temporary name and renamed once complete, so an interrupted
    // write isn't taken for a finished one.
    let ensure = |path: &Path, write: fn(&Path, &Path)| {
        if !path.exists() {
            eprintln!("writing {}", path.display());
            let tmp = path.with_extension("tmp");
            write(source, &tmp);
            std::fs::rename(&tmp, path).unwrap();
        }
    };
    ensure(&files.parquet, write_parquet_v2);
    ensure(&files.vortex, write_vortex);
    files
}

/// Read the Parquet file at `path`.
fn open_parquet(path: &Path) -> ParquetRecordBatchReaderBuilder<File> {
    ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap()).unwrap()
}

fn write_parquet_v2(source: &Path, dest: &Path) {
    let source = open_parquet(source);
    // Keep the source's row group size, so the format version is the only thing that changes.
    let row_group_rows = source.metadata().row_group(0).num_rows() as usize;
    let reader = source.with_batch_size(CONVERT_BATCH_SIZE).build().unwrap();
    let props = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(1).unwrap()))
        .set_max_row_group_row_count(Some(row_group_rows))
        .build();
    let mut writer =
        ArrowWriter::try_new(File::create(dest).unwrap(), reader.schema(), Some(props)).unwrap();
    for batch in reader {
        writer.write(&batch.unwrap()).unwrap();
    }
    writer.close().unwrap();
}

/// Profile the source, then write it with the plan that follows, recorded in the file.
fn write_vortex(source: &Path, dest: &Path) {
    // Vortex's defaults, able to write the file (see `vortex_ohlc::edition`), on a runtime that
    // runs everything on this thread.
    let runtime = CurrentThreadRuntime::new();
    let session = VortexSession::default().with_handle(runtime.handle());
    vortex_ohlc::edition::enable(&session).unwrap();
    let batches = || {
        open_parquet(source)
            .with_batch_size(CONVERT_BATCH_SIZE)
            .build()
            .unwrap()
    };
    let schema = batches().schema();
    let mut profiler = DecimalProfiler::new(&schema);
    for batch in batches() {
        profiler.add(&batch.unwrap());
    }
    let plan = plan(&profiler.finish());
    eprintln!("  plan: {plan:?}");

    let arrays = stored_arrays(&session, &plan, &schema, batches().map(|b| b.unwrap())).unwrap();
    write_options(&session, &schema, &plan, DEFAULT_ROW_BLOCK)
        .blocking(&runtime)
        .write(BufWriter::new(File::create(dest).unwrap()), arrays)
        .unwrap();
}
