//! The files the benches read, in [`DATA_DIR`], written once from the source
//! ([`SOURCE`]) and reused after that:
//!
//! - `v2.parquet`: the source rewritten as Parquet v2 (zstd level 1, the source's row groups),
//!   the Parquet baseline;
//! - `bars-<VORTEX_VERSION>.vortex`: the source written with [`vortex_ohlc::write`], so under the
//!   bars layout. Bump [`VORTEX_VERSION`] whenever what the crate writes changes, so the file is
//!   written again.
//!
//! `DF_BENCH_VORTEX=<path>` reads the Vortex file at `<path>` in place of `bars-*.vortex`
//! ([`Files::vortex_name`]).

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
use vortex_ohlc::bars::ProductResidual;
use vortex_ohlc::write::{arrays, write_options, DEFAULT_ROW_BLOCK};

/// The environment variable naming the source: a month of minute bars, sorted by ticker.
const SOURCE: &str = "VORTEX_OHLC_SOURCE";

/// Where the files are written, under the workspace's `target`.
const DATA_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/vortex-ohlc-bench"
);

/// The version suffix for Vortex files.
const VORTEX_VERSION: u32 = 28;

/// Rows per batch when converting the source.
const CONVERT_BATCH_SIZE: usize = 65_536;

pub struct Files {
    pub parquet: PathBuf,
    pub vortex: PathBuf,
    /// The Vortex file's name without its extension, when `DF_BENCH_VORTEX` names it in place of
    /// the one written here.
    pub vortex_name: Option<String>,
}

/// The files written from the source (see the module docs), writing those that are missing.
pub fn files() -> Files {
    let source = std::env::var_os(SOURCE).unwrap_or_else(|| panic!("set {SOURCE} to the source"));
    let source = Path::new(&source);
    assert!(source.exists(), "no source at {}", source.display());
    let dir = Path::new(DATA_DIR);
    std::fs::create_dir_all(dir).unwrap();
    let other = std::env::var_os("DF_BENCH_VORTEX").map(PathBuf::from);
    if let Some(other) = &other {
        assert!(other.exists(), "no Vortex file at {}", other.display());
    }
    let vortex_name = other.as_ref().map(|path| {
        let stem = path.file_stem().and_then(|stem| stem.to_str());
        stem.expect("a UTF-8 file name").to_string()
    });
    let files = Files {
        parquet: dir.join("v2.parquet"),
        vortex: other.unwrap_or_else(|| dir.join(format!("bars-{VORTEX_VERSION}.vortex"))),
        vortex_name,
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

fn write_vortex(source: &Path, dest: &Path) {
    // Vortex's defaults, able to write the file (see `vortex_ohlc::edition`), on a runtime that
    // runs everything on this thread.
    let runtime = CurrentThreadRuntime::new();
    let session = VortexSession::default().with_handle(runtime.handle());
    vortex_ohlc::edition::enable(&session).unwrap();
    vortex_ohlc::register(&session);
    let batches = open_parquet(source)
        .with_batch_size(CONVERT_BATCH_SIZE)
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
