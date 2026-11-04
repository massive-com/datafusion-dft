//! The files the benches read, in [`DATA_DIR`], written once from the source
//! ([`SOURCE`]) and reused after that:
//!
//! - `v2.parquet`: the source rewritten as Parquet v2 (zstd level 1, the source's row groups),
//!   the Parquet baseline;
//! - `bars-<VORTEX_VERSION>.vortex`: the source written with [`vortex_ohlc::write`], so under the
//!   bars layout. Bump [`VORTEX_VERSION`] whenever what the crate writes changes, so the file is
//!   written again.

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
use vortex_ohlc::write::{arrays, write_options, DEFAULT_ROW_BLOCK};

/// The environment variable naming the source: a month of minute bars, sorted by ticker.
const SOURCE: &str = "VORTEX_OHLC_SOURCE";

/// Where the files are written, under the workspace's `target`.
const DATA_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/vortex-ohlc-bench"
);

/// The version of what the crate writes, in the Vortex file's name: 2 since the OHLC layout, 3
/// since the bars layout at the root, 4 since it's gone, 5 since Vortex's compressor packs the
/// planned columns, 6 since Vortex's default strategy writes every column, 8 since the bars layout
/// is back at the root (7 was a trial of a patched compressor), 17 since the layout computes the
/// parts itself and nothing is planned or rescaled before writing, 18 since it keeps the prices'
/// own zones, 19 since the compressor may divide blocks by a power of ten (`vortex_ohlc::scaled`),
/// 20 since it may give decimals a FoR reference and bit width per block
/// (`vortex_ohlc::blocked_for`), 21 since that choice is estimated from the widths blocked
/// bit-packing picks, 22 since that encoding is `vortex_ohlc.scaled` (`vortex_ohlc.pow10` before),
/// 23 since its metadata has a base, 24 since every decimal is rescaled a chunk at a time
/// (`vortex_ohlc::scaled::DecimalPow10Scheme`), 25 since the dollar volume is stored as a residual
/// against volume × open (`vortex_ohlc::dollar_volume`), 26 since the layout's metadata says
/// whether it is. 9 to 16 were these steps in an earlier order.
const VORTEX_VERSION: u32 = 26;

/// Rows per batch when converting the source.
const CONVERT_BATCH_SIZE: usize = 65_536;

pub struct Files {
    pub parquet: PathBuf,
    pub vortex: PathBuf,
}

/// The files written from the source (see the module docs), writing those that are missing.
pub fn files() -> Files {
    let source = std::env::var_os(SOURCE).unwrap_or_else(|| panic!("set {SOURCE} to the source"));
    let source = Path::new(&source);
    assert!(source.exists(), "no source at {}", source.display());
    let dir = Path::new(DATA_DIR);
    std::fs::create_dir_all(dir).unwrap();
    let files = Files {
        parquet: dir.join("v2.parquet"),
        vortex: dir.join(format!("bars-{VORTEX_VERSION}.vortex")),
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
    write_options(&session, DEFAULT_ROW_BLOCK, true)
        .blocking(&runtime)
        .write(BufWriter::new(File::create(dest).unwrap()), arrays)
        .unwrap();
}
