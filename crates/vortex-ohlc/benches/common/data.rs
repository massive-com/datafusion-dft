//! The files the benches read, in [`DATA_DIR`], written once from [`SOURCE`] and reused after
//! that:
//!
//! - `v2.parquet`: the source rewritten as Parquet v2 (zstd level 1, the source's row groups),
//!   the Parquet baseline;
//! - `upstream-<VORTEX_VERSION>.vortex`: the source with the plan [`vortex_ohlc::profile`] picks
//!   for it (and `transactions` packed), written with [`vortex_ohlc::strategy`]. Bump
//!   [`VORTEX_VERSION`] whenever what the crate writes changes, so the file is written again.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::RecordBatchReader;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::{WriterProperties, WriterVersion};
use vortex::array::iter::ArrayIteratorAdapter;
use vortex::arrow::ArrowSessionExt;
use vortex::buffer::ByteBuffer;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::session::VortexSession;
use vortex_ohlc::plan::{apply_plan, planned_schema, transforms_json, Transform, TRANSFORMS_KEY};
use vortex_ohlc::profile::{plan, DecimalProfiler};
use vortex_ohlc::strategy::{table_strategy, write_strategy, DEFAULT_ROW_BLOCK};

use crate::common::parquet_read::open_parquet;

/// The month of minute bars `decimal_parts_bench` reads: August 2006, every ticker from A to ZZ,
/// sorted by ticker.
const SOURCE: &str = "/Users/techpizza/parquet/stag/atlas-serve/us_stocks_sip/\
    minute_aggs_by_ticker_monthly_v3/year_month=2006-08/_v=1771756535542/data.01.parquet";

/// Where the files are written, under the workspace's `target`.
const DATA_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/vortex-ohlc-bench"
);

/// The version of what the crate writes, in the Vortex file's name: 2 since the OHLC layout, 3
/// since the bars layout at the root.
const VORTEX_VERSION: u32 = 3;

/// Rows per batch when converting the source.
const CONVERT_BATCH_SIZE: usize = 65_536;

/// The integer column the plan packs, as `decimal_parts_bench` does.
const TRANSACTIONS: &str = "transactions";

pub struct Files {
    pub source: PathBuf,
    pub parquet: PathBuf,
    pub vortex: PathBuf,
    /// The same, without the bars layout (see [`vortex_ohlc::strategy::table_strategy`]).
    pub vortex_plain: PathBuf,
    /// The same, without the bars layout or the OHLC struct: the four prices rescaled like any
    /// other decimal column.
    pub vortex_prices: PathBuf,
}

/// The source, and the files written from it (see the module docs), writing those that are
/// missing.
pub fn files(runtime: &CurrentThreadRuntime, session: &VortexSession) -> Files {
    let source = PathBuf::from(SOURCE);
    assert!(source.exists(), "no source at {}", source.display());
    let dir = Path::new(DATA_DIR);
    std::fs::create_dir_all(dir).unwrap();
    let files = Files {
        parquet: dir.join("v2.parquet"),
        vortex: dir.join(format!("upstream-{VORTEX_VERSION}.vortex")),
        vortex_plain: dir.join(format!("upstream-plain-{VORTEX_VERSION}.vortex")),
        vortex_prices: dir.join(format!("upstream-prices-{VORTEX_VERSION}.vortex")),
        source,
    };
    // Each file is written under a temporary name and renamed once complete, so an interrupted
    // write isn't taken for a finished one.
    let tmp = |path: &Path| path.with_extension("tmp");
    if !files.parquet.exists() {
        eprintln!("writing {}", files.parquet.display());
        write_parquet_v2(&files.source, &tmp(&files.parquet));
        std::fs::rename(tmp(&files.parquet), &files.parquet).unwrap();
    }
    let vortex = [
        (&files.vortex, true, true),
        (&files.vortex_plain, false, true),
        (&files.vortex_prices, false, false),
    ];
    for (path, bars, ohlc) in vortex {
        if !path.exists() {
            eprintln!("writing {}", path.display());
            write_vortex(runtime, session, &files.source, &tmp(path), bars, ohlc);
            std::fs::rename(tmp(path), path).unwrap();
        }
    }
    files
}

fn write_parquet_v2(source: &Path, dest: &Path) {
    let source = open_parquet(source, false);
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

/// Profile the source, then write it with the plan that follows, recorded in the file, with the
/// bars layout or without it (`bars`), and with the prices in the OHLC struct or each rescaled on
/// its own (`ohlc`).
fn write_vortex(
    runtime: &CurrentThreadRuntime,
    session: &VortexSession,
    source: &Path,
    dest: &Path,
    bars: bool,
    ohlc: bool,
) {
    let batches = || {
        open_parquet(source, false)
            .with_batch_size(CONVERT_BATCH_SIZE)
            .build()
            .unwrap()
    };
    let schema = batches().schema();
    let mut profiler = DecimalProfiler::new(&schema);
    for batch in batches() {
        profiler.add(&batch.unwrap());
    }
    let mut plan = plan(&profiler.finish());
    plan.insert(TRANSACTIONS.to_string(), Transform::Packed);
    if !ohlc {
        for transform in plan.values_mut() {
            if let Transform::Ohlc(k) = *transform {
                *transform = Transform::Rescale(k);
            }
        }
    }
    eprintln!("  plan: {plan:?}");

    let strategy = if bars { write_strategy } else { table_strategy };
    let transforms = serde_json::to_vec(&transforms_json(&plan, &schema)).unwrap();
    let planned = planned_schema(&plan, &schema);
    let dtype = session.arrow().from_arrow_schema(&planned).unwrap();
    let chunks = batches().map({
        let session = session.clone();
        let (plan, planned) = (plan.clone(), Arc::clone(&planned));
        move |batch| {
            let batch = apply_plan(&plan, &planned, batch.unwrap());
            session.arrow().from_arrow_record_batch(batch, &planned)
        }
    });
    session
        .write_options()
        .with_strategy(strategy(&schema, &plan, DEFAULT_ROW_BLOCK))
        .with_metadata_segment(TRANSFORMS_KEY, ByteBuffer::copy_from(transforms))
        .blocking(runtime)
        .write(
            BufWriter::new(File::create(dest).unwrap()),
            ArrayIteratorAdapter::new(dtype, chunks),
        )
        .unwrap();
}
