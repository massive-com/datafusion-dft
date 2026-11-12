//! How much FastLanes Delta could save if it were forced onto a file's integer columns: for each
//! chunk, every outermost integer array as long as the chunk (a column, or the integers under our
//! encodings) is decoded and compressed twice with the writer's compressor, as it is and as
//! Delta's bases and deltas (checked to decode to the same values). Prints bytes per array path,
//! summed over the chunks: as stored, recompressed, with Delta forced, and the smaller of stored
//! and Delta in each chunk.
//!
//! ```text
//! cargo run --release -p vortex-ohlc --example delta_bound -- <file.vortex> [<chunk>]
//! ```
//!
//! With a chunk index, it also prints that chunk's Delta arrays as trees.
//!
//! Delta isn't particularly GPU-friendly: decoding it is a prefix sum along each FastLanes lane,
//! a serial dependency within the lane, and Vortex's CUDA compressor preset leaves it out.

use std::collections::BTreeMap;

use vortex::array::arrays::PrimitiveArray;
use vortex::array::match_each_integer_ptype;
use vortex::array::{ArrayRef, ExecutionCtx, IntoArray, VortexSessionExecute};
use vortex::compressor::{BtrBlocksCompressor, BtrBlocksCompressorBuilder};
use vortex::encodings::fastlanes::{delta_compress, Delta};
use vortex::file::OpenOptionsSessionExt;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_ohlc::bars::data_child;
use vortex_ohlc::write::with_decimal_pow10;

#[derive(Default)]
struct Sizes {
    stored: u64,
    plain: u64,
    delta: u64,
    best: u64,
    delta_wins: u64,
    chunks: u64,
}

fn main() {
    let path = std::env::args().nth(1).expect("a file");
    let show: Option<usize> = std::env::args().nth(2).map(|chunk| chunk.parse().unwrap());
    let runtime = CurrentThreadRuntime::new();
    let session = VortexSession::default().with_handle(runtime.handle());
    vortex_ohlc::edition::enable(&session).unwrap();
    vortex_ohlc::register(&session);
    let compressor = with_decimal_pow10(BtrBlocksCompressorBuilder::from_session(&session)).build();
    let mut ctx = session.create_execution_ctx();

    let file = session
        .open_options()
        .open_buffer(std::fs::read(&path).unwrap())
        .unwrap();
    let stored = data_child(file.footer().layout())
        .unwrap_or_else(|_| std::sync::Arc::clone(file.footer().layout()));
    let reader = stored
        .new_reader(
            "".into(),
            file.segment_source(),
            &session,
            &Default::default(),
        )
        .unwrap();
    let chunks = ScanBuilder::new(session.clone(), reader)
        .into_array_iter(&runtime)
        .unwrap();
    let mut sizes: BTreeMap<String, Sizes> = BTreeMap::new();
    for (index, chunk) in chunks.enumerate() {
        let chunk = chunk.unwrap();
        let len = chunk.len();
        for (name, column) in chunk.named_children() {
            let mut leaves = Vec::new();
            integer_arrays(&name, &column, len, &mut leaves);
            for (path, array) in leaves {
                let entry = sizes.entry(path.clone()).or_default();
                let (plain, delta) = measure(&array, &compressor, &mut ctx);
                if show == Some(index) {
                    println!("chunk {index}, {path} ({}):", array.dtype());
                    print_tree("delta", &delta, 1);
                }
                let delta = delta.nbytes();
                entry.stored += array.nbytes();
                entry.plain += plain;
                entry.delta += delta;
                entry.best += array.nbytes().min(delta);
                entry.delta_wins += u64::from(delta < array.nbytes());
                entry.chunks += 1;
            }
        }
    }

    let mib = |bytes: u64| bytes as f64 / 1048576.0;
    println!(
        "{:<44} {:>9} {:>9} {:>9} {:>9} {:>11}",
        "array (MiB)", "stored", "recomp", "delta", "best", "delta wins"
    );
    for (path, s) in &sizes {
        println!(
            "{path:<44} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>5}/{:<5}",
            mib(s.stored),
            mib(s.plain),
            mib(s.delta),
            mib(s.best),
            s.delta_wins,
            s.chunks
        );
    }
    let total = |f: fn(&Sizes) -> u64| mib(sizes.values().map(f).sum());
    println!(
        "{:<44} {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
        "total",
        total(|s| s.stored),
        total(|s| s.plain),
        total(|s| s.delta),
        total(|s| s.best)
    );
}

/// The outermost integer arrays of length `len` at or under `array`, with their slot paths. A
/// `vortex_ohlc.scaled` array is looked through, to the divided values it stores.
fn integer_arrays(path: &str, array: &ArrayRef, len: usize, out: &mut Vec<(String, ArrayRef)>) {
    let scaled = array.encoding_id().as_ref() == "vortex_ohlc.scaled";
    if array.len() == len && array.dtype().is_int() && !scaled {
        out.push((path.to_string(), array.clone()));
        return;
    }
    for (name, child) in array.named_children() {
        integer_arrays(&format!("{path}/{name}"), &child, len, out);
    }
}

/// `array`'s bytes compressed by `compressor` as it is, and as Delta over its compressed bases
/// and deltas, which must decode to `array`'s values.
fn measure(
    array: &ArrayRef,
    compressor: &BtrBlocksCompressor,
    ctx: &mut ExecutionCtx,
) -> (u64, ArrayRef) {
    let primitive = array.clone().execute::<PrimitiveArray>(ctx).unwrap();
    let plain = compressor
        .compress(&primitive.clone().into_array(), ctx)
        .unwrap()
        .nbytes();
    let (bases, deltas) = delta_compress(&primitive, ctx).unwrap();
    let [bases, deltas] =
        [bases, deltas].map(|part| compressor.compress(&part.into_array(), ctx).unwrap());
    let delta = Delta::try_new(
        bases,
        deltas,
        primitive.validity().unwrap(),
        0,
        primitive.len(),
    )
    .unwrap()
    .into_array();
    let decoded = delta.clone().execute::<PrimitiveArray>(ctx).unwrap();
    match_each_integer_ptype!(primitive.ptype(), |T| {
        assert_eq!(decoded.as_slice::<T>(), primitive.as_slice::<T>());
    });
    (plain, delta)
}

/// `array` and everything under it, one line each: its slot name, encoding, length and bytes.
fn print_tree(name: &str, array: &ArrayRef, depth: usize) {
    println!(
        "{:indent$}{name}: {} len {} {} B",
        "",
        array.encoding_id(),
        array.len(),
        array.nbytes(),
        indent = depth * 2
    );
    for (name, child) in array.named_children() {
        print_tree(&name, &child, depth + 1);
    }
}
