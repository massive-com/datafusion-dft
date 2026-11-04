//! What storing a decimal column a block of 1024 values at a time could save: its values' bits
//! under each way of packing a block, summed over a vortex-ohlc file's stored chunks.
//!
//! ```text
//! cargo run --release -p vortex-ohlc --example block_model -- <file.vortex> <column>...
//! ```
//!
//! A column is a stored column's name, or `ohlc.<part>`. Each block is packed at the width of
//! its widest value, as:
//!
//! - `bitpack`: the values themselves (zigzag for negatives);
//! - `for`: less the block's minimum (blocked FoR);
//! - `pow10`: divided by the most trailing zeros all its values have, then less the minimum;
//! - `pow10+patches`: divided by the exponent that packs it smallest once the values that
//!   aren't its multiples are patched, at 96 bits a patch (a row index and a value), then less
//!   the minimum;
//! - `pow10+patches, no for`: the same without the minimum taken off;
//! - `pow10+patches+trim`: as `pow10+patches`, also patching up to 8 of the lowest and 8 of the
//!   highest values where that narrows the block more than the patches cost.

use vortex::array::arrays::DecimalArray;
use vortex::array::{ArrayRef, VortexSessionExecute};
use vortex::dtype::DecimalType;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_ohlc::bars::data_child;

const BLOCK: usize = 1024;
const PATCH_BITS: u64 = 96;
const MODELS: [&str; 6] = [
    "bitpack",
    "for",
    "pow10",
    "pow10+patches",
    "pow10+patches, no for",
    "pow10+patches+trim",
];
const MAX_TRIM: usize = 8;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("a file");
    let columns: Vec<String> = args.collect();
    let runtime = CurrentThreadRuntime::new();
    let session = VortexSession::default().with_handle(runtime.handle());
    vortex_ohlc::register(&session);
    let file = session
        .open_options()
        .open_buffer(std::fs::read(&path).unwrap())
        .unwrap();
    let stored = data_child(file.footer().layout()).unwrap();
    let reader = stored
        .new_reader(
            "".into(),
            file.segment_source(),
            &session,
            &Default::default(),
        )
        .unwrap();
    let mut ctx = session.create_execution_ctx();
    let mut bits = vec![[0u64; MODELS.len()]; columns.len()];
    let mut exponents = vec![[0u64; 40]; columns.len()];
    for chunk in ScanBuilder::new(session.clone(), reader)
        .into_array_iter(&runtime)
        .unwrap()
    {
        let chunk = chunk.unwrap();
        for (c, column) in columns.iter().enumerate() {
            let values = values(&field(&chunk, column), &mut ctx);
            for block in values.chunks(BLOCK) {
                let (block_bits, exponent) = model(block);
                for (total, b) in bits[c].iter_mut().zip(block_bits) {
                    *total += b;
                }
                exponents[c][exponent as usize] += 1;
            }
        }
    }
    for (c, column) in columns.iter().enumerate() {
        println!("{column}:");
        for (name, b) in MODELS.iter().zip(bits[c]) {
            println!("  {name:<14} {:>7.1} MiB", b as f64 / 8.0 / 1048576.0);
        }
        let used: Vec<String> = (0..40)
            .filter(|&e| exponents[c][e] > 0)
            .map(|e| format!("{e}: {}", exponents[c][e]))
            .collect();
        println!("  blocks by pow10+patches exponent: {}", used.join(", "));
    }
}

/// The column `path` (`name` or `struct.field`) of a chunk.
fn field(chunk: &ArrayRef, path: &str) -> ArrayRef {
    let mut array = chunk.clone();
    for name in path.split('.') {
        array = array
            .named_children()
            .into_iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("no column {path}"))
            .1;
    }
    array
}

/// A decimal column's unscaled values.
fn values(array: &ArrayRef, ctx: &mut vortex::array::ExecutionCtx) -> Vec<i128> {
    let decimal = array.clone().execute::<DecimalArray>(ctx).unwrap();
    match decimal.values_type() {
        DecimalType::I8 => decimal.buffer::<i8>().iter().map(|&v| v.into()).collect(),
        DecimalType::I16 => decimal.buffer::<i16>().iter().map(|&v| v.into()).collect(),
        DecimalType::I32 => decimal.buffer::<i32>().iter().map(|&v| v.into()).collect(),
        DecimalType::I64 => decimal.buffer::<i64>().iter().map(|&v| v.into()).collect(),
        DecimalType::I128 => decimal.buffer::<i128>().to_vec(),
        other => panic!("{other:?} values"),
    }
}

fn width(range: i128) -> u64 {
    (128 - range.leading_zeros()) as u64
}

fn trailing_zeros(v: i128) -> u32 {
    if v == 0 {
        return 38;
    }
    let (mut v, mut zeros) = (v, 0);
    while v % 10 == 0 {
        v /= 10;
        zeros += 1;
    }
    zeros
}

/// A block's bits under each of [`MODELS`], and the exponent `pow10+patches` picked.
fn model(block: &[i128]) -> ([u64; MODELS.len()], u32) {
    let n = block.len() as u64;
    let (min, max) = (*block.iter().min().unwrap(), *block.iter().max().unwrap());
    let zigzag = |v: i128| if v < 0 { -2 * v - 1 } else { 2 * v };
    let packed = |values: &[i128]| n * values.iter().map(|&v| width(zigzag(v))).max().unwrap_or(0);
    let bitpack = packed(block);
    let for_ = n * width(max - min);
    let zeros: Vec<u32> = block.iter().map(|&v| trailing_zeros(v)).collect();
    let common = *zeros.iter().min().unwrap();
    let factor = |e: u32| 10i128.pow(e.min(18));
    let pow10 = n * width(max / factor(common) - min / factor(common));
    let (mut patched, mut exponent) = (pow10, common);
    let (mut no_for, mut trimmed) = (
        packed(
            &block
                .iter()
                .map(|&v| v / factor(common))
                .collect::<Vec<_>>(),
        ),
        pow10,
    );
    for e in common..=18 {
        let mut kept: Vec<i128> = block
            .iter()
            .zip(&zeros)
            .filter(|&(_, &z)| z >= e)
            .map(|(&v, _)| v / factor(e))
            .collect();
        let misfits = n - kept.len() as u64;
        if kept.is_empty() {
            let cost = misfits * PATCH_BITS;
            (patched, no_for, trimmed) = (patched.min(cost), no_for.min(cost), trimmed.min(cost));
            break;
        }
        kept.sort_unstable();
        let cost = n * width(kept[kept.len() - 1] - kept[0]) + misfits * PATCH_BITS;
        if cost < patched {
            (patched, exponent) = (cost, e);
        }
        no_for = no_for.min(packed(&kept) + misfits * PATCH_BITS);
        for low in 0..=MAX_TRIM.min(kept.len() - 1) {
            for high in 0..=MAX_TRIM.min(kept.len() - 1 - low) {
                let range = kept[kept.len() - 1 - high] - kept[low];
                let cost = n * width(range) + (misfits + (low + high) as u64) * PATCH_BITS;
                trimmed = trimmed.min(cost);
            }
        }
    }
    ([bitpack, for_, pow10, patched, no_for, trimmed], exponent)
}
