//! What a scan's split computation (`SplitBy::Layout`, which every scan runs over the whole file)
//! walks in Vortex files: each chunked layout's path and chunk count, and the time to compute
//! the splits of all columns, with how many it finds: with `FieldMask::All` or a mask per source
//! column (as a `SELECT *` scan asks), each on one reader or on a fresh reader each time (as
//! each query gets).
//!
//! ```text
//! cargo run --release -p vortex-ohlc --example split_boundaries -- <file.vortex>...
//! ```

use std::time::Instant;

use vortex::array::dtype::{FieldMask, FieldPath};
use vortex::file::OpenOptionsSessionExt;
use vortex::layout::scan::split_by::SplitBy;
use vortex::layout::LayoutRef;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;

/// How many times the splits are computed, for the time of one.
const ITERATIONS: u32 = 2000;

fn main() {
    let session = VortexSession::default();
    vortex_ohlc::register(&session);
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).unwrap();
        let file = session.open_options().open_buffer(bytes).unwrap();
        println!("{path}: {} rows", file.row_count());
        let mut chunked = Vec::new();
        walk(file.footer().layout(), String::new(), &mut chunked);
        for (path, chunks, first) in &chunked {
            println!("  {path:<40} {chunks:>6} chunks, the first of {first} rows");
        }
        println!("  {} chunked layouts", chunked.len());
        // `FieldMask::All`, and a mask per source column, as a scan of every column asks.
        let all = vec![FieldMask::All];
        let per_column: Vec<FieldMask> = (file.dtype().as_struct_fields().names().iter())
            .map(|name| FieldMask::Prefix(FieldPath::from_name(name.clone())))
            .collect();
        let rows = 0..file.row_count();
        for (masks, label) in [(&all, "all"), (&per_column, "per column")] {
            for fresh in [false, true] {
                let reader = file.layout_reader().unwrap();
                let mut len = 0;
                let start = Instant::now();
                for _ in 0..ITERATIONS {
                    let fresh_reader;
                    let reader = if fresh {
                        fresh_reader = file.layout_reader().unwrap();
                        &fresh_reader
                    } else {
                        &reader
                    };
                    let splits = SplitBy::Layout
                        .splits(reader.as_ref(), &rows, masks)
                        .unwrap();
                    len = splits.len();
                    std::hint::black_box(splits);
                }
                let us = start.elapsed().as_secs_f64() * 1e6 / f64::from(ITERATIONS);
                let reader = if fresh {
                    "a fresh reader"
                } else {
                    "one reader"
                };
                println!("  masks {label:<10} {reader:<14} {len:>5} boundaries {us:>7.1} µs");
            }
        }
    }
}

/// Every chunked layout at or under `layout` (at `path`), with its chunk count and the first
/// chunk's rows.
fn walk(layout: &LayoutRef, path: String, out: &mut Vec<(String, usize, u64)>) {
    let children = layout.children().unwrap();
    if layout.encoding_id().as_ref() == "vortex.chunked" {
        let first = children.first().map_or(0, |child| child.row_count());
        out.push((path.clone(), layout.nchildren(), first));
        return;
    }
    for (name, child) in layout.child_names().zip(children.iter()) {
        let path = if path.is_empty() {
            name.to_string()
        } else {
            format!("{path}/{name}")
        };
        walk(child, path, out);
    }
}
