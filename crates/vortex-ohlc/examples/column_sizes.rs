//! Each top-level column's share of Vortex files: the bytes of every segment under it, its zone
//! maps and dictionary included.
//!
//! ```text
//! cargo run --release -p vortex-ohlc --example column_sizes -- <file.vortex>...
//! ```

use std::sync::Arc;

use vortex::file::OpenOptionsSessionExt;
use vortex::layout::LayoutRef;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;

fn main() {
    let session = VortexSession::default();
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).unwrap();
        let total = bytes.len();
        let file = session.open_options().open_buffer(bytes).unwrap();
        let segments = Arc::clone(file.footer().segment_map());
        let fields = file.dtype().as_struct_fields_opt().expect("a table");
        let names: Vec<_> = fields.names().iter().map(|name| name.to_string()).collect();
        let table = find_table(file.footer().layout(), &names).expect("the table's layout");
        println!("{path}: {total} bytes");
        for (name, column) in table.child_names().zip(table.children().unwrap()) {
            let bytes: u64 = segment_ids(&column)
                .into_iter()
                .map(|id| u64::from(segments[id as usize].length))
                .sum();
            println!(
                "  {name:<16} {bytes:>12}  {:>7.1} MiB",
                bytes as f64 / 1048576.0
            );
        }
    }
}

/// The first layout at or under `layout` whose children are the columns `names`.
fn find_table(layout: &LayoutRef, names: &[String]) -> Option<LayoutRef> {
    if layout
        .child_names()
        .map(|name| name.to_string())
        .eq(names.iter().cloned())
    {
        return Some(Arc::clone(layout));
    }
    layout
        .children()
        .ok()?
        .iter()
        .find_map(|child| find_table(child, names))
}

/// The segments of `layout` and every layout under it.
fn segment_ids(layout: &LayoutRef) -> Vec<u32> {
    let mut ids: Vec<u32> = layout.segment_ids().iter().map(|id| **id).collect();
    for child in layout.children().unwrap() {
        ids.extend(segment_ids(&child));
    }
    ids
}
