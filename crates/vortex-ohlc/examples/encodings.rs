//! How the chunks of a vortex-ohlc file are stored: each column's encodings, as a tree with each
//! array's bytes, for the chunks asked for.
//!
//! ```text
//! cargo run --release -p vortex-ohlc --example encodings -- <file.vortex> <chunk>...
//! ```

use vortex::array::ArrayRef;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_ohlc::bars::data_child;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("a file");
    let wanted: Vec<usize> = args.map(|chunk| chunk.parse().unwrap()).collect();
    let runtime = CurrentThreadRuntime::new();
    let session = VortexSession::default().with_handle(runtime.handle());
    vortex_ohlc::register(&session);
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
    for (index, chunk) in chunks.enumerate() {
        if wanted.contains(&index) {
            println!("chunk {index}:");
            for (name, column) in chunk.unwrap().named_children() {
                print_tree(&name, &column, 1);
            }
        }
    }
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
