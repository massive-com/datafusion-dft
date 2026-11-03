//! The encodings Vortex chose for one chunk of each file given: every column, a struct's fields
//! each on their own line, with its bytes in memory and its tree of encodings.
//!
//! ```text
//! cargo run --release -p vortex-ohlc --example chunk_encodings -- <chunk> <file.vortex>...
//! ```

use vortex::array::ArrayRef;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;

fn main() {
    let runtime = CurrentThreadRuntime::new();
    let session = VortexSession::default().with_handle(runtime.handle());
    let mut args = std::env::args().skip(1);
    let chunk: usize = args.next().expect("a chunk index").parse().unwrap();
    for path in args {
        let bytes = std::fs::read(&path).unwrap();
        let file = session.open_options().open_buffer(bytes).unwrap();
        let array = file
            .scan()
            .unwrap()
            .into_array_iter(&runtime)
            .unwrap()
            .nth(chunk)
            .expect("no such chunk")
            .unwrap();
        println!("{path}, chunk {chunk}:");
        for (name, column) in array.named_children() {
            match column.dtype().is_struct() {
                true => {
                    for (field, array) in column.named_children() {
                        print(&format!("{name}.{field}"), &array);
                    }
                }
                false => print(&name, &column),
            }
        }
    }
}

fn print(name: &str, array: &ArrayRef) {
    println!("  {name:<16} {:>9} B  {}", array.nbytes(), tree(array));
}

/// `array`'s encoding, with its children's in parentheses.
fn tree(array: &ArrayRef) -> String {
    let children: Vec<String> = array
        .named_children()
        .into_iter()
        .map(|(name, child)| format!("{name}={}", tree(&child)))
        .collect();
    match children.is_empty() {
        true => array.encoding_id().to_string(),
        false => format!("{}({})", array.encoding_id(), children.join(", ")),
    }
}
