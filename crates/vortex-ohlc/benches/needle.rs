//! Needle lookups, every column of the matching rows, single-threaded: the Parquet v2 file (row
//! group statistics and a row filter), and the Vortex file read both ways (see
//! `common::vortex_read`) with a filtered scan (`filter`) and a row-index-first one
//! (`two-phase`). Every reader is checked against Parquet first (skip that with `VERIFIED=1`).
//!
//! Each lookup runs against a file that is already open (its footer parsed); `COLD=1` also times
//! them opening the file for every lookup (`/open`). `NEEDLES=<name>,<name>` runs only the
//! needles named, and criterion's own filter picks lookups by name, e.g. `-- two-phase`. See
//! `common::data` for the files and how to point at them.

#[allow(dead_code)]
mod common;

use std::sync::Arc;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use vortex::io::runtime::current::CurrentThreadRuntime;

use common::needle::{needle_selected, needles};
use common::parquet_read::{parquet_needle_batches, scan_metadata};
use common::vortex_read::{new_session, open_vortex, read_plan, vortex_batches, Lookup, ReadPath};
use common::{assert_same, data, drain, verified};

fn lookups(c: &mut Criterion) {
    let runtime = CurrentThreadRuntime::new();
    let session = new_session(&runtime);
    let files = data::files(&runtime, &session);
    let open = open_vortex(&runtime, &session, &files.vortex);
    let plan = read_plan(&runtime, &session, &files.vortex);
    let metadata = scan_metadata(&files.parquet);
    let source = Arc::clone(metadata.schema());
    let cold = std::env::var("COLD").is_ok_and(|v| v == "1");

    for needle in needles().iter().filter(|n| needle_selected(n.name)) {
        let parquet = |metadata| parquet_needle_batches(&files.parquet, metadata, needle);
        let vortex = |file, path, lookup| {
            vortex_batches(&runtime, &session, file, &plan, &source, None, path, lookup)
        };
        let ways = |path| {
            [
                ("filter", Lookup::Filter(needle)),
                ("two-phase", Lookup::TwoPhase(needle)),
            ]
            .map(|(how, lookup)| (format!("{} {how}", ReadPath::name(path)), path, lookup))
        };
        let ways: Vec<_> = ReadPath::ALL.into_iter().flat_map(ways).collect();

        // Every reader, and both Vortex lookups, must return exactly the same rows.
        let mut rows = drain(parquet(&metadata));
        for (name, path, lookup) in ways.iter().filter(|_| !verified()) {
            let label = format!("{}: {name}", needle.name);
            rows = assert_same(&label, parquet(&metadata), vortex(&open, *path, *lookup));
        }

        let mut group = c.benchmark_group(format!("needle/{}", needle.name));
        group
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(3))
            .throughput(Throughput::Elements(rows as u64));
        group.bench_function("parquet v2", |b| b.iter(|| drain(parquet(&metadata))));
        if cold {
            group.bench_function("parquet v2/open", |b| {
                b.iter(|| {
                    let metadata = scan_metadata(&files.parquet);
                    drain(parquet_needle_batches(&files.parquet, &metadata, needle))
                })
            });
        }
        for (name, path, lookup) in &ways {
            group.bench_function(name.as_str(), |b| {
                b.iter(|| drain(vortex(&open, *path, *lookup)))
            });
            if cold {
                group.bench_function(format!("{name}/open"), |b| {
                    b.iter(|| {
                        let file = open_vortex(&runtime, &session, &files.vortex);
                        drain(vortex_batches(
                            &runtime, &session, &file, &plan, &source, None, *path, *lookup,
                        ))
                    })
                });
            }
        }
        group.finish();
    }
}

criterion_group!(benches, lookups);
criterion_main!(benches);
