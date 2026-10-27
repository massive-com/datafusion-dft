//! Needle lookups, every column of the matching rows, single-threaded: the Parquet v2 file (row
//! group statistics and a row filter), and the Vortex file read both ways (see
//! `common::vortex_read`) with a filtered scan (`filter`) and a row-index-first one
//! (`two-phase`). Each runs against a file that is already open (its footer parsed), and, as
//! `<reader>/open`, opening it for every lookup.
//!
//! Criterion's filter picks the lookups, e.g. `cargo bench --bench needle -- 'high > 1000'`, or
//! `-- 'needle/high > 1000/vortex expr two-phase$'`. Before timing a needle, every reader of it
//! that runs is checked against Parquet. See `common::data` for the files.

#[allow(dead_code)]
mod common;

use std::sync::Arc;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use vortex::io::runtime::current::CurrentThreadRuntime;

use common::needle::needles;
use common::parquet_read::{parquet_needle_batches, scan_metadata};
use common::vortex_read::{new_session, open_vortex, read_plan, vortex_batches, Lookup, ReadPath};
use common::{assert_same, data, drain, Selection};

const PARQUET: &str = "parquet v2";

fn lookups(c: &mut Criterion) {
    let selection = Selection::from_args();
    let runtime = CurrentThreadRuntime::new();
    let session = new_session(&runtime);
    let files = data::files(&runtime, &session);
    let open = open_vortex(&runtime, &session, &files.vortex);
    let plan = read_plan(&runtime, &session, &files.vortex);
    let metadata = scan_metadata(&files.parquet);
    let source = Arc::clone(metadata.schema());

    for needle in &needles() {
        let id = |reader: &str| format!("needle/{}/{reader}", needle.name);
        let runs = |reader: &str| {
            selection.runs(&id(reader)) || selection.runs(&format!("{}/open", id(reader)))
        };
        let ways = |path| {
            [
                ("filter", Lookup::Filter(needle)),
                ("two-phase", Lookup::TwoPhase(needle)),
            ]
            .map(|(how, lookup)| (format!("{} {how}", ReadPath::name(path)), path, lookup))
        };
        let ways: Vec<_> = ReadPath::ALL
            .into_iter()
            .flat_map(ways)
            .filter(|(name, _, _)| runs(name))
            .collect();
        if ways.is_empty() && !runs(PARQUET) {
            continue;
        }

        let parquet = |metadata| parquet_needle_batches(&files.parquet, metadata, needle);
        let vortex = |file, path, lookup| {
            vortex_batches(&runtime, &session, file, &plan, &source, None, path, lookup)
        };
        // Every reader, and both Vortex lookups, must return exactly the same rows.
        let mut rows = drain(parquet(&metadata));
        for (name, path, lookup) in &ways {
            let label = format!("{}: {name}", needle.name);
            rows = assert_same(&label, parquet(&metadata), vortex(&open, *path, *lookup));
        }

        let mut group = c.benchmark_group(format!("needle/{}", needle.name));
        group
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(3))
            .throughput(Throughput::Elements(rows as u64));
        group.bench_function(PARQUET, |b| b.iter(|| drain(parquet(&metadata))));
        group.bench_function(format!("{PARQUET}/open"), |b| {
            b.iter(|| {
                let metadata = scan_metadata(&files.parquet);
                drain(parquet_needle_batches(&files.parquet, &metadata, needle))
            })
        });
        for (name, path, lookup) in &ways {
            group.bench_function(name.as_str(), |b| {
                b.iter(|| drain(vortex(&open, *path, *lookup)))
            });
            group.bench_function(format!("{name}/open"), |b| {
                b.iter(|| {
                    let file = open_vortex(&runtime, &session, &files.vortex);
                    drain(vortex_batches(
                        &runtime, &session, &file, &plan, &source, None, *path, *lookup,
                    ))
                })
            });
        }
        group.finish();
    }
}

criterion_group!(benches, lookups);
criterion_main!(benches);
