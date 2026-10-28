//! Needle lookups, every column of the matching rows, single-threaded: the Parquet v2 file (row
//! group statistics and a row filter), and the Vortex file read both ways (see
//! `common::vortex_read`) with a filtered scan (`filter`) and a row-index-first one
//! (`two-phase`). Each runs against a file that is already open (its footer parsed), and, as
//! `<reader>/open`, opening it for every lookup.
//!
//! Criterion's filter picks the lookups, e.g. `cargo bench --bench needle -- 'high > 1000'`, or
//! `-- 'needle/high > 1000/vortex expr two-phase$'`. Before timing a needle, every reader of it
//! that runs is checked against Parquet, and what the Vortex engine counted for each lookup is
//! printed: the I/O, next to the chunks that hold a match and the row groups Parquet keeps. See
//! `common::data` for the files.

#[allow(dead_code)]
mod common;

use std::fs::File;
use std::sync::Arc;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex_ohlc::strategy::DEFAULT_ROW_BLOCK;

use common::needle::{candidate_row_groups, needles, vortex_filter};
use common::parquet_read::{parquet_needle_batches, scan_metadata};
use common::vortex_read::{
    lookup_stats, matching_rows, new_session, open_vortex, read_plan, vortex_batches, Lookup,
    ReadPath,
};
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

        // What the engines did: the chunks that hold a match (where the whole needle goes into
        // the Vortex filter), the row groups Parquet keeps (one per chunk), and what the Vortex
        // engine counted for each lookup.
        let file = File::open(&files.parquet).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata.clone());
        let row_groups = candidate_row_groups(&builder, &needle.terms).len();
        let chunks = builder.metadata().num_row_groups();
        let hit = match vortex_filter(&open, needle, &plan) {
            (Some(filter), after) if after.is_empty() => {
                let mut hit: Vec<u64> = matching_rows(&runtime, &session, &open, &filter)
                    .into_iter()
                    .map(|row| row / DEFAULT_ROW_BLOCK as u64)
                    .collect();
                hit.dedup();
                hit.len().to_string()
            }
            _ => "?".to_string(),
        };
        println!(
            "{}: {rows} rows in {hit} of {chunks} chunks; {PARQUET} keeps {row_groups} row groups",
            needle.name
        );
        for (name, path, lookup) in &ways {
            let (n, counters) = lookup_stats(
                &runtime,
                &files.vortex,
                &plan,
                &source,
                None,
                *path,
                *lookup,
            );
            assert_eq!(n, rows, "{}: {name}, counted", needle.name);
            let counters: Vec<String> = counters.iter().map(|(k, v)| format!("{k}={v}")).collect();
            println!("  {name}: {}", counters.join(" "));
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
