//! Linear scans to Arrow in the source's types, single-threaded: the Parquet v2 file, and the
//! Vortex file read both ways (see `common::vortex_read`), for three projections: every column,
//! the decimal columns, and dollar_volume alone. Every reader is checked against Parquet first
//! (skip that with `VERIFIED=1`). See `common::data` for the files and how to point at them.

#[allow(dead_code)]
mod common;

use std::sync::Arc;

use arrow::datatypes::DataType;
use criterion::{criterion_group, criterion_main, Criterion, SamplingMode, Throughput};
use vortex::io::runtime::current::CurrentThreadRuntime;

use common::parquet_read::{parquet_batches, scan_metadata};
use common::vortex_read::{new_session, open_vortex, read_plan, vortex_batches, Lookup, ReadPath};
use common::{assert_same, data, drain, verified};

fn scans(c: &mut Criterion) {
    let runtime = CurrentThreadRuntime::new();
    let session = new_session(&runtime);
    let files = data::files(&runtime, &session);
    let plan = read_plan(&runtime, &session, &files.vortex);
    let source = Arc::clone(scan_metadata(&files.parquet).schema());
    let decimals: Vec<&str> = source
        .fields()
        .iter()
        .filter(|f| matches!(f.data_type(), DataType::Decimal128(..)))
        .map(|f| f.name().as_str())
        .collect();
    let projections: [(&str, Option<&[&str]>); 3] = [
        ("all columns", None),
        ("decimal columns", Some(&decimals)),
        ("dollar_volume", Some(&["dollar_volume"])),
    ];

    for (label, columns) in projections {
        let read_vortex = |path| {
            let file = open_vortex(&runtime, &session, &files.vortex);
            vortex_batches(
                &runtime,
                &session,
                &file,
                &plan,
                &source,
                columns,
                path,
                Lookup::All,
            )
        };
        let rows = match verified() {
            true => drain(parquet_batches(&files.parquet, columns)),
            false => ReadPath::ALL
                .map(|path| {
                    assert_same(
                        &format!("{label}: {}", path.name()),
                        parquet_batches(&files.parquet, columns),
                        read_vortex(path),
                    )
                })
                .into_iter()
                .max()
                .unwrap(),
        };

        let mut group = c.benchmark_group(format!("scan/{label}"));
        group
            .sampling_mode(SamplingMode::Flat)
            .sample_size(10)
            .throughput(Throughput::Elements(rows as u64));
        group.bench_function("parquet v2", |b| {
            b.iter(|| drain(parquet_batches(&files.parquet, columns)))
        });
        for path in ReadPath::ALL {
            group.bench_function(path.name(), |b| b.iter(|| drain(read_vortex(path))));
        }
        group.finish();
    }
}

criterion_group!(benches, scans);
criterion_main!(benches);
