//! What the scan and needle benches share, ported from encoderrr's `decimal_parts_bench`: the
//! files they read (see [`data`]), the needles ([`needle`]), and the readers for each format
//! ([`parquet_read`], [`vortex_read`]).

pub mod data;
pub mod needle;
pub mod parquet_read;
pub mod vortex_read;

use std::hint::black_box;

use arrow::array::{Array, RecordBatch};
use regex::Regex;

/// Check that two streams of batches hold the same values, whatever their batch boundaries;
/// returns the row count.
pub fn assert_same(
    label: &str,
    expected: impl Iterator<Item = RecordBatch>,
    actual: impl Iterator<Item = RecordBatch>,
) -> usize {
    let mut expected = expected.peekable();
    let mut actual = actual.peekable();
    let (mut e_off, mut a_off, mut rows) = (0, 0, 0);
    loop {
        // Drop exhausted (and empty) batches from the front of each stream.
        while expected.peek().is_some_and(|b| e_off == b.num_rows()) {
            expected.next();
            e_off = 0;
        }
        while actual.peek().is_some_and(|b| a_off == b.num_rows()) {
            actual.next();
            a_off = 0;
        }
        let (e, a) = match (expected.peek(), actual.peek()) {
            (None, None) => return rows,
            (Some(_), None) | (None, Some(_)) => {
                panic!("{label}: row count differs after {rows} rows")
            }
            (Some(e), Some(a)) => (e, a),
        };
        let n = (e.num_rows() - e_off).min(a.num_rows() - a_off);
        assert_eq!(e.num_columns(), a.num_columns(), "{label}: column count");
        for idx in 0..e.num_columns() {
            let want = e.column(idx).slice(e_off, n);
            let got = a.column(idx).slice(a_off, n);
            assert!(
                want.as_ref() == got.as_ref(),
                "{label}: column {} differs in rows {rows}..{} ({} vs {})",
                e.schema().field(idx).name(),
                rows + n,
                want.data_type(),
                got.data_type(),
            );
        }
        e_off += n;
        a_off += n;
        rows += n;
    }
}

/// Consume `batches`, keeping each from being optimized away; returns the row count.
pub fn drain(batches: impl Iterator<Item = RecordBatch>) -> usize {
    batches
        .map(|batch| {
            let rows = batch.num_rows();
            black_box(batch);
            rows
        })
        .sum()
}

/// Which benchmarks criterion runs, read from the command line it parses (`cargo bench --bench
/// needle -- <FILTER>`): a regex over benchmark ids, or the exact id with `--exact`. A bench sets
/// up and checks only the readers that will run.
pub struct Selection {
    filter: Filter,
}

enum Filter {
    All,
    Regex(Regex),
    Exact(String),
    None,
}

/// Criterion's options that take a value, which isn't a filter.
const VALUE_FLAGS: &[&str] = &[
    "-c",
    "--color",
    "-s",
    "--save-baseline",
    "-b",
    "--baseline",
    "--baseline-lenient",
    "--format",
    "--profile-time",
    "--load-baseline",
    "--sample-size",
    "--warm-up-time",
    "--measurement-time",
    "--nresamples",
    "--noise-threshold",
    "--confidence-level",
    "--significance-level",
    "--plotting-backend",
    "--output-format",
];

impl Selection {
    pub fn from_args() -> Self {
        let mut args = std::env::args().skip(1);
        let (mut filter, mut exact, mut ignored) = (None, false, false);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--exact" => exact = true,
                "--ignored" => ignored = true,
                flag if VALUE_FLAGS.contains(&flag) => {
                    args.next();
                }
                flag if flag.starts_with('-') => {}
                _ => filter = Some(arg),
            }
        }
        let filter = match (filter, exact, ignored) {
            (_, _, true) => Filter::None,
            (None, _, _) => Filter::All,
            (Some(id), true, _) => Filter::Exact(id),
            (Some(regex), false, _) => Filter::Regex(Regex::new(&regex).expect("a valid filter")),
        };
        Self { filter }
    }

    /// Whether the benchmark `id` runs.
    pub fn runs(&self, id: &str) -> bool {
        match &self.filter {
            Filter::All => true,
            Filter::Regex(regex) => regex.is_match(id),
            Filter::Exact(exact) => id == exact,
            Filter::None => false,
        }
    }
}
