//! What the benches use, ported from encoderrr's `decimal_parts_bench`: the files they read
//! (see [`data`]), the needles ([`needle`]), and which benchmarks run ([`Selection`]).

pub mod data;
pub mod needle;

use regex::Regex;

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
