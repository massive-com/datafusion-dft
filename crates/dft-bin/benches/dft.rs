//! Bench needles through dft, Vortex against Parquet: re-encode a Parquet file of minute bars to
//! Vortex as `dft -o bars.vortex --format-option ohlc=true` does, check that each query in
//! `needles.sql` returns as many rows on both, and time each with `dft --bench`.
//!
//! ```sh
//! cargo bench -p datafusion-dft --features vortex --bench dft -- \
//!     <bars.parquet> [-n ITERATIONS] [-p PARTITIONS,...] [-r READERS,...] [-o DIR] \
//!     [--rewrite] [--format-option KEY=VALUE ...]
//! ```
//!
//! See [`Args`] (or `--help`) for the options. The Vortex file is written again whenever its
//! options differ from the last write's.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, ValueEnum};
use datafusion::arrow::array::AsArray;
use datafusion::arrow::datatypes::Int64Type;
use datafusion_app::config::ExecutionConfig;
use datafusion_app::local::ExecutionContext;
use datafusion_dft::args::DftArgs;
use datafusion_dft::config::AppConfig;
use datafusion_dft::APP_NAME;

type Error = Box<dyn std::error::Error + Send + Sync>;
type Result<T> = std::result::Result<T, Error>;

/// The queries, one per line, each named by the comment above it.
const NEEDLES: &str = include_str!("needles.sql");

/// The table the queries read.
const TABLE: &str = "bars";

/// Each `--format-option` comes after these, and dft keeps an option's last value.
/// Disables coalescing of narrow columns, so a needle reads small segments.
const FORMAT_OPTIONS: [&str; 3] = [
    "ohlc=true",
    "row_block_size=65536",
    "data_block_target_bytes=none",
];

/// Where the Vortex file, the results and the summary go, under the workspace's `target`.
const OUT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/ohlc-demo");

/// How a table reads its file.
#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum Reader {
    Parquet,
    /// Parquet with `pushdown_filters` and `reorder_filters`.
    Pushdown,
    Vortex,
}

impl Reader {
    const ALL: [Reader; 3] = [Reader::Parquet, Reader::Pushdown, Reader::Vortex];

    fn name(self) -> &'static str {
        match self {
            Reader::Parquet => "parquet",
            Reader::Pushdown => "pushdown",
            Reader::Vortex => "vortex",
        }
    }

    /// DataFusion's settings for this reader on `partitions` partitions.
    fn settings(self, partitions: usize) -> Vec<(&'static str, String)> {
        let mut settings = vec![(
            "datafusion.execution.target_partitions",
            partitions.to_string(),
        )];
        if self == Reader::Pushdown {
            for key in [
                "datafusion.execution.parquet.pushdown_filters",
                "datafusion.execution.parquet.reorder_filters",
            ] {
                settings.push((key, "true".to_string()));
            }
        }
        settings
    }

    /// Where this reader's medians on `partitions` partitions are kept, under `out`.
    fn results(self, out: &Path, partitions: usize) -> PathBuf {
        out.join(format!("{}-{partitions}.tsv", self.name()))
    }
}

/// Bench needles through dft, Vortex against Parquet.
#[derive(Parser)]
#[command(name = "cargo bench -p datafusion-dft --features vortex --bench dft --")]
struct Args {
    /// The Parquet file of minute bars.
    source: PathBuf,
    /// Iterations per query.
    #[arg(short = 'n', default_value_t = 10)]
    iterations: usize,
    /// Partition counts to run at [default: 1 and every core].
    #[arg(short = 'p', value_delimiter = ',')]
    partitions: Vec<usize>,
    /// Readers to time; the summary takes the others' times from their last run's results.
    #[arg(short = 'r', value_enum, value_delimiter = ',', default_values_t = Reader::ALL)]
    readers: Vec<Reader>,
    /// Where the Vortex file, the results and the summary go.
    #[arg(short = 'o', default_value = OUT)]
    out: PathBuf,
    /// Write the Vortex file again even if it exists.
    #[arg(long)]
    rewrite: bool,
    /// Passed to dft, over [`FORMAT_OPTIONS`].
    #[arg(long = "format-option", value_name = "KEY=VALUE")]
    format_options: Vec<String>,
    /// What `cargo bench` passes.
    #[arg(long, hide = true)]
    bench: bool,
}

/// A query in `needles.sql`.
struct Query {
    name: String,
    sql: String,
}

/// The queries in `text`, in order, each named by the comment right above it.
fn needles(text: &str) -> Vec<Query> {
    let mut queries = Vec::new();
    let mut name = None;
    for line in text.lines().map(str::trim) {
        if let Some(comment) = line.strip_prefix("--") {
            name = Some(comment.trim());
        } else if !line.is_empty() {
            let sql = line.trim_end_matches(';');
            let name = name.take().unwrap_or(sql).to_string();
            let sql = sql.to_string();
            queries.push(Query { name, sql });
        }
    }
    queries
}

fn main() -> Result<()> {
    let mut args = Args::parse();
    if args.partitions.is_empty() {
        args.partitions = vec![1, std::thread::available_parallelism()?.get()];
    }
    // dft's runtime (see its `main`): one worker, which runs a query on one partition.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()?;
    runtime.block_on(run(args))
}

async fn run(args: Args) -> Result<()> {
    // Absolute paths, for the DDL.
    let source = canonical(&args.source)?;
    std::fs::create_dir_all(&args.out)?;
    let out = canonical(&args.out)?;
    let vortex = out.join("bars.vortex");

    // 1. Re-encode, unless the file was last written with these options. The options go next to
    // the file, so a run with others writes it again in place of timing a stale file.
    let options: Vec<String> = (FORMAT_OPTIONS.iter().map(|option| option.to_string()))
        .chain(args.format_options)
        .collect();
    let options_text = options.join(" ");
    let written_with = out.join("bars.vortex.options");
    let last_options = match std::fs::read_to_string(&written_with) {
        Ok(text) => Some(text),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e.into()),
    };
    let stale = last_options.as_deref() != Some(options_text.as_str());
    if args.rewrite || !vortex.exists() || stale {
        println!("== writing {} with {options_text}", vortex.display());
        let start = Instant::now();
        // No record while the file is being written: a failed write leaves it stale.
        if last_options.is_some() {
            std::fs::remove_file(&written_with)?;
        }
        write_vortex(&source, &vortex, &options).await?;
        std::fs::write(&written_with, &options_text)?;
        println!("{:.1} s", start.elapsed().as_secs_f64());
    }
    for path in [&source, &vortex] {
        let mib = std::fs::metadata(path)?.len() as f64 / 1048576.0;
        println!("{mib:8.1} MiB  {}", path.display());
    }

    let queries = needles(NEEDLES);
    let ddl = |format: &str, path: &Path| {
        let path = path.display();
        format!("CREATE EXTERNAL TABLE {TABLE} STORED AS {format} LOCATION '{path}'")
    };
    let parquet_ddl = ddl("PARQUET", &source);
    let vortex_ddl = ddl("VORTEX", &vortex);

    // 2. Each query's rows on both: the two should agree.
    println!("== counting rows");
    let parquet_rows = count_rows(&parquet_ddl, &queries).await?;
    let vortex_rows = count_rows(&vortex_ddl, &queries).await?;

    // 3. Time every query on each reader.
    for &partitions in &args.partitions {
        for &reader in &args.readers {
            let iterations = args.iterations;
            let name = reader.name();
            println!(
                "== timing {name}, {partitions} partition(s), {iterations} iterations a query"
            );
            let table_ddl = match reader {
                Reader::Parquet | Reader::Pushdown => &parquet_ddl,
                Reader::Vortex => &vortex_ddl,
            };
            let settings = reader.settings(partitions);
            let ctx = context(&settings, partitions > 1, table_ddl).await?;
            let mut results = String::new();
            for query in &queries {
                let stats =
                    (ctx.benchmark_query(&query.sql, Some(iterations), false, None)).await?;
                let ms = stats.total().median.as_secs_f64() * 1e3;
                println!("{:>8}  {}", fmt_ms(Some(ms)), query.name);
                writeln!(results, "{ms}\t{}", query.sql)?;
            }
            if let Some(executor) = ctx.executor() {
                executor.join().await;
            }
            std::fs::write(reader.results(&out, partitions), results)?;
        }
    }

    // 4. The summary.
    let summary = summary(
        &queries,
        &out,
        &args.partitions,
        &parquet_rows,
        &vortex_rows,
    )?;
    print!("{summary}");
    std::fs::write(out.join("summary.md"), summary)?;
    Ok(())
}

fn canonical(path: &Path) -> Result<PathBuf> {
    std::fs::canonicalize(path).map_err(|e| format!("{}: {e}", path.display()).into())
}

/// Write `source` to `vortex` with dft's CLI, as `dft -c "SELECT * FROM '<source>'" -o <vortex>
/// --format-option ...` would, with each of `options`. One partition reads the Parquet file in
/// order, so the Vortex file keeps sort order and zone maps stay packed.
async fn write_vortex(source: &Path, vortex: &Path, options: &[String]) -> Result<()> {
    let mut argv = vec![
        APP_NAME.to_string(),
        "-c".to_string(),
        format!("SELECT * FROM '{}'", source.display()),
        "-o".to_string(),
        vortex.display().to_string(),
    ];
    for option in options {
        argv.extend(["--format-option".to_string(), option.clone()]);
    }
    let config = app_config(&Reader::Parquet.settings(1), false);
    datafusion_dft::cli::try_run(DftArgs::try_parse_from(argv)?, config).await?;
    // dft writes nothing for a query whose first batch fails.
    if !vortex.exists() {
        return Err(format!("dft wrote no {}", vortex.display()).into());
    }
    Ok(())
}

fn app_config(settings: &[(&str, String)], dedicated: bool) -> AppConfig {
    let datafusion = settings.iter().map(|(k, v)| (k.to_string(), v.clone()));
    let execution = ExecutionConfig {
        ddl_path: None,
        datafusion: Some(datafusion.collect()),
        dedicated_executor_enabled: dedicated,
        ..ExecutionConfig::default()
    };
    // The CLI merges `cli.execution` over `shared`
    let mut config = AppConfig {
        shared: execution.clone(),
        ..AppConfig::default()
    };
    config.cli.execution = execution;
    config
}

async fn context(
    settings: &[(&str, String)],
    dedicated: bool,
    ddl: &str,
) -> Result<ExecutionContext> {
    let config = app_config(settings, dedicated);
    let ctx = datafusion_dft::cli::execution_context(&config).await?;
    ctx.execute_sql_and_discard_results(ddl).await?;
    Ok(ctx)
}

/// Each query's row count on [`TABLE`] made by `ddl`, with dft's defaults.
async fn count_rows(ddl: &str, queries: &[Query]) -> Result<Vec<i64>> {
    let ctx = context(&[], false, ddl).await?;
    let mut rows = Vec::new();
    for query in queries {
        let sql = format!("SELECT count(*) FROM ({})", query.sql);
        let batches = ctx.session_ctx().sql(&sql).await?.collect().await?;
        let batch = batches.first().ok_or(format!("no count: {sql}"))?;
        rows.push(batch.column(0).as_primitive::<Int64Type>().value(0));
    }
    Ok(rows)
}

/// The medians of `reader` on `partitions` kept in `out`, by query.
fn load(out: &Path, reader: Reader, partitions: usize) -> Result<HashMap<String, f64>> {
    let text = match std::fs::read_to_string(reader.results(out, partitions)) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(HashMap::new()),
        Err(e) => return Err(e.into()),
    };
    let mut medians = HashMap::new();
    for line in text.lines() {
        let (ms, sql) = line.split_once('\t').ok_or(format!("bad line: {line}"))?;
        medians.insert(sql.to_string(), ms.parse()?);
    }
    Ok(medians)
}

/// Build a Markdown summary: at each number of partitions, each query's median on each reader;
/// and whether the formats agree on rows.
fn summary(
    queries: &[Query],
    out: &Path,
    partitions: &[usize],
    parquet_rows: &[i64],
    vortex_rows: &[i64],
) -> Result<String> {
    let mut s = String::new();
    for &p in partitions {
        let [parquet, pushdown, vortex] = Reader::ALL.map(|reader| load(out, reader, p));
        let (parquet, pushdown, vortex) = (parquet?, pushdown?, vortex?);
        let plural = if p == 1 { "" } else { "s" };
        writeln!(s, "\n### {p} partition{plural} (median ms)\n")?;
        writeln!(
            s,
            "| query | rows | Parquet | pushdown | Vortex | ÷ Parquet | ÷ pushdown |"
        )?;
        writeln!(s, "|---|---:|---:|---:|---:|---:|---:|")?;
        // Vortex's times over Parquet's and over pushdown's, by query.
        let (mut over_parquets, mut over_pushdowns) = (Vec::new(), Vec::new());
        for (query, rows) in queries.iter().zip(vortex_rows) {
            let [parquet, pushdown, vortex] =
                [&parquet, &pushdown, &vortex].map(|times| times.get(&query.sql).copied());
            let ratio = |base: Option<f64>| vortex.zip(base).map(|(vortex, base)| vortex / base);
            let [over_parquet, over_pushdown] = [parquet, pushdown].map(ratio);
            over_parquets.extend(over_parquet);
            over_pushdowns.extend(over_pushdown);
            let [over_parquet, over_pushdown] = [over_parquet, over_pushdown].map(fmt_ratio);
            let [parquet, pushdown, vortex] = [parquet, pushdown, vortex].map(fmt_ms);
            let name = &query.name;
            writeln!(
                s,
                "| {name} | {rows} | {parquet} | {pushdown} | {vortex} | {over_parquet} | {over_pushdown} |"
            )?;
        }
        for (base, ratios) in [("Parquet", over_parquets), ("pushdown", over_pushdowns)] {
            if ratios.is_empty() {
                continue;
            }
            let mean_log = ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len() as f64;
            let geomean = mean_log.exp();
            let n = ratios.len();
            writeln!(
                s,
                "\nGeometric mean ÷ {base}: {geomean:.2} over {n} queries."
            )?;
        }
    }
    let rows = parquet_rows.iter().zip(vortex_rows);
    let differ: Vec<&str> = (queries.iter().zip(rows))
        .filter(|(_, (parquet, vortex))| parquet != vortex)
        .map(|(query, _)| query.name.as_str())
        .collect();
    if differ.is_empty() {
        writeln!(
            s,
            "\nParquet and Vortex return the same number of rows on every query."
        )?;
    } else {
        let differ = differ.join(", ");
        writeln!(
            s,
            "\nRow counts differ between Parquet and Vortex on: {differ}"
        )?;
    }
    Ok(s)
}

/// Vortex's time over another reader's, to two decimals, in bold where Vortex is no faster.
fn fmt_ratio(ratio: Option<f64>) -> String {
    match ratio {
        None => String::new(),
        Some(ratio) if ratio >= 1.0 => format!("**{ratio:.2}**"),
        Some(ratio) => format!("{ratio:.2}"),
    }
}

/// A time in ms to three significant digits below 100 ms, and to the ms above.
fn fmt_ms(ms: Option<f64>) -> String {
    let Some(ms) = ms else {
        return String::new();
    };
    if ms >= 100.0 {
        return format!("{ms:.0}");
    }
    if ms <= 0.0 {
        return "0".to_string();
    }
    let decimals = (2 - ms.log10().floor() as i32).max(0) as usize;
    let text = format!("{ms:.decimals$}");
    let text = if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.')
    } else {
        &text
    };
    text.to_string()
}
