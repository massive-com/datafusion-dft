#!/usr/bin/env bash
# The OHLC demo: re-encode a Parquet file of minute bars to Vortex with dft
# (`--format-option ohlc=true`), then time the queries in needles.sql on both with `dft --bench`,
# and print each query's median against Parquet's.
#
#   bench/ohlc/demo.sh <bars.parquet> [-n ITERATIONS] [-p "PARTITIONS ..."] [-r "READERS ..."]
#                      [-o DIR] [--rewrite]
#
#   -n          iterations per query (default 10)
#   -p          partition counts to run at (default: 1 and every core)
#   -r          readers to time (default "parquet pushdown vortex"); the summary takes the others'
#               times from their last run's logs
#   -o          where the Vortex file, configs and logs go (default: target/ohlc-demo)
#   --rewrite   write the Vortex file again even if it exists
#
# Each query runs against Parquet, Parquet with filter pushdown, and the Vortex file. Set DFT to
# a dft binary built with `--features vortex` to skip the build, and VORTEX_OPTIONS to write the
# Vortex file with other `--format-option`s.
#
# One partition runs on dft's own runtime, one thread; more run on its dedicated executor, a
# thread a core.
set -euo pipefail

# The bench's file: a row block, and zones, every 65,536 rows (the source's row group), and no
# coalescing of a narrow column's row blocks into bigger chunks, so a needle reads small segments.
VORTEX_OPTIONS=${VORTEX_OPTIONS:-ohlc=true row_block_size=65536 data_block_target_bytes=none}

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
source=""
iterations=10
partitions="1 $(getconf _NPROCESSORS_ONLN)"
readers="parquet pushdown vortex"
out="$root/target/ohlc-demo"
rewrite=0
while (($#)); do
  case $1 in
    -n) iterations=$2; shift 2 ;;
    -p) partitions=$2; shift 2 ;;
    -r) readers=$2; shift 2 ;;
    -o) out=$2; shift 2 ;;
    --rewrite) rewrite=1; shift ;;
    -h | --help) sed -n '2,21p' "$0"; exit 0 ;;
    *) source=$1; shift ;;
  esac
done
if [[ -z $source || ! -f $source ]]; then
  echo "usage: $0 <bars.parquet> [-n ITERATIONS] [-p \"PARTITIONS ...\"] [-r \"READERS ...\"] [-o DIR] [--rewrite]" >&2
  exit 2
fi
# Absolute paths, for the DDL.
source="$(cd "$(dirname "$source")" && pwd)/$(basename "$source")"
mkdir -p "$out"
out=$(cd "$out" && pwd)
vortex="$out/bars.vortex"

if [[ -z ${DFT:-} ]]; then
  echo "== building dft (release, --features vortex)"
  cargo build --release --features vortex -p datafusion-dft --manifest-path "$root/Cargo.toml"
  DFT="$root/target/release/dft"
fi
# Keep a Mac awake while timing.
awake=()
if command -v caffeinate >/dev/null; then awake=(caffeinate -dims); fi

# `config NAME DDL [SETTING=VALUE ...]` writes NAME.toml, a dft config whose DDL file (run with
# --run-ddl) is DDL, with those settings: DataFusion's (`datafusion.*`) and dft's own execution
# settings (any other key, its value written as is).
config() {
  local name=$1 ddl=$2 setting
  shift 2
  printf '%s;\n' "$ddl" >"$out/$name.sql"
  {
    echo "[cli.execution]"
    echo "ddl_path = \"$out/$name.sql\""
    for setting in "$@"; do
      case $setting in datafusion.*) ;; *) echo "${setting%%=*} = ${setting#*=}" ;; esac
    done
    echo "[cli.execution.datafusion]"
    for setting in "$@"; do
      case $setting in datafusion.*) echo "\"${setting%%=*}\" = \"${setting#*=}\"" ;; esac
    done
  } >"$out/$name.toml"
}

# 1. Re-encode. One partition reads the Parquet file in order, so the Vortex file keeps its rows'
# order (sorted by ticker, then time) and its zone maps stay tight.
if ((rewrite)) || [[ ! -f $vortex ]]; then
  echo "== writing $vortex"
  config write "" "datafusion.execution.target_partitions=1"
  options=()
  for option in $VORTEX_OPTIONS; do options+=(--format-option "$option"); done
  time "$DFT" --config "$out/write.toml" -c "SELECT * FROM '$source'" -o "$vortex" "${options[@]}"
fi
ls -l "$source" "$vortex" | awk '{printf "%8.1f MiB  %s\n", $5 / 1048576, $NF}'

# The queries, one per line of needles.sql.
queries=()
while IFS= read -r line; do
  case $line in '' | --*) continue ;; esac
  queries+=("${line%;}")
done <"$here/needles.sql"

parquet_ddl="CREATE EXTERNAL TABLE bars STORED AS PARQUET LOCATION '$source'"
vortex_ddl="CREATE EXTERNAL TABLE bars STORED AS VORTEX LOCATION '$vortex'"

# 2. Each query's rows on both, as one query of counts: the two should agree.
counts=""
i=0
for query in "${queries[@]}"; do
  if ((i > 0)); then counts+=" UNION ALL "; fi
  counts+="SELECT $i AS query, count(*) AS rows FROM ($query)"
  i=$((i + 1))
done
for format in parquet vortex; do
  ddl=$parquet_ddl
  if [[ $format == vortex ]]; then ddl=$vortex_ddl; fi
  config "rows-$format" "$ddl"
  echo "== counting rows ($format)"
  "$DFT" --config "$out/rows-$format.toml" --run-ddl -j -c "$counts ORDER BY query" \
    >"$out/rows-$format.json"
done

# 3. Time every query on each reader, all queries in one dft process per reader.
commands=()
for query in "${queries[@]}"; do commands+=(-c "$query"); done
for p in $partitions; do
  for reader in $readers; do
    settings=("datafusion.execution.target_partitions=$p")
    if ((p > 1)); then settings+=(dedicated_executor_enabled=true); fi
    case $reader in
      parquet) ddl=$parquet_ddl ;;
      pushdown)
        ddl=$parquet_ddl
        settings+=(datafusion.execution.parquet.pushdown_filters=true
          datafusion.execution.parquet.reorder_filters=true)
        ;;
      vortex) ddl=$vortex_ddl ;;
    esac
    config "$reader-$p" "$ddl" "${settings[@]}"
    echo "== timing $reader, $p partition(s), $iterations iterations a query"
    ${awake[@]+"${awake[@]}"} "$DFT" --config "$out/$reader-$p.toml" --run-ddl --bench \
      -n "$iterations" "${commands[@]}" >"$out/$reader-$p.log"
  done
done

# 4. The summary.
python3 "$here/summarize.py" "$here/needles.sql" "$out" $partitions | tee "$out/summary.md"
