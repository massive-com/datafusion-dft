"""The OHLC demo's summary (see demo.sh): each query's median time on Parquet, Parquet with
pushdown and Vortex, from the logs `dft --bench` printed, with the rows each format returned.

Usage: summarize.py <needles.sql> <out dir> <partitions>..."""
import json
import math
import re
import sys

UNITS = {'ns': 1e-6, 'µs': 1e-3, 'us': 1e-3, 'ms': 1.0, 's': 1000.0}
DURATION = re.compile(r'^([\d.]+)(ns|µs|us|ms|s)$')


def needles(path):
    """(name, query) for each query in needles.sql, in order."""
    out, name = [], None
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if line.startswith('-- ') and ': ' in line:
            name = line[3:]
        elif line and not line.startswith('--'):
            out.append((name or line, line.rstrip(';')))
            name = None
    return out


def medians(path):
    """Each query's median total time in ms, from what `dft --bench` printed."""
    lines = open(path, encoding='utf-8').read().splitlines()
    out, query, section = {}, None, None
    for i, line in enumerate(lines):
        if line.startswith('Benchmark Stats ('):
            query = next(l for l in lines[i + 1:] if l and not l.startswith('---'))
        elif line in ('Logical Planning', 'Physical Planning', 'Execution', 'Total'):
            section = line
        elif section == 'Total' and line.startswith('Median: ') and query is not None:
            m = DURATION.match(line[len('Median: '):].strip())
            out[query] = float(m.group(1)) * UNITS[m.group(2)]
            query = section = None
    return out


def rows(path):
    """Each query's row count, by its index in needles.sql."""
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if line.startswith('{'):
            row = json.loads(line)
            out[int(row['query'])] = int(row['rows'])
    return out


def fmt(ms):
    if ms is None:
        return ''
    return f'{ms:.3g}' if ms < 100 else f'{ms:.0f}'


def main():
    queries = needles(sys.argv[1])
    out, partitions = sys.argv[2], sys.argv[3:]
    parquet_rows, vortex_rows = rows(f'{out}/rows-parquet.json'), rows(f'{out}/rows-vortex.json')
    differ = [name for i, (name, _) in enumerate(queries)
              if parquet_rows.get(i) != vortex_rows.get(i)]
    for p in partitions:
        times = {reader: medians(f'{out}/{reader}-{p}.log')
                 for reader in ('parquet', 'pushdown', 'vortex')}
        print(f'\n### {p} partition{"s" if p != "1" else ""} (median ms)\n')
        print('| query | rows | Parquet | pushdown | Vortex | ÷ best Parquet |')
        print('|---|---:|---:|---:|---:|---:|')
        ratios = []
        for i, (name, query) in enumerate(queries):
            parquet, pushdown, vortex = (times[r].get(query) for r in ('parquet', 'pushdown', 'vortex'))
            best = min(t for t in (parquet, pushdown) if t is not None) if parquet or pushdown else None
            ratio = vortex / best if vortex is not None and best else None
            if ratio is not None:
                ratios.append(ratio)
            cell = '' if ratio is None else (f'**{ratio:.2f}**' if ratio >= 1 else f'{ratio:.2f}')
            count = vortex_rows.get(i, '')
            print(f'| {name} | {count} | {fmt(parquet)} | {fmt(pushdown)} | {fmt(vortex)} | {cell} |')
        if ratios:
            geomean = math.exp(sum(map(math.log, ratios)) / len(ratios))
            print(f'\nGeometric mean ÷ best Parquet: {geomean:.2f} over {len(ratios)} queries.')
    if differ:
        print(f'\nRow counts differ between Parquet and Vortex on: {", ".join(differ)}')
    else:
        print('\nParquet and Vortex return the same number of rows on every query.')


if __name__ == '__main__':
    main()
