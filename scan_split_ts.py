"""One-shot timestamp scanner for picking a row-level train/valid split.

Why this exists
---------------
The Row-Group-statistics-only estimator in ``dataset.decide_row_level_split_threshold``
assumes a uniform time density inside each Parquet Row Group. On this
dataset that assumption is badly wrong: 80%+ of rows are concentrated in
the very last day, so the estimator over-shoots ``split_ts`` and gives a
near-empty validation set.

This script reads only the ``timestamp`` column (8 bytes per row, ~8 MB
for ~1M rows) of every Parquet file under ``data_dir``, sorts the values
once, and prints the *exact* row-level quantile cut(s) you need.

Usage
-----
    python scan_split_ts.py                      # default: ./data, valid_ratio=0.10
    python scan_split_ts.py --valid_ratio 0.05
    python scan_split_ts.py --data_dir /path/to/data --valid_ratios 0.05,0.10,0.15

The reported ``split_ts`` is the smallest int such that
``count(rows with ts >= split_ts) <= valid_ratio * total_rows``. Pass it
back to ``dataset.py`` (or hardcode it as a constant) and use it as a
hard row-level filter:

    train: ts <  split_ts
    valid: ts >= split_ts

Output is printed to stdout AND written to ``split_ts_report.json`` next
to the data directory for easy machine reuse.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pyarrow.parquet as pq


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout,
    )


def _human_ts(ts: Optional[int]) -> str:
    if ts is None:
        return '<none>'
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return '<out-of-range>'


def scan_timestamps(
    data_dir: str,
    timestamp_col: str = 'timestamp',
) -> np.ndarray:
    """Read the timestamp column from every Parquet file and return a
    sorted int64 array.

    Reads one column at a time (zero-copy where possible) so the memory
    footprint stays at ~8 bytes per row.
    """
    pq_files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    if not pq_files:
        raise FileNotFoundError(f"No .parquet files in {data_dir}")
    logging.info("Found %d Parquet files in %s", len(pq_files), data_dir)

    chunks: List[np.ndarray] = []
    total_files = len(pq_files)
    t0 = time.time()
    for idx, f in enumerate(pq_files, 1):
        pf = pq.ParquetFile(f)
        try:
            tbl = pf.read(columns=[timestamp_col])
        except KeyError as exc:
            raise KeyError(
                f"File {f} has no '{timestamp_col}' column. "
                f"Available: {pf.schema_arrow.names}"
            ) from exc
        arr = tbl.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        # Drop non-positive sentinels (treated as "unknown" everywhere else).
        arr = arr[arr > 0]
        chunks.append(arr)
        if idx % 50 == 0 or idx == total_files:
            logging.info(
                "  scanned %d/%d files (%.1fs elapsed, %d total rows so far)",
                idx, total_files, time.time() - t0,
                sum(c.shape[0] for c in chunks),
            )
    all_ts = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    del chunks
    logging.info(
        "Concatenated %s timestamps; sorting in-place...", f"{all_ts.shape[0]:,}"
    )
    all_ts.sort()
    logging.info("Sort done (%.1fs total).", time.time() - t0)
    return all_ts


def report_split(
    sorted_ts: np.ndarray,
    valid_ratios: List[float],
) -> dict:
    """For each target ``valid_ratio``, compute the exact quantile-based
    ``split_ts`` and the achieved partition counts.

    Definition: ``split_ts`` is the smallest integer such that
    ``count(ts >= split_ts) <= valid_ratio * total``. This is precisely
    ``sorted_ts[ceil((1 - valid_ratio) * total)]`` (with index clamping).
    Choosing the smallest qualifying ``split_ts`` keeps validation as
    close to the target ratio as possible without overshooting.
    """
    n = int(sorted_ts.shape[0])
    if n == 0:
        raise ValueError("No usable timestamps; nothing to split.")

    summary = {
        'total_rows': n,
        'ts_min': int(sorted_ts[0]),
        'ts_min_iso': _human_ts(int(sorted_ts[0])),
        'ts_max': int(sorted_ts[-1]),
        'ts_max_iso': _human_ts(int(sorted_ts[-1])),
        'splits': [],
    }

    print()
    print("=" * 78)
    print(f"Total rows scanned: {n:,}")
    print(f"Global ts range:    {summary['ts_min']} ({summary['ts_min_iso']})  "
          f"->  {summary['ts_max']} ({summary['ts_max_iso']})")
    print(f"Coarse percentiles (UTC):")
    for p in (10, 25, 50, 75, 90, 95, 99):
        idx = max(0, min(n - 1, int(round(p / 100.0 * (n - 1)))))
        ts = int(sorted_ts[idx])
        print(f"  p{p:>2d}: {ts}  ({_human_ts(ts)})")
    print("=" * 78)

    for ratio in valid_ratios:
        # Target: keep ``ratio * n`` rows on the >= side.
        # Pick the cut index = total - target so that everything from that
        # index onward is "valid". split_ts = sorted_ts[cut_idx].
        target_valid = int(round(ratio * n))
        # Ensure we keep at least 1 row on each side when feasible.
        target_valid = max(1, min(n - 1, target_valid))
        cut_idx = n - target_valid

        split_ts = int(sorted_ts[cut_idx])

        # Recount precisely: count rows with ts >= split_ts (handles ties).
        # np.searchsorted with side='left' gives the first index where
        # sorted_ts[i] >= split_ts, so valid_count = n - that_index.
        first_ge = int(np.searchsorted(sorted_ts, split_ts, side='left'))
        valid_count = n - first_ge
        train_count = first_ge
        achieved_ratio = valid_count / n

        rec = {
            'target_valid_ratio': ratio,
            'split_ts': split_ts,
            'split_ts_iso': _human_ts(split_ts),
            'train_rows': train_count,
            'valid_rows': valid_count,
            'achieved_valid_ratio': achieved_ratio,
        }
        summary['splits'].append(rec)

        print()
        print(f"--- target valid_ratio = {ratio:.4f} ---")
        print(f"  split_ts          = {split_ts}      "
              f"({_human_ts(split_ts)})")
        print(f"  train rows (ts <  split_ts) = {train_count:,}")
        print(f"  valid rows (ts >= split_ts) = {valid_count:,}")
        print(f"  achieved valid ratio        = {achieved_ratio:.4f}")
        print(f"  ==> Use this in dataset.py:")
        print(f"      train_ts_filter = (None, {split_ts})")
        print(f"      valid_ts_filter = ({split_ts}, None)")
    print("=" * 78)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing *.parquet files. '
                             'Default: ./data')
    parser.add_argument('--timestamp_col', type=str, default='timestamp',
                        help='Name of the timestamp column. '
                             'Default: timestamp')
    parser.add_argument('--valid_ratio', type=float, default=0.10,
                        help='Single target validation row fraction. '
                             'Ignored if --valid_ratios is given.')
    parser.add_argument('--valid_ratios', type=str, default='',
                        help='Comma-separated list of valid_ratio values to '
                             'evaluate in one pass, e.g. "0.05,0.10,0.15". '
                             'Overrides --valid_ratio.')
    parser.add_argument('--output_json', type=str, default='split_ts_report.json',
                        help='Where to dump a machine-readable summary.')
    args = parser.parse_args()

    _setup_logging()

    if args.valid_ratios.strip():
        try:
            ratios = [float(x.strip()) for x in args.valid_ratios.split(',')
                      if x.strip()]
        except ValueError as exc:
            raise ValueError(
                f"--valid_ratios must be comma-separated floats, got "
                f"{args.valid_ratios!r}"
            ) from exc
    else:
        ratios = [args.valid_ratio]
    for r in ratios:
        if not (0.0 < r < 1.0):
            raise ValueError(f"valid_ratio={r} not in (0, 1)")

    sorted_ts = scan_timestamps(args.data_dir, timestamp_col=args.timestamp_col)
    summary = report_split(sorted_ts, ratios)

    out_path = args.output_json
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=2)
    logging.info("Wrote machine-readable summary to %s", out_path)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
