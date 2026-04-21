#!/usr/bin/env python3
"""
Append one row to results/sweep_metrics.tsv from a devel-*.stats file.

Usage (from bin/):
  python collect_run_metrics.py <stats_path> <run_id> <context_window> <flags_csv>

flags_csv: six integers 0/1 in order
  med_patterns,stopwords,external_phrase,affix5_alpha,length_extras,length_next
"""

import argparse
import csv
import datetime as _dt
import os
import re
import sys

import paths

from extract_features import FEATURE_FLAG_KEYS


def parse_stats_f1(stats_path: str) -> tuple:
    """Return (f1_m_avg, f1_drug_n) as percentage strings like '86.8%', or None if missing."""
    f1_drug_n = None
    f1_m_avg = None
    with open(stats_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.rstrip("\n")
            st = s.strip()
            if st.startswith("drug_n") and not st.startswith("drug_n_"):
                pcts = re.findall(r"([\d.]+%)", s)
                if len(pcts) >= 3:
                    f1_drug_n = pcts[-1]
            elif st.startswith("m.avg") and "no class" not in st:
                pcts = re.findall(r"([\d.]+%)", s)
                if len(pcts) >= 3:
                    f1_m_avg = pcts[-1]
    return f1_m_avg, f1_drug_n


def main() -> int:
    ap = argparse.ArgumentParser(description="Append devel stats row to sweep_metrics.tsv")
    ap.add_argument("stats_path", help="Path to devel-CRF.stats (or other model .stats)")
    ap.add_argument("run_id", help="Run identifier")
    ap.add_argument("context_window", type=int)
    ap.add_argument(
        "flags_csv",
        help="Comma-separated 0/1 for: " + ",".join(FEATURE_FLAG_KEYS),
    )
    args = ap.parse_args()

    parts = [p.strip() for p in args.flags_csv.split(",")]
    if len(parts) != len(FEATURE_FLAG_KEYS):
        print(
            f"flags_csv must have {len(FEATURE_FLAG_KEYS)} values, got {len(parts)}",
            file=sys.stderr,
        )
        return 1
    for p in parts:
        if p not in ("0", "1"):
            print(f"Invalid flag value {p!r} (expected 0 or 1)", file=sys.stderr)
            return 1

    if not os.path.isfile(args.stats_path):
        print(f"Stats file not found: {args.stats_path}", file=sys.stderr)
        return 1

    f1_m_avg, f1_drug_n = parse_stats_f1(args.stats_path)
    os.makedirs(paths.RESULTS, exist_ok=True)
    out_tsv = os.path.join(paths.RESULTS, "sweep_metrics.tsv")
    ts = _dt.datetime.now().isoformat(timespec="seconds")

    header = (
        ["timestamp", "run_id", "context_window"]
        + list(FEATURE_FLAG_KEYS)
        + ["f1_m_avg", "f1_drug_n", "stats_path"]
    )
    row = [ts, args.run_id, str(args.context_window)] + parts + [
        f1_m_avg or "",
        f1_drug_n or "",
        os.path.abspath(args.stats_path),
    ]

    write_header = not os.path.isfile(out_tsv)
    with open(out_tsv, "a", newline="", encoding="utf-8") as wf:
        w = csv.writer(wf, delimiter="\t")
        if write_header:
            w.writerow(header)
        w.writerow(row)

    print(f"Appended metrics to {out_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
