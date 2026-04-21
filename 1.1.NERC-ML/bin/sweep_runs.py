#!/usr/bin/env python3
"""
Run ablation grid:

1. Base (all six optional flag groups off).
2. Singletons (exactly one group on at a time).
3. Cumulative ladder: same fixed order, adding one group per run until all are on
   (cumulative_step06 == full extractor; same as omitting feature_flags in run.py).

Execute from the bin/ directory (same as run.py):
  python sweep_runs.py

Resume (default): skips runs that already have a complete ``devel-CRF.stats``;
  if stats exist but the row is missing from ``sweep_metrics.tsv``, only runs collect.
  ``python sweep_runs.py --force`` re-runs everything from scratch.

Uses context_window=1 for all runs (edit CONTEXT_WINDOW below to change).
Skips logically redundant configs: length_next with context_window==0.
"""

import argparse
import csv
import os
import subprocess
import sys

import paths
from extract_features import (
    FEATURE_FLAG_KEYS,
    FeatureFlags,
    feature_flags_to_csv_tuple,
    parse_feature_flags_param,
)

BIN_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_PY = os.path.join(BIN_DIR, "run.py")
COLLECT_PY = os.path.join(BIN_DIR, "collect_run_metrics.py")

CONTEXT_WINDOW = 1


def _run_cmd(argv: list) -> None:
    print("RUN:", " ".join(argv), flush=True)
    subprocess.run(argv, cwd=BIN_DIR, check=True)


def _flags_csv(flags: FeatureFlags) -> str:
    return ",".join(str(x) for x in feature_flags_to_csv_tuple(flags))


def _collect(stats_path: str, run_id: str, cw: int, flags: FeatureFlags) -> None:
    _run_cmd(
        [
            sys.executable,
            COLLECT_PY,
            stats_path,
            run_id,
            str(cw),
            _flags_csv(flags),
        ]
    )


def _devel_stats_complete(stats_path: str) -> bool:
    """True if evaluate already wrote a usable devel-CRF.stats."""
    if not os.path.isfile(stats_path) or os.path.getsize(stats_path) < 50:
        return False
    try:
        with open(stats_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return False
    return "m.avg" in text and "drug_n" in text


def _sweep_tsv_has_run_id(tsv_path: str, run_id: str) -> bool:
    if not os.path.isfile(tsv_path):
        return False
    try:
        with open(tsv_path, encoding="utf-8", newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            if not r.fieldnames or "run_id" not in r.fieldnames:
                return False
            for row in r:
                if row.get("run_id") == run_id:
                    return True
    except OSError:
        return False
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Ablation / cumulative CRF sweep")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-run extract/train/predict even if devel-CRF.stats already exists",
    )
    args = ap.parse_args()
    force = args.force

    cw = CONTEXT_WINDOW
    field_names = list(FeatureFlags.__dataclass_fields__.keys())

    runs: list[tuple[str, str | None, FeatureFlags]] = []
    # (run_id, feature_flags argv value: None=omit for all-on, ""=empty for all-off, else key name)
    runs.append(("ablate_base__cw1", "", parse_feature_flags_param("")))

    for key in field_names:
        kwargs = {fn: (fn == key) for fn in field_names}
        fl = FeatureFlags(**kwargs)
        runs.append((f"ablate_only_{key}__cw1", key, fl))

    # Cumulative: step01 = first flag only, step02 = first two, …, step06 = all (full model).
    for k in range(1, len(FEATURE_FLAG_KEYS) + 1):
        enabled = FEATURE_FLAG_KEYS[:k]
        ff_str = ",".join(enabled)
        fl = parse_feature_flags_param(ff_str)
        run_id = f"cumulative_step{k:02d}__cw{cw}"
        runs.append((run_id, ff_str, fl))

    metrics_tsv = os.path.join(paths.RESULTS, "sweep_metrics.tsv")

    for run_id, ff_arg, flags in runs:
        if cw == 0 and flags.length_next:
            print(f"SKIP {run_id}: length_next with context_window=0", flush=True)
            continue

        stats_path = os.path.join(paths.RESULTS, "runs", run_id, "devel-CRF.stats")

        if not force and _devel_stats_complete(stats_path):
            if _sweep_tsv_has_run_id(metrics_tsv, run_id):
                print(f"SKIP (done) {run_id}", flush=True)
                continue
            print(f"RESUME collect only {run_id}", flush=True)
            _collect(stats_path, run_id, cw, flags)
            continue

        cmd = [
            sys.executable,
            RUN_PY,
            "extract",
            "train",
            "predict",
            "CRF",
            f"context_window={cw}",
            f"run_id={run_id}",
        ]
        if ff_arg is not None:
            cmd.append(f"feature_flags={ff_arg}")
        _run_cmd(cmd)

        _collect(stats_path, run_id, cw, flags)

    print("Sweep finished. See results/sweep_metrics.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
