#!/usr/bin/env python3
"""Append one experiment row to ``results/EXPERIMENT_LOG.tsv`` (creates header if missing)."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys

import paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", required=True, help="fewshot | finetune | other label")
    p.add_argument("--model", default="", help="model id or path token")
    p.add_argument("--prompts", default="", help="prompt JSON path")
    p.add_argument("--split", default="", help="train | devel | test")
    p.add_argument("--k", default="", help="few-shot k (if applicable)")
    p.add_argument("--quant", default="", help="quant | fp16 | ollama | empty")
    p.add_argument("--weights", default="", help="FT adapter basename (if applicable)")
    p.add_argument("--score", default="", help="primary metric from evaluator, if known")
    p.add_argument("--outfile", default="", help="prediction .out path written by pipeline")
    p.add_argument("--notes", default="", help="free-text notes")
    args = p.parse_args()

    os.makedirs(paths.RESULTS, exist_ok=True)
    log_path = os.path.join(paths.RESULTS, "EXPERIMENT_LOG.tsv")
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    fieldnames = [
        "timestamp_utc",
        "strategy",
        "model",
        "prompts",
        "split",
        "k",
        "quant",
        "weights",
        "score",
        "outfile",
        "notes",
    ]
    row = {
        "timestamp_utc": ts,
        "strategy": args.strategy,
        "model": args.model,
        "prompts": args.prompts,
        "split": args.split,
        "k": args.k,
        "quant": args.quant,
        "weights": args.weights,
        "score": args.score,
        "outfile": args.outfile,
        "notes": args.notes,
    }

    new_file = not os.path.isfile(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if new_file:
            w.writeheader()
        w.writerow(row)

    print(log_path, file=sys.stderr)


if __name__ == "__main__":
    main()
