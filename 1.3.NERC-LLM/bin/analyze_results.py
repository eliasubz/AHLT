"""
analyze_results.py  —  Collect all experiment results and visualize them.

Usage:
    python3 analyze_results.py [results_dir]

Reads all *.stats files from the results directory, parses metrics,
prints a comparison table, and saves bar charts to results/plots/.

Stats file naming convention:
    FS-{model}-{prompt}-{shots}-{testdata}[-quant].stats
    FT-{model}[-quant]-{testdata}.stats
"""

import os, sys, re
import glob

# ---- optional imports: table + plots ----
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------
# Parse a single .stats file — handles multiple evaluator formats
# ---------------------------------------------------------------
def parse_stats(filepath):
    """
    Returns a dict with keys: precision, recall, f1
    plus per-type entries like drug_f1, group_f1, brand_f1, drug_n_f1.
    Returns None if the file cannot be parsed.
    """
    try:
        with open(filepath) as f:
            text = f.read()
    except Exception:
        return None

    result = {}

    # ---- try format: "precision= 0.1234  recall= 0.1234  F1= 0.1234" ----
    m = re.search(r"precision\s*=\s*([0-9.]+)", text, re.IGNORECASE)
    if m:
        result["precision"] = float(m.group(1))
    m = re.search(r"recall\s*=\s*([0-9.]+)", text, re.IGNORECASE)
    if m:
        result["recall"] = float(m.group(1))
    m = re.search(r"\bF1\s*=\s*([0-9.]+)", text, re.IGNORECASE)
    if m:
        result["f1"] = float(m.group(1))

    # ---- try format: tab/space separated table with header line ----
    #   type    tp    fp    fn    P       R       F1
    #   drug    123   45    67    0.732   0.647   0.687
    #   TOTAL   ...
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 7:
            label = parts[0].lower()
            try:
                p, r, f = float(parts[-3]), float(parts[-2]), float(parts[-1])
                if label in ("drug", "group", "brand", "drug_n"):
                    result[f"{label}_p"]  = p
                    result[f"{label}_r"]  = r
                    result[f"{label}_f1"] = f
                elif label in ("total", "macro", "micro", "overall"):
                    if "precision" not in result: result["precision"] = p
                    if "recall"    not in result: result["recall"]    = r
                    if "f1"        not in result: result["f1"]        = f
            except ValueError:
                pass

    return result if result else None


# ---------------------------------------------------------------
# Parse filename into experiment metadata
# ---------------------------------------------------------------
def parse_filename(fname):
    """
    Parses a stats filename into a metadata dict.
    Handles:
        FS-{model}-{prompt}-{shots}-{testdata}[-quant].stats
        FT-{model}[-quant]-{testdata}.stats
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    meta = {"filename": base, "type": "?", "model": "?",
            "prompt": "-", "shots": "-", "testdata": "?", "quant": False}

    parts = base.split("-")
    if not parts:
        return meta

    meta["type"] = parts[0]  # FS or FT

    if meta["type"] == "FS" and len(parts) >= 5:
        # FS - model - prompt - shots - testdata [- quant]
        meta["model"]    = parts[1]
        meta["prompt"]   = parts[2]
        meta["shots"]    = parts[3]
        meta["testdata"] = parts[4]
        meta["quant"]    = len(parts) > 5 and parts[5] == "quant"

    elif meta["type"] == "FT" and len(parts) >= 3:
        # FT - model [- quant] - testdata
        if parts[-2] == "quant":
            meta["model"]    = parts[1]
            meta["quant"]    = True
            meta["testdata"] = parts[-1]
        else:
            meta["model"]    = parts[1]
            meta["testdata"] = parts[-1]

    return meta


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    # results dir: argument or auto-detect relative to this script
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(os.path.dirname(here), "results")

    stats_files = sorted(glob.glob(os.path.join(results_dir, "*.stats")))

    if not stats_files:
        print(f"No .stats files found in: {results_dir}")
        sys.exit(0)

    rows = []
    for sf in stats_files:
        meta   = parse_filename(sf)
        metrics = parse_stats(sf)
        if metrics is None:
            print(f"  WARNING: could not parse {sf}", flush=True)
            continue
        row = {**meta, **metrics}
        rows.append(row)

    if not rows:
        print("No results could be parsed.")
        sys.exit(0)

    # ---- Print table ----
    col_order = ["type", "model", "prompt", "shots", "testdata", "quant",
                 "precision", "recall", "f1",
                 "drug_f1", "group_f1", "brand_f1", "drug_n_f1"]

    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        # keep only columns that exist
        cols = [c for c in col_order if c in df.columns]
        df = df[cols].sort_values(["testdata", "f1"], ascending=[True, False])

        float_cols = [c for c in df.columns if df[c].dtype == float]
        print("\n=== Experiment Results ===\n")
        print(df.to_string(index=False,
                           float_format=lambda x: f"{x:.4f}"))

        # ---- Save CSV ----
        csv_path = os.path.join(results_dir, "summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

    else:
        # plain text fallback
        header = f"{'type':<4} {'model':<14} {'prompt':<12} {'shots':<6} {'data':<8} {'quant':<6} {'P':>7} {'R':>7} {'F1':>7}"
        print("\n=== Experiment Results ===\n")
        print(header)
        print("-" * len(header))
        for r in sorted(rows, key=lambda x: -x.get("f1", 0)):
            print(f"{r.get('type','?'):<4} {r.get('model','?'):<14} "
                  f"{r.get('prompt','-'):<12} {str(r.get('shots','-')):<6} "
                  f"{r.get('testdata','?'):<8} {str(r.get('quant',False)):<6} "
                  f"{r.get('precision',0):>7.4f} {r.get('recall',0):>7.4f} "
                  f"{r.get('f1',0):>7.4f}")

    # ---- Plots ----
    if HAS_MATPLOTLIB and HAS_PANDAS and len(df) > 0:
        plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        for testdata, group in df.groupby("testdata"):
            if "f1" not in group.columns or group["f1"].isna().all():
                continue

            group = group.copy().sort_values("f1", ascending=False)

            # Build a label that captures the key variables
            def make_label(row):
                parts = [row.get("type","?"), row.get("model","?")]
                if row.get("prompt", "-") != "-":
                    parts.append(row.get("prompt",""))
                if str(row.get("shots", "-")) != "-":
                    parts.append(f"{row.get('shots','')}shot")
                if row.get("quant", False):
                    parts.append("quant")
                return "\n".join(parts)

            labels = [make_label(r) for _, r in group.iterrows()]

            # --- Overview bar chart: P / R / F1 ---
            fig, ax = plt.subplots(figsize=(max(8, len(group)*1.5), 5))
            x = range(len(group))
            width = 0.25
            for i, (metric, color) in enumerate(
                    [("precision","steelblue"), ("recall","darkorange"), ("f1","green")]):
                vals = group[metric].fillna(0).tolist() if metric in group else [0]*len(group)
                ax.bar([xi + i*width for xi in x], vals, width,
                       label=metric.capitalize(), color=color, alpha=0.85)

            ax.set_xticks([xi + width for xi in x])
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
            ax.set_title(f"P / R / F1 — {testdata}")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            path = os.path.join(plots_dir, f"overview_{testdata}.png")
            plt.savefig(path, dpi=120)
            plt.close()
            print(f"Saved: {path}")

            # --- Per-type F1 stacked bar chart ---
            type_cols = [c for c in ["drug_f1","group_f1","brand_f1","drug_n_f1"]
                         if c in group.columns]
            if type_cols:
                fig, ax = plt.subplots(figsize=(max(8, len(group)*1.5), 5))
                colors = ["steelblue","darkorange","green","crimson"]
                bottoms = [0.0] * len(group)
                for col, color in zip(type_cols, colors):
                    vals = group[col].fillna(0).tolist()
                    ax.bar(x, vals, label=col.replace("_f1",""), color=color,
                           alpha=0.8, bottom=bottoms)
                    bottoms = [b+v for b,v in zip(bottoms, vals)]
                ax.set_xticks(list(x))
                ax.set_xticklabels(labels, fontsize=8)
                ax.set_ylabel("F1 (stacked sum — lower is not worse, use for comparison)")
                ax.set_title(f"Per-type F1 — {testdata}")
                ax.legend()
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                path = os.path.join(plots_dir, f"per_type_{testdata}.png")
                plt.savefig(path, dpi=120)
                plt.close()
                print(f"Saved: {path}")

            # --- F1 line plot over #shots (few-shot only) ---
            fs_group = group[group["type"] == "FS"].copy()
            fs_group["shots_int"] = pd.to_numeric(fs_group["shots"], errors="coerce")
            if len(fs_group) > 1 and not fs_group["shots_int"].isna().all():
                for prompt_name, pg in fs_group.groupby("prompt"):
                    pg = pg.sort_values("shots_int")
                    ax_fig, ax2 = plt.subplots(figsize=(7, 4))
                    ax2.plot(pg["shots_int"], pg["f1"], marker="o", label=prompt_name)
                    ax2.set_xlabel("Number of few-shot examples")
                    ax2.set_ylabel("F1")
                    ax2.set_title(f"F1 vs #shots — {testdata} — {prompt_name}")
                    ax2.set_ylim(0, 1.0)
                    ax2.grid(alpha=0.3)
                    ax2.legend()
                    plt.tight_layout()
                    path = os.path.join(plots_dir, f"shots_curve_{testdata}_{prompt_name}.png")
                    plt.savefig(path, dpi=120)
                    plt.close()
                    print(f"Saved: {path}")

    elif not HAS_MATPLOTLIB:
        print("\n(install matplotlib + pandas for charts: pip install matplotlib pandas)")


if __name__ == "__main__":
    main()
