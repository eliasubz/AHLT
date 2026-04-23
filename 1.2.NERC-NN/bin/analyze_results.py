"""
analyze_results.py  —  Collect all experiment results and visualize them.

Usage:
    python3 analyze_results.py [results_dir]

Reads all *.stats files from the results directory, parses metrics and
model hyperparameters from file names, prints a ranked comparison table,
and saves bar/line charts to results/plots/.

Stats file naming convention (set by run.py):
    {devel|test}-model_h{H}_l{L}_d{D}_e{E}_f{F}_s{S}.stats
"""

import os, sys, re, glob

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
# Parse a .stats file — tolerates multiple evaluator formats
# ---------------------------------------------------------------
def parse_stats(filepath):
    try:
        text = open(filepath).read()
    except Exception:
        return None

    result = {}

    m = re.search(r"precision\s*=?\s*([0-9.]+)", text, re.IGNORECASE)
    if m:
        result["precision"] = float(m.group(1))
    m = re.search(r"recall\s*=?\s*([0-9.]+)", text, re.IGNORECASE)
    if m:
        result["recall"] = float(m.group(1))
    m = re.search(r"\bF1\s*=?\s*([0-9.]+)", text, re.IGNORECASE)
    if m:
        result["f1"] = float(m.group(1))

    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue

        # Evaluator prints percentages like "79.1%"; strip '%' if present.
        def _to_float(x):
            x = x.strip()
            if x.endswith("%"):
                x = x[:-1]
                return float(x) / 100.0
            return float(x)

        # Handle special summary rows first (they contain spaces in the label)
        # Examples:
        #   M.avg            -  -  -  -  -  65.4% 55.7% 59.0%
        #   m.avg(no class) 2532 291 593 2823 3125 89.7% 81.0% 85.1%
        if line.startswith("M.avg"):
            try:
                p, r, f = (
                    _to_float(parts[-3]),
                    _to_float(parts[-2]),
                    _to_float(parts[-1]),
                )
                result["macro_precision"] = p
                result["macro_recall"] = r
                result["macro_f1"] = f
            except ValueError:
                pass
            continue
        if line.startswith("m.avg(no"):
            try:
                p, r, f = (
                    _to_float(parts[-3]),
                    _to_float(parts[-2]),
                    _to_float(parts[-1]),
                )
                result["micro_noclass_precision"] = p
                result["micro_noclass_recall"] = r
                result["micro_noclass_f1"] = f
            except ValueError:
                pass
            continue
        if line.startswith("m.avg"):
            try:
                p, r, f = (
                    _to_float(parts[-3]),
                    _to_float(parts[-2]),
                    _to_float(parts[-1]),
                )
                result["micro_precision"] = p
                result["micro_recall"] = r
                result["micro_f1"] = f
            except ValueError:
                pass
            continue

        label = parts[0].lower()
        try:
            p, r, f = _to_float(parts[-3]), _to_float(parts[-2]), _to_float(parts[-1])
            if label in ("drug", "group", "brand", "drug_n"):
                result[f"{label}_f1"] = f
        except ValueError:
            pass

    # Pick a default overall score for ranking/reporting.
    # Prefer macro-F1 (assignment asks for per-type + macro), fallback to micro-F1.
    if "macro_f1" in result:
        result.setdefault("precision", result.get("macro_precision"))
        result.setdefault("recall", result.get("macro_recall"))
        result.setdefault("f1", result.get("macro_f1"))
    elif "micro_f1" in result:
        result.setdefault("precision", result.get("micro_precision"))
        result.setdefault("recall", result.get("micro_recall"))
        result.setdefault("f1", result.get("micro_f1"))

    return result if result else None


# ---------------------------------------------------------------
# Reverse the FEAT_SHORT map so we can decode short codes
# ---------------------------------------------------------------
FEAT_DECODE = {
    "Or": "ortho",
    "EF": "ext_full",
    "EP": "ext_part",
    "DM": "drug_morph",
    "Gk": "greek",
    "Ln": "length",
    "Sp": "spacy",
    "Sh": "shape",
    "BL": "biolex",
}


def decode_feat_str(s):
    """Turn 'OrEFEP' → 'ortho+ext_full+ext_part'."""
    decoded = []
    i = 0
    while i < len(s):
        matched = False
        for code in sorted(FEAT_DECODE, key=len, reverse=True):
            if s[i:].startswith(code):
                decoded.append(FEAT_DECODE[code])
                i += len(code)
                matched = True
                break
        if not matched:
            decoded.append(s[i])
            i += 1
    return "+".join(decoded)


# ---------------------------------------------------------------
# Parse filename → metadata dict
# ---------------------------------------------------------------
def parse_filename(fname):
    base = os.path.splitext(os.path.basename(fname))[0]
    meta = {
        "filename": base,
        "split": "?",
        "hidden": "?",
        "layers": "?",
        "dropout": "?",
        "emb": "?",
        "features": "?",
        "suf_len": "?",
    }

    # split is "devel" or "test" (prefix before first dash)
    dash = base.find("-")
    if dash > 0:
        meta["split"] = base[:dash]
        rest = base[dash + 1 :]
    else:
        rest = base

    m = re.search(r"_h(\d+)", rest)
    meta["hidden"] = int(m.group(1)) if m else "?"
    m = re.search(r"_l(\d+)", rest)
    meta["layers"] = int(m.group(1)) if m else "?"
    m = re.search(r"_d([0-9.]+)", rest)
    meta["dropout"] = float(m.group(1)) if m else "?"
    m = re.search(r"_e([\d-]+)", rest)
    meta["emb"] = m.group(1) if m else "?"
    m = re.search(r"_f([A-Za-z]+)", rest)
    if m:
        meta["features"] = decode_feat_str(m.group(1))
    m = re.search(r"_s(\d+)", rest)
    meta["suf_len"] = int(m.group(1)) if m else "?"

    return meta


def _df_to_markdown_fallback(df, floatfmt=".4f"):
    """Render a simple GitHub-flavored markdown table without tabulate."""

    def fmt(v):
        if v is None:
            return ""
        # pandas may pass numpy scalars
        try:
            if isinstance(v, (float, int)):
                if isinstance(v, float):
                    return format(v, floatfmt)
                return str(v)
        except Exception:
            pass
        return str(v)

    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
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
        meta = parse_filename(sf)
        metrics = parse_stats(sf)
        if metrics is None:
            print(f"  WARNING: could not parse {sf}")
            continue
        rows.append({**meta, **metrics})

    if not rows:
        print("No results could be parsed.")
        sys.exit(0)

    col_order = [
        "split",
        "hidden",
        "layers",
        "dropout",
        "emb",
        "features",
        "suf_len",
        "precision",
        "recall",
        "f1",
        "drug_f1",
        "group_f1",
        "brand_f1",
        "drug_n_f1",
    ]

    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        cols = [c for c in col_order if c in df.columns]
        df = df[cols].sort_values(["split", "f1"], ascending=[True, False])

        print("\n=== NN-NERC Experiment Results ===\n")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        csv_path = os.path.join(results_dir, "summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

        # also dump a markdown table for copy-paste into REPORT.md
        md_path = os.path.join(results_dir, "summary_table.md")
        with open(md_path, "w") as f:
            f.write("## Latest Results\n\n")
            try:
                f.write(df.to_markdown(index=False, floatfmt=".4f"))
            except Exception:
                f.write(_df_to_markdown_fallback(df, floatfmt=".4f"))
            f.write("\n")
        print(f"Saved: {md_path}")

    else:
        header = (
            f"{'split':<6} {'h':>4} {'l':>2} {'d':>5} "
            f"{'features':<35} {'suf':>3} {'P':>7} {'R':>7} {'F1':>7}"
        )
        print("\n=== NN-NERC Experiment Results ===\n")
        print(header)
        print("-" * len(header))
        for r in sorted(rows, key=lambda x: -x.get("f1", 0)):
            print(
                f"{r.get('split','?'):<6} {str(r.get('hidden','?')):>4} "
                f"{str(r.get('layers','?')):>2} {str(r.get('dropout','?')):>5} "
                f"{r.get('features','?'):<35} {str(r.get('suf_len','?')):>3} "
                f"{r.get('precision',0):>7.4f} {r.get('recall',0):>7.4f} "
                f"{r.get('f1',0):>7.4f}"
            )

    # ---- Plots ----
    if not (HAS_MATPLOTLIB and HAS_PANDAS and len(df) > 0):
        if not HAS_MATPLOTLIB:
            print(
                "\n(install matplotlib+pandas for plots: pip install matplotlib pandas)"
            )
        return

    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for split, group in df.groupby("split"):
        if "f1" not in group.columns or group["f1"].isna().all():
            continue
        group = group.sort_values("f1", ascending=False).head(20)

        def label(row):
            parts = [
                f"h{row.get('hidden','?')}",
                f"l{row.get('layers','?')}",
                f"d{row.get('dropout','?')}",
            ]
            feats = str(row.get("features", ""))
            if feats:
                parts.append(feats.replace("+", "\n+"))
            return "\n".join(parts)

        labels = [label(r) for _, r in group.iterrows()]
        x = range(len(group))

        # --- P / R / F1 overview ---
        fig, ax = plt.subplots(figsize=(max(8, len(group) * 1.5), 5))
        for i, (metric, color) in enumerate(
            [("precision", "steelblue"), ("recall", "darkorange"), ("f1", "green")]
        ):
            vals = (
                group[metric].fillna(0).tolist()
                if metric in group
                else [0] * len(group)
            )
            ax.bar(
                [xi + i * 0.25 for xi in x],
                vals,
                0.25,
                label=metric.capitalize(),
                color=color,
                alpha=0.85,
            )
        ax.set_xticks([xi + 0.25 for xi in x])
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"P / R / F1 — {split} set (top 20)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(plots_dir, f"overview_{split}.png")
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"Saved: {path}")

        # --- F1 vs hidden_size (num_layers as hue) ---
        if "hidden" in group.columns and "layers" in group.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            for nl, sg in group.groupby("layers"):
                sg = sg.sort_values("hidden")
                if sg["f1"].notna().any():
                    ax.plot(sg["hidden"], sg["f1"], marker="o", label=f"{nl} layer(s)")
            ax.set_xlabel("Hidden size")
            ax.set_ylabel("F1")
            ax.set_title(f"F1 vs hidden size — {split}")
            ax.set_ylim(0, 1.0)
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            path = os.path.join(plots_dir, f"arch_curve_{split}.png")
            plt.savefig(path, dpi=120)
            plt.close()
            print(f"Saved: {path}")

        # --- per-type F1 ---
        type_cols = [
            c
            for c in ["drug_f1", "group_f1", "brand_f1", "drug_n_f1"]
            if c in group.columns
        ]
        if type_cols:
            fig, ax = plt.subplots(figsize=(max(8, len(group) * 1.5), 5))
            colors = ["steelblue", "darkorange", "green", "crimson"]
            bottoms = [0.0] * len(group)
            for col, color in zip(type_cols, colors):
                vals = group[col].fillna(0).tolist()
                ax.bar(
                    x,
                    vals,
                    label=col.replace("_f1", ""),
                    color=color,
                    alpha=0.8,
                    bottom=bottoms,
                )
                bottoms = [b + v for b, v in zip(bottoms, vals)]
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylabel("F1 (stacked)")
            ax.set_title(f"Per-type F1 — {split}")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            path = os.path.join(plots_dir, f"per_type_{split}.png")
            plt.savefig(path, dpi=120)
            plt.close()
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
