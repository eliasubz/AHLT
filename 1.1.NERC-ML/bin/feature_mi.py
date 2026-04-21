#!/usr/bin/env python3

import argparse
import os
from collections import defaultdict

import numpy as np
import scipy.sparse
from sklearn.preprocessing import LabelEncoder

from dataset import Dataset


def compute_binary_feature_mi(X, y_enc):
    """Compute MI(feature; target) for binary sparse features and multiclass targets."""
    X = X.tocsr()
    n_samples, n_features = X.shape
    if n_samples == 0:
        return np.zeros(n_features, dtype=float)

    classes = int(np.max(y_enc)) + 1
    y_counts = np.bincount(y_enc, minlength=classes).astype(float)
    p_y = y_counts / float(n_samples)

    # Number of rows where feature j is present.
    n1 = np.asarray(X.getnnz(axis=0), dtype=float)
    p_x1 = n1 / float(n_samples)
    p_x0 = 1.0 - p_x1

    # Build sparse one-hot labels and get counts n11[j, c] = #(x_j=1, y=c).
    rows = np.arange(n_samples)
    data = np.ones(n_samples, dtype=float)
    y_onehot = scipy.sparse.csr_matrix((data, (rows, y_enc)), shape=(n_samples, classes))
    n11 = X.T.dot(y_onehot).toarray().astype(float)
    p11 = n11 / float(n_samples)

    scores = np.zeros(n_features, dtype=float)
    eps = 1e-20

    for j in range(n_features):
        mi = 0.0
        for c in range(classes):
            py = p_y[c]
            if py <= 0.0:
                continue

            p_x1y = p11[j, c]
            p_x0y = py - p_x1y

            if p_x1y > 0.0 and p_x1[j] > 0.0:
                mi += p_x1y * np.log((p_x1y + eps) / (p_x1[j] * py + eps))
            if p_x0y > 0.0 and p_x0[j] > 0.0:
                mi += p_x0y * np.log((p_x0y + eps) / (p_x0[j] * py + eps))

        scores[j] = mi

    return scores


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute feature mutual information scores against BIO labels."
    )
    parser.add_argument("datafile", help="Path to a .feat file (e.g., preprocessed/train.feat)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output TSV file path. Defaults to <datafile>.mi.tsv",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="How many top features to print to stdout (default: 50)",
    )
    parser.add_argument(
        "--granularity",
        choices=["templates", "values"],
        default="templates",
        help="Rank feature templates or concrete feature values (default: templates)",
    )
    return parser.parse_args()


def feature_template(feature_name):
    return feature_name.split("=", 1)[0] if "=" in feature_name else feature_name


def main():
    args = parse_args()

    ds = Dataset(args.datafile)
    X, y = ds.csr_matrix()

    if len(y) == 0:
        raise ValueError(f"No instances found in data file: {args.datafile}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scores = compute_binary_feature_mi(X, y_enc)

    idx_to_feature = {idx: feat for feat, idx in ds.feature_index().items()}
    ranked_values = sorted(
        ((idx_to_feature[i], float(scores[i])) for i in range(len(scores))),
        key=lambda x: x[1],
        reverse=True,
    )

    if args.granularity == "templates":
        template_scores = defaultdict(float)
        template_counts = defaultdict(int)
        for feat, mi in ranked_values:
            tpl = feature_template(feat)
            template_scores[tpl] += mi
            template_counts[tpl] += 1

        ranked = sorted(template_scores.items(), key=lambda x: x[1], reverse=True)
    else:
        ranked = ranked_values

    out_path = args.output or f"{args.datafile}.mi.tsv"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out:
        if args.granularity == "templates":
            out.write("rank\tfeature\tmi\tn_values\n")
            for rank, (feat, mi) in enumerate(ranked, start=1):
                out.write(f"{rank}\t{feat}\t{mi:.12f}\t{template_counts[feat]}\n")
        else:
            out.write("rank\tfeature\tmi\n")
            for rank, (feat, mi) in enumerate(ranked, start=1):
                out.write(f"{rank}\t{feat}\t{mi:.12f}\n")

    print(f"Saved MI scores to: {out_path}")
    unit = "templates" if args.granularity == "templates" else "features"
    print(f"Total {unit} scored: {len(ranked)}")
    print(f"Top {min(args.topk, len(ranked))} {unit}:")
    for rank, (feat, mi) in enumerate(ranked[: args.topk], start=1):
        print(f"{rank:4d}  {mi: .8f}  {feat}")


if __name__ == "__main__":
    main()
