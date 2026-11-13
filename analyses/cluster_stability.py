#!/usr/bin/env python3
"""Cluster stability (S1 Table 12): re-cluster random subsamples with the same HDBSCAN
settings and compare with the full-data labels by adjusted Rand index (ARI).

Inputs come from `make clustering`. S1 Table 12 uses 10 subsamples of 80% (breakfast)
and 50% (lunch, dinner) of meals; `make supplementary` passes those.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def hdbscan_params(clusterer) -> dict:
    """Settings of the fitted baseline clusterer (per-meal epsilon included)."""
    return dict(
        min_cluster_size=int(getattr(clusterer, "min_cluster_size", 200)),
        min_samples=int(getattr(clusterer, "min_samples", 100)),
        alpha=float(getattr(clusterer, "alpha", 1.0)),
        cluster_selection_epsilon=float(getattr(clusterer, "cluster_selection_epsilon", 0.0)),
        cluster_selection_method=str(getattr(clusterer, "cluster_selection_method", "eom")),
        metric=str(getattr(clusterer, "metric", "euclidean")),
    )


def n_clusters(labels: np.ndarray) -> int:
    s = set(labels.tolist())
    return len(s) - (1 if -1 in s else 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--meal_type", required=True, choices=["breakfast", "lunch", "dinner"])
    ap.add_argument("--B", type=int, default=10, help="Number of subsamples.")
    ap.add_argument("--subsample_frac", type=float, default=0.8, help="Fraction of meals per subsample.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features_cache", default="outputs/clustering/cached_features_hybrid_zscore.pkl")
    ap.add_argument("--clustering_cache", default="outputs/clustering/cached_clustering.pkl")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = pickle.loads(Path(args.features_cache).read_bytes())
    clustering = pickle.loads(Path(args.clustering_cache).read_bytes())

    X = features["meal_type_features"][args.meal_type]
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.asarray(X, dtype=np.float32)

    base = clustering["meal_type_clusters"][args.meal_type]
    y0 = np.asarray(base["labels"])
    params = hdbscan_params(base["clusterer"])

    rng = np.random.default_rng(int(args.seed))
    n = X.shape[0]
    m = int(max(2, round(float(args.subsample_frac) * n)))

    rows = []
    for b in range(int(args.B)):
        idx = rng.choice(n, size=m, replace=False)
        y0b = y0[idx]
        yb = hdbscan.HDBSCAN(**params).fit_predict(X[idx])

        nn = (y0b != -1) & (yb != -1)
        rows.append({
            "replicate": b,
            "n": int(m),
            "ari_vs_baseline": float(adjusted_rand_score(y0b, yb)),
            "ari_vs_baseline_nonnoise": float(adjusted_rand_score(y0b[nn], yb[nn])) if int(nn.sum()) >= 10 else float("nan"),
            "n_clusters_baseline_sub": n_clusters(y0b),
            "n_clusters_recluster": n_clusters(yb),
            "noise_frac_baseline_sub": float(np.mean(y0b == -1)),
            "noise_frac_recluster": float(np.mean(yb == -1)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stability_replicates.csv", index=False)

    pd.DataFrame([{
        "meal_type": args.meal_type,
        "B": int(args.B),
        "subsample_frac": float(args.subsample_frac),
        "ari_median": float(df["ari_vs_baseline"].median()),
        "ari_mean": float(df["ari_vs_baseline"].mean()),
        "ari_ci95_lo": float(df["ari_vs_baseline"].quantile(0.025)),
        "ari_ci95_hi": float(df["ari_vs_baseline"].quantile(0.975)),
        "ari_nonnoise_median": float(df["ari_vs_baseline_nonnoise"].median(skipna=True)),
        "ari_nonnoise_mean": float(df["ari_vs_baseline_nonnoise"].mean(skipna=True)),
    }]).to_csv(out_dir / "stability_summary.csv", index=False)

    print(f"Wrote {out_dir / 'stability_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
