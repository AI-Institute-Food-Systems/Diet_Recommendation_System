#!/usr/bin/env python3
"""Fold-wise ("leakage-safe") clustering check: within each fold, fit a z-score scaling and
HDBSCAN on the training meals only, assign the held-out meals with `approximate_predict`,
and compare with the full-data labels by adjusted Rand index (ARI).

The input is the feature matrix exported by `make clustering`
(<meal>_feature_matrix_unscaled.csv), which is already z-scored over all meals. `make
supplementary` uses all meals (`--max_samples 0`), in 3 folds for breakfast and 2 for lunch
and dinner.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from hdbscan import prediction as hdbscan_prediction
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold

from analyses.cluster_stability import hdbscan_params, n_clusters


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--meal_type", required=True, choices=["breakfast", "lunch", "dinner"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=12000,
                    help="Random subsample of meals for faster runs (0 = all meals).")
    ap.add_argument("--clustering_cache", default="outputs/clustering/cached_clustering.pkl")
    ap.add_argument("--unscaled_feature_csv", default=None,
                    help="Default: outputs/clustering/<meal>_feature_matrix_unscaled.csv")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pickle.loads(Path(args.clustering_cache).read_bytes())["meal_type_clusters"][args.meal_type]
    y_base = np.asarray(base["labels"])
    params = dict(hdbscan_params(base["clusterer"]), prediction_data=True)

    csv_path = Path(args.unscaled_feature_csv or f"outputs/clustering/{args.meal_type}_feature_matrix_unscaled.csv")
    X = pd.read_csv(csv_path).values.astype(np.float32)
    n = X.shape[0]
    assert y_base.shape[0] == n, f"Baseline labels length {y_base.shape[0]} != X rows {n}"

    if int(args.max_samples) > 0 and n > int(args.max_samples):
        idx = np.random.default_rng(int(args.seed)).choice(n, size=int(args.max_samples), replace=False)
        idx.sort()
        X, y_base = X[idx], y_base[idx]
        n = X.shape[0]

    kf = KFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.seed))
    rows = []
    for fold, (tr, te) in enumerate(kf.split(np.arange(n))):
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)

        cl = hdbscan.HDBSCAN(**params)
        y_tr = cl.fit_predict((X[tr] - mu) / sd)
        y_te, _ = hdbscan_prediction.approximate_predict(cl, (X[te] - mu) / sd)

        yb = y_base[te]
        nn = (yb != -1) & (y_te != -1)
        rows.append({
            "fold": fold,
            "n_train": int(tr.size),
            "n_test": int(te.size),
            "ari_test_vs_baseline": float(adjusted_rand_score(yb, y_te)),
            "ari_test_vs_baseline_nonnoise": float(adjusted_rand_score(yb[nn], y_te[nn])) if int(nn.sum()) >= 10 else float("nan"),
            "noise_frac_test_vs_baseline": float(np.mean(yb == -1)),
            "noise_frac_test_leakage_safe": float(np.mean(y_te == -1)),
            "n_clusters_train": n_clusters(y_tr),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "leakage_safe_clustering_folds.csv", index=False)

    pd.DataFrame([{
        "meal_type": args.meal_type,
        "folds": int(args.folds),
        "ari_test_median": float(df["ari_test_vs_baseline"].median()),
        "ari_test_mean": float(df["ari_test_vs_baseline"].mean()),
        "ari_test_ci95_lo": float(df["ari_test_vs_baseline"].quantile(0.025)),
        "ari_test_ci95_hi": float(df["ari_test_vs_baseline"].quantile(0.975)),
        "ari_test_nonnoise_median": float(df["ari_test_vs_baseline_nonnoise"].median(skipna=True)),
        "ari_test_nonnoise_mean": float(df["ari_test_vs_baseline_nonnoise"].mean(skipna=True)),
    }]).to_csv(out_dir / "leakage_safe_clustering_summary.csv", index=False)

    print(f"Wrote {out_dir / 'leakage_safe_clustering_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
