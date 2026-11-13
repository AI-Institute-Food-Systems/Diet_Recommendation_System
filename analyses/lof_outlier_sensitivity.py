#!/usr/bin/env python3
"""LOF outlier sensitivity (S1 Table 17): drop the most outlying real meals at several
contamination levels and recompute the median per-cluster reduction in RDI deviation.

LOF is fitted on binary food presence of the evaluation cohort's real meals, so this
tests evaluation-time outlier removal, not the 0.3% LOF step in preprocessing.
Input: <run>/data_exports from `make eval`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from analyses._common import (DEFAULT_MEAL_FRACTIONS, MEAL_TYPES, cluster_level_medians,
                              load_exports, median_percent_reduction)
from cli.evaluate import per_meal_deviation_from_rdi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exports_dir", default="outputs/runs/eval_run/data_exports")
    ap.add_argument("--out_dir", default="outputs/runs/eval_run/supplementary/outlier_sensitivity")
    ap.add_argument("--contaminations", type=float, nargs="+", default=[0.001, 0.003, 0.005, 0.01],
                    help="Fractions removed as outliers (0.003 = 0.3%%).")
    ap.add_argument("--n_neighbors", type=int, default=35, help="LOF neighbourhood size.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ex = load_exports(args.exports_dir)
    W, RDI, names = ex["W_per100g"], ex["RDI"], ex["nutrient_names"]

    rows = []
    for mt in MEAL_TYPES:
        Xr = np.asarray(ex["real_by_type"][mt], dtype=np.float32)
        Xg = np.asarray(ex["gen_by_type"][mt], dtype=np.float32)
        Cr = np.asarray(ex["clusters_by_type"][mt])

        lof = LocalOutlierFactor(n_neighbors=int(args.n_neighbors), contamination=float(max(args.contaminations)))
        lof.fit_predict((Xr > 0).astype(np.float32))
        scores = np.asarray(lof.negative_outlier_factor_, dtype=np.float32)  # lower = more outlying

        for c in args.contaminations:
            c = float(c)
            keep = scores > np.quantile(scores, c) if c > 0 else np.ones(Xr.shape[0], dtype=bool)

            dev_r = np.abs(per_meal_deviation_from_rdi(Xr[keep], mt, W, names, RDI, DEFAULT_MEAL_FRACTIONS, mode="log2"))
            dev_g = np.abs(per_meal_deviation_from_rdi(Xg[keep], mt, W, names, RDI, DEFAULT_MEAL_FRACTIONS, mode="log2"))
            red = median_percent_reduction(cluster_level_medians(dev_r, Cr[keep]),
                                           cluster_level_medians(dev_g, Cr[keep]))
            rows.append({
                "meal_type": mt,
                "contamination": c,
                "n_meals_before": int(Xr.shape[0]),
                "n_meals_after": int(keep.sum()),
                "median_cluster_percent_reduction_generated_vs_real": float(red),
            })

    pd.DataFrame(rows).to_csv(out_dir / "outlier_sensitivity_summary.csv", index=False)
    print(f"Wrote {out_dir / 'outlier_sensitivity_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
