#!/usr/bin/env python3
"""Portion ablation (S1 Table 19): separate the portion optimizer's contribution from the
CVAE's food choices. Compares, per cluster, the RDI deviation of
  - real meals as eaten,
  - real food sets re-portioned by the optimizer ("real + optimizer"),
  - generated meals ("generated + optimizer", as evaluated in the paper),
and reports the median per-cluster reduction with a cluster bootstrap CI.

At most `--max_meals_per_cluster` meals per cluster are used (random, seeded).
Input: <run>/data_exports from `make eval`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from analyses._common import (DEFAULT_MEAL_FRACTIONS, MEAL_TYPES, cluster_level_medians,
                              cluster_reductions, load_exports, median_percent_reduction,
                              portion_only_generator, reportion, stable_seed,
                              subsample_per_cluster)
from cli.evaluate import per_meal_deviation_from_rdi


def bootstrap_median_ci(values: np.ndarray, *, n_boot: int, seed: int) -> Tuple[float, float]:
    """Percentile bootstrap CI for the median."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    boots = [float(np.median(rng.choice(values, size=values.size, replace=True))) for _ in range(int(n_boot))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exports_dir", default="outputs/runs/eval_run/data_exports")
    ap.add_argument("--out_dir", default="outputs/runs/eval_run/supplementary/ablation_portions")
    ap.add_argument("--meal_fraction_map_json", default=None,
                    help="JSON mapping Breakfast/Lunch/Dinner to energy fractions (default 25/35/40).")
    ap.add_argument("--n_boot", type=int, default=1000, help="Cluster-bootstrap resamples.")
    ap.add_argument("--max_meals_per_cluster", type=int, default=200, help="0 = all meals.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ex = load_exports(args.exports_dir)
    W, RDI, names = ex["W_per100g"], ex["RDI"], ex["nutrient_names"]
    fractions = DEFAULT_MEAL_FRACTIONS
    if args.meal_fraction_map_json:
        fractions = json.loads(Path(args.meal_fraction_map_json).read_text())
    genr = portion_only_generator(ex, fractions, args.seed)

    def deviation(X, mt):
        return np.abs(per_meal_deviation_from_rdi(X, mt, W, names, RDI, fractions, mode="log2"))

    rows, per_cluster_rows = [], []
    for mt in MEAL_TYPES:
        Xr = np.asarray(ex["real_by_type"][mt], dtype=np.float32)
        Xg = np.asarray(ex["gen_by_type"][mt], dtype=np.float32)
        Cr = np.asarray(ex["clusters_by_type"][mt])

        rng = np.random.default_rng(stable_seed(args.seed, mt))
        keep = subsample_per_cluster(Cr, int(args.max_meals_per_cluster), rng)
        Xr, Xg, Cr = Xr[keep], Xg[keep], Cr[keep]

        real_meds = cluster_level_medians(deviation(Xr, mt), Cr)
        realopt_meds = cluster_level_medians(deviation(reportion(genr, Xr > 0, mt), mt), Cr)
        gen_meds = cluster_level_medians(deviation(Xg, mt), Cr)

        ci_realopt = bootstrap_median_ci(cluster_reductions(real_meds, realopt_meds), n_boot=args.n_boot, seed=args.seed + 10)
        ci_gen = bootstrap_median_ci(cluster_reductions(real_meds, gen_meds), n_boot=args.n_boot, seed=args.seed + 20)

        rows.append({
            "meal_type": mt,
            "n_meals": int(Xr.shape[0]),
            "n_clusters": int(len(real_meds)),
            "median_cluster_percent_reduction_real_plus_optimizer": median_percent_reduction(real_meds, realopt_meds),
            "ci95_lo_real_plus_optimizer": ci_realopt[0],
            "ci95_hi_real_plus_optimizer": ci_realopt[1],
            "median_cluster_percent_reduction_generated_plus_optimizer": median_percent_reduction(real_meds, gen_meds),
            "ci95_lo_generated_plus_optimizer": ci_gen[0],
            "ci95_hi_generated_plus_optimizer": ci_gen[1],
        })
        for cid in sorted(real_meds):
            per_cluster_rows.append({
                "meal_type": mt,
                "cluster_id": int(cid),
                "median_dev_real_as_eaten": float(real_meds[cid]),
                "median_dev_real_plus_optimizer": float(realopt_meds.get(cid, float("nan"))),
                "median_dev_generated_plus_optimizer": float(gen_meds.get(cid, float("nan"))),
            })

    pd.DataFrame(rows).to_csv(out_dir / "ablation_summary.csv", index=False)
    pd.DataFrame(per_cluster_rows).to_csv(out_dir / "ablation_by_cluster.csv", index=False)
    print(f"Wrote {out_dir / 'ablation_summary.csv'} and ablation_by_cluster.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
