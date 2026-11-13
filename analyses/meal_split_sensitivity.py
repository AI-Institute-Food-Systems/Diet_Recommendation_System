#!/usr/bin/env python3
"""Meal energy-split sensitivity (S1 Table 11): re-portion the generated food sets for
alternative breakfast/lunch/dinner energy splits and recompute the median per-cluster
reduction in RDI deviation against the real meals.

To keep the iterative portion optimizer fast, at most `--max_meals_per_cluster` meals
per cluster are used (random, seeded). Input: <run>/data_exports from `make eval`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from analyses._common import (MEAL_TYPES, cluster_level_medians, load_exports,
                              median_percent_reduction, portion_only_generator, reportion,
                              stable_seed, subsample_per_cluster)
from cli.evaluate import per_meal_deviation_from_rdi


def parse_split(spec: str) -> Dict[str, float]:
    """'25,35,40' (normalised to sum 1) or a JSON file mapping Breakfast/Lunch/Dinner to fractions."""
    p = Path(spec)
    if p.exists():
        return {k: float(v) for k, v in json.loads(p.read_text()).items()}
    parts = [float(x) for x in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid split spec: {spec} (expected 'b,l,d' or a JSON file)")
    tot = sum(parts)
    return {"Breakfast": parts[0] / tot, "Lunch": parts[1] / tot, "Dinner": parts[2] / tot}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exports_dir", default="outputs/runs/eval_run/data_exports")
    ap.add_argument("--out_dir", default="outputs/runs/eval_run/supplementary/meal_split_sensitivity")
    ap.add_argument("--splits", nargs="+", default=["25,35,40", "30,35,35", "20,40,40", "33,33,34"])
    ap.add_argument("--max_meals_per_cluster", type=int, default=200, help="0 = all meals.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ex = load_exports(args.exports_dir)
    W, RDI, names = ex["W_per100g"], ex["RDI"], ex["nutrient_names"]

    rows = []
    for spec in args.splits:
        fractions = parse_split(spec)
        genr = portion_only_generator(ex, fractions, args.seed)

        for mt in MEAL_TYPES:
            Xr = np.asarray(ex["real_by_type"][mt], dtype=np.float32)
            Xg = np.asarray(ex["gen_by_type"][mt], dtype=np.float32)
            Cr = np.asarray(ex["clusters_by_type"][mt])

            rng = np.random.default_rng(stable_seed(args.seed, mt, spec))
            keep = subsample_per_cluster(Cr, int(args.max_meals_per_cluster), rng)
            Xr, Xg, Cr = Xr[keep], Xg[keep], Cr[keep]

            Xg_portioned = reportion(genr, Xg > 0, mt)
            dev_r = np.abs(per_meal_deviation_from_rdi(Xr, mt, W, names, RDI, fractions, mode="log2"))
            dev_g = np.abs(per_meal_deviation_from_rdi(Xg_portioned, mt, W, names, RDI, fractions, mode="log2"))
            real_meds = cluster_level_medians(dev_r, Cr)

            rows.append({
                "split_spec": spec,
                "meal_type": mt,
                "n_meals": int(Xr.shape[0]),
                "n_clusters": int(len(real_meds)),
                "median_cluster_percent_reduction_generated_vs_real":
                    median_percent_reduction(real_meds, cluster_level_medians(dev_g, Cr)),
                "breakfast_frac": float(fractions["Breakfast"]),
                "lunch_frac": float(fractions["Lunch"]),
                "dinner_frac": float(fractions["Dinner"]),
            })

    pd.DataFrame(rows).to_csv(out_dir / "meal_split_sensitivity_summary.csv", index=False)
    print(f"Wrote {out_dir / 'meal_split_sensitivity_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
