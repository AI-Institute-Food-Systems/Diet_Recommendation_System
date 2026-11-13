#!/usr/bin/env python3
"""Nutrient-level summary: per nutrient and meal type, the percent reduction in the median
deviation from the per-meal target (generated vs real), with a bootstrap CI.

Deviation per meal and nutrient:
  adequacy nutrients:    |log2((N + eps) / (target + eps))|
  moderation nutrients:  |log2(max(1, (N + eps) / (target + eps)))|   (only excess counts)
Moderation nutrients default to sodium, total sugars and saturated fat.
Input: <run>/data_exports from `make eval`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

DEFAULT_MODERATION = ["sodium_mg", "sugars_total_g", "sfa_g"]
DEFAULT_SPOTLIGHT = ["sodium_mg", "sugars_total_g", "sfa_g", "fiber_g", "potassium_mg", "protein_g"]
MEAL_FRACTIONS = {"breakfast": 0.25, "lunch": 0.35, "dinner": 0.40}


def deviation_per_nutrient(N: np.ndarray, RDI: np.ndarray, nutrient_names: List[str], meal_fraction: float,
                           moderation: Iterable[str], eps: float = 1e-6) -> np.ndarray:
    """(meals x nutrients) deviation matrix aligned with nutrient_names."""
    targets = np.asarray(RDI, float) * float(meal_fraction)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (N + eps) / (targets + eps)
    ratio[~np.isfinite(ratio)] = np.nan

    mod = {m.lower() for m in moderation}
    out = np.full_like(ratio, np.nan, dtype=float)
    for j, name in enumerate(nutrient_names):
        r = ratio[:, j]
        if str(name).lower() in mod:
            r = np.maximum(1.0, r)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:, j] = np.abs(np.log2(r))
    return out


def bootstrap_pct_reduction(real_dev: np.ndarray, gen_dev: np.ndarray, B: int, seed: int) -> Tuple[float, float, float]:
    """Percent reduction in median deviation, with a paired bootstrap CI over meals."""
    rng = np.random.default_rng(seed)
    n = int(real_dev.shape[0])
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    med_r = float(np.nanmedian(real_dev))
    med_g = float(np.nanmedian(gen_dev))
    pct = float((med_r - med_g) / med_r * 100.0) if np.isfinite(med_r) and med_r > 0 else float("nan")

    boots = []
    for _ in range(int(B)):
        idx = rng.integers(0, n, size=n)
        mr = float(np.nanmedian(real_dev[idx]))
        mg = float(np.nanmedian(gen_dev[idx]))
        if np.isfinite(mr) and mr > 0:
            boots.append((mr - mg) / mr * 100.0)
    if not boots:
        return pct, float("nan"), float("nan")
    lo, hi = np.nanpercentile(np.asarray(boots, float), [2.5, 97.5])
    return pct, float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exports_dir", default="outputs/runs/eval_run/data_exports")
    ap.add_argument("--out_dir", default="outputs/runs/eval_run/supplementary/nutrient_level_summary")
    ap.add_argument("--bootstraps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--moderation", default=",".join(DEFAULT_MODERATION),
                    help="Comma-separated nutrients treated as moderation (one-sided).")
    ap.add_argument("--spotlight", default=",".join(DEFAULT_SPOTLIGHT),
                    help="Comma-separated nutrients for the compact spotlight table.")
    args = ap.parse_args()

    exports_dir = Path(args.exports_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = {k: np.asarray(v) for k, v in pd.read_pickle(exports_dir / "generated_meals_by_type.pkl").items()}
    real = {k: np.asarray(v) for k, v in pd.read_pickle(exports_dir / "real_meals_by_type.pkl").items()}
    W = np.load(exports_dir / "nutrient_matrix.npy")
    RDI = np.load(exports_dir / "RDI.npy")
    nutrient_names = json.loads((exports_dir / "nutrient_names.json").read_text())
    name_idx: Dict[str, int] = {str(n).lower(): i for i, n in enumerate(nutrient_names)}

    moderation = [s.strip().lower() for s in args.moderation.split(",") if s.strip()]
    spotlight = [s.strip().lower() for s in args.spotlight.split(",") if s.strip()]

    rows = []
    for meal, frac in MEAL_FRACTIONS.items():
        gX, rX = gen.get(meal.capitalize()), real.get(meal.capitalize())
        if gX is None or rX is None:
            continue
        gD = deviation_per_nutrient((np.asarray(gX, float) / 100.0) @ W, RDI, nutrient_names, frac, moderation)
        rD = deviation_per_nutrient((np.asarray(rX, float) / 100.0) @ W, RDI, nutrient_names, frac, moderation)

        for name, j in name_idx.items():
            pct, lo, hi = bootstrap_pct_reduction(rD[:, j], gD[:, j], B=args.bootstraps, seed=args.seed + j)
            rows.append({
                "meal_type": meal,
                "nutrient": name,
                "meal_fraction": frac,
                "median_dev_real": float(np.nanmedian(rD[:, j])),
                "median_dev_generated": float(np.nanmedian(gD[:, j])),
                "pct_reduction_median_dev": pct,
                "pct_reduction_ci95_low": lo,
                "pct_reduction_ci95_high": hi,
                "is_moderation": name in set(moderation),
                "n_real": int(rX.shape[0]),
                "n_generated": int(gX.shape[0]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "nutrient_level_summary.csv", index=False)

    spot = df[df["nutrient"].isin(set(spotlight))]
    if not spot.empty:
        wide = spot.pivot_table(index=["nutrient", "is_moderation"], columns="meal_type",
                                values=["pct_reduction_median_dev", "pct_reduction_ci95_low", "pct_reduction_ci95_high"],
                                aggfunc="first")
        wide.columns = [f"{a}__{b}" for a, b in wide.columns]
        wide.reset_index().to_csv(out_dir / "nutrient_spotlight_table.csv", index=False)

    meta = {
        "exports_dir": str(exports_dir),
        "bootstraps": int(args.bootstraps),
        "seed": int(args.seed),
        "moderation": moderation,
        "spotlight": spotlight,
        "meal_fraction_map": MEAL_FRACTIONS,
    }
    (out_dir / "nutrient_level_summary_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_dir / 'nutrient_level_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
