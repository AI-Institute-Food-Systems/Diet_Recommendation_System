#!/usr/bin/env python3
"""Cost-model sensitivity (S1 Tables 14 and 15).

Overhead (S-COST1): from a k=1 substitution run with restaurant pricing and no overhead
(<runs_dir>/restaurant_overhead0), the median cost change when a flat overhead H is added
to every meal, ΔCost% = (gen − real) / (real + H) × 100, for H = 0, 1, 2, 3 USD. The
dollar difference does not depend on H, so one run covers every H. Bootstrap CIs over meals.

Grocery-only pricing (S-COST2): median savings (−ΔCost %) of the selected substitutions in
runs priced from the per-100 g price table (with the substitution script's food-category
multipliers), without the portion model or overhead (<runs_dir>/grocery_k{1,2,3}).

`make sup-substitutions` runs the substitution analyses and then this script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_median_ci(vals: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    meds = np.array([np.median(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(n_boot)])
    return float(np.median(vals)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def overhead_summary(matches_csv: Path, overheads: list[float], n_boot: int) -> pd.DataFrame:
    df = pd.read_csv(matches_csv)
    real = df["real_cost"].astype(float)
    gen = df["gen_cost"].astype(float)
    rng = np.random.default_rng(42)
    rows = []
    for H in overheads:
        real_h = real + H
        pct = np.where(real_h > 0, (gen - real) / real_h * 100, np.nan)
        pct = pct[np.isfinite(pct)]
        if len(pct) == 0:
            continue
        med = float(np.nanmedian(pct))
        boots = np.array([np.nanmedian(pct[rng.integers(0, len(pct), len(pct))]) for _ in range(n_boot)])
        rows.append({
            "overhead_usd": H,
            "n_meals": len(pct),
            "median_dCost_pct": med,
            "median_savings_pct": -med,
            "ci_lo_dCost_pct": float(np.percentile(boots, 2.5)),
            "ci_hi_dCost_pct": float(np.percentile(boots, 97.5)),
        })
    return pd.DataFrame(rows)


def grocery_table(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for k in (1, 2, 3):
        path = runs_dir / f"grocery_k{k}" / "substitution_matches.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if "selected" in df.columns:
            df = df[df["selected"] == True]
        if df.empty:
            rows.append({"k": k, "n_meals": 0 if path.exists() else None, "median_savings_pct": np.nan,
                         "ci_lo_savings_pct": np.nan, "ci_hi_savings_pct": np.nan})
            continue
        savings = -pd.to_numeric(df["dCost_pct"], errors="coerce").to_numpy()
        savings = savings[np.isfinite(savings)]
        med, lo, hi = bootstrap_median_ci(savings, 1000, 42)
        rows.append({"k": k, "n_meals": len(savings), "median_savings_pct": round(med, 1),
                     "ci_lo_savings_pct": round(lo, 1), "ci_hi_savings_pct": round(hi, 1)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs_dir", default="outputs/runs/eval_run/supplementary/cost_sensitivity",
                    help="Folder with restaurant_overhead0/ and grocery_k{1,2,3}/ substitution runs.")
    ap.add_argument("--out_dir", default=None, help="Default: --runs_dir.")
    ap.add_argument("--overheads", default="0,1,2,3", help="Comma-separated overheads in USD.")
    ap.add_argument("--bootstraps", type=int, default=1000)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else runs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = overhead_summary(runs_dir / "restaurant_overhead0" / "substitution_matches.csv",
                               [float(x) for x in args.overheads.split(",")], args.bootstraps)
    summary.to_csv(out_dir / "cost_sensitivity_summary.csv", index=False)

    s1 = summary.assign(ci_lo_savings_pct=-summary["ci_hi_dCost_pct"], ci_hi_savings_pct=-summary["ci_lo_dCost_pct"])
    s1 = s1[["overhead_usd", "n_meals", "median_dCost_pct", "median_savings_pct", "ci_lo_savings_pct", "ci_hi_savings_pct"]]
    s1.to_csv(out_dir / "table_S_COST1_overhead_sensitivity.csv", index=False)

    s2 = grocery_table(runs_dir)
    s2.to_csv(out_dir / "table_S_COST2_grocery_only.csv", index=False)

    print("Overhead sensitivity (1-hop, restaurant pricing):")
    print(s1.to_string(index=False))
    print("\nGrocery-only pricing:")
    print(s2.to_string(index=False))
    print(f"\nWrote tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
