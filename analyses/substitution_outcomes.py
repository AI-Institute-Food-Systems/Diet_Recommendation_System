#!/usr/bin/env python3
"""Substitution coverage and outcome distribution (per-hop counts in the Results; S1
robustness metrics "S-ROB5").

Each generated meal is assigned to the lowest k (1, 2, 3) at which it has a selected
substitution. Reports how many meals have one at each k, the share of the evaluation
cohort with none, and, by k and meal type: the share with worse RDI alignment, median and
IQR of ΔRDI % and ΔCost %, the share with a cost increase, and the share worse on either.

Input: substitution runs, one folder per k (`--k_dirs`), or one run holding all k
(`--matches`). The cohort size is read from the run's tables/cluster_counts_totals.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MEALS = ["Breakfast", "Lunch", "Dinner"]


def lowest_k_per_meal(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """One row per (Meal Type, gen_idx): its selected substitution with the lowest k."""
    rows = []
    for df in frames:
        if "selected" in df.columns:
            df = df[df["selected"] == True]
        k = df["k_substitution"] if "k_substitution" in df.columns else df["total_food_diff"]
        part = pd.DataFrame({
            "Meal Type": df["Meal Type"].values,
            "gen_idx": df["gen_idx"].values,
            "k": k.astype(int).values,
            "dRDI_pct": pd.to_numeric(df["dRDI_pct"], errors="coerce").values if "dRDI_pct" in df.columns
            else 100.0 * pd.to_numeric(df["improvement"], errors="coerce").values,
            "dCost_pct": pd.to_numeric(df["dCost_pct"], errors="coerce").values if "dCost_pct" in df.columns
            else (100.0 * df["delta_cost"] / df["real_cost"].replace(0, np.nan)).values,
        })
        rows.append(part)
    merged = pd.concat(rows, ignore_index=True)
    merged = merged.sort_values(["Meal Type", "gen_idx", "k"], kind="mergesort")
    return merged.drop_duplicates(subset=["Meal Type", "gen_idx"], keep="first")


def outcome_stats(sub: pd.DataFrame) -> dict:
    rdi, cost = sub["dRDI_pct"].dropna(), sub["dCost_pct"].dropna()
    return {
        "n": int(len(sub)),
        "pct_worse_rdi": float((rdi < 0).mean() * 100),
        "median_dRDI_pct": float(rdi.median()),
        "q25_dRDI_pct": float(rdi.quantile(0.25)),
        "q75_dRDI_pct": float(rdi.quantile(0.75)),
        "median_dCost_pct": float(cost.median()),
        "q25_dCost_pct": float(cost.quantile(0.25)),
        "q75_dCost_pct": float(cost.quantile(0.75)),
        "pct_cost_increase": float((cost > 0).mean() * 100),
        "pct_worse_rdi_or_cost": float(((sub["dRDI_pct"] < 0) | (sub["dCost_pct"] > 0)).mean() * 100),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--k_dirs", nargs="+", type=Path, help="Run folders for k = 1, 2, 3.")
    src.add_argument("--matches", type=Path, help="One substitution_matches.csv holding all k.")
    ap.add_argument("--cohort_counts", default="outputs/runs/eval_run/tables/cluster_counts_totals.csv",
                    help="Evaluation cohort size (row ALL), the denominator for 'no substitution'.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [args.matches] if args.matches else [d / "substitution_matches.csv" for d in args.k_dirs]
    merged = lowest_k_per_meal([pd.read_csv(p) for p in paths if p.exists()])
    merged = merged[merged["k"].isin([1, 2, 3])]  # k = 0 would be an unchanged meal

    totals = pd.read_csv(args.cohort_counts)
    n_total = int(totals.loc[totals["meal_type"] == "ALL", "assessed_count"].iloc[0])
    n_with = int(len(merged))
    counts = {"n_total": n_total, "n_with_substitution": n_with,
              "pct_with_substitution": 100.0 * n_with / n_total,
              "pct_no_substitution": 100.0 * (n_total - n_with) / n_total}
    for k in (1, 2, 3):
        counts[f"n_k{k}"] = int((merged["k"] == k).sum())
    for meal in MEALS:
        counts[f"n_{meal.lower()}"] = int((merged["Meal Type"] == meal).sum())
    pd.DataFrame([counts]).to_csv(out_dir / "substitution_counts.csv", index=False)

    rows = [{"k": k, "meal_type": "all", **outcome_stats(merged[merged["k"] == k])} for k in (1, 2, 3)]
    rows += [{"k": "all", "meal_type": m, **outcome_stats(merged[merged["Meal Type"] == m])} for m in MEALS]
    rows += [{"k": k, "meal_type": m, **outcome_stats(merged[(merged["k"] == k) & (merged["Meal Type"] == m)])}
             for k in (1, 2, 3) for m in MEALS]
    pd.DataFrame([r for r in rows if r["n"] > 0]).to_csv(out_dir / "substitution_outcomes.csv", index=False)

    print(f"With a substitution: {n_with:,} of {n_total:,} ({counts['pct_with_substitution']:.1f}%); "
          f"k=1/2/3: {counts['n_k1']:,} / {counts['n_k2']:,} / {counts['n_k3']:,}")
    print(f"Wrote {out_dir / 'substitution_counts.csv'} and substitution_outcomes.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
