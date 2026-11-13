#!/usr/bin/env python3
"""Confidence intervals for the primary outcomes (S1 Table 18).

RDI alignment: per meal type, the generated-meal median per-cluster reduction in RDI
deviation and its cluster-bootstrap CI, read from the portion ablation
(analyses/portion_ablation.py). As in S1 Table 18, the "Overall" row is the
median of the three point estimates and its interval is the range spanned by the three
meal-type CIs; it is not a bootstrap interval.

Substitution operating points: per k, the median health gain (ΔRDI %) and cost saving
(−ΔCost %) of the selected substitutions, with 1,000-resample bootstrap CIs. As in
S1 Table 18 and the Results text, every k is taken at θ = 1 by default; `--theta knee`
uses each k's knee θ from the run's θ sweep instead (the points labelled in Figure 4B).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rdi_alignment_rows(ablation_summary: Path) -> list[dict]:
    abl = pd.read_csv(ablation_summary)
    abl["pct_med"] = abl["median_cluster_percent_reduction_generated_plus_optimizer"] * 100
    abl["ci_lo"] = abl["ci95_lo_generated_plus_optimizer"] * 100
    abl["ci_hi"] = abl["ci95_hi_generated_plus_optimizer"] * 100

    def row(stratum, med, lo, hi):
        return {"outcome": "RDI alignment (median % reduction)", "stratum": stratum,
                "point_estimate_pct": round(med, 1), "ci95_lo": round(lo, 1), "ci95_hi": round(hi, 1),
                "ci_str": f"[{lo:.1f}–{hi:.1f}]"}

    rows = [row(r["meal_type"], r["pct_med"], r["ci_lo"], r["ci_hi"]) for _, r in abl.iterrows()]
    rows.append(row("Overall", float(np.median(abl["pct_med"].values)),
                    float(np.min(abl["ci_lo"].values)), float(np.max(abl["ci_hi"].values))))
    return rows


def theta_tag(theta: float) -> str:
    return f"{float(theta):g}".replace(".", "p")


def knee_rows(sweep_dir: Path, theta: str, n_boot: int = 1000, seed: int = 42) -> list[dict]:
    if theta == "knee":
        theta_by_k = {int(k): float(v) for k, v in
                      json.loads((sweep_dir / "figures_theta_sweep" / "optimal_theta_by_k.json").read_text()).items()}
    else:
        theta_by_k = {k: float(theta) for k in (1, 2, 3)}

    rng = np.random.default_rng(seed)
    rows = []
    for k in (1, 2, 3):
        path = sweep_dir / f"tradeoff_theta_{theta_tag(theta_by_k[k])}_k{k}" / "substitution_matches.csv"
        if not path.exists():
            print(f"missing {path}; skipping k={k}")
            continue
        df = pd.read_csv(path)
        df = df[df["k_substitution"] == k]
        if df.empty:
            continue
        h = pd.to_numeric(df["dRDI_pct"], errors="coerce").to_numpy()  # health gain, %
        savings = -pd.to_numeric(df["dCost_pct"], errors="coerce").to_numpy()
        m = np.isfinite(h) & np.isfinite(savings)
        h, savings = h[m], savings[m]
        if len(h) < 10:
            continue

        n = len(h)
        h_boot, s_boot = [], []
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n, n)
            h_boot.append(np.nanmedian(h[idx]))
            s_boot.append(np.nanmedian(savings[idx]))
        h_lo, h_hi = np.percentile(h_boot, [2.5, 97.5])
        s_lo, s_hi = np.percentile(s_boot, [2.5, 97.5])
        rows.append({
            "k": k,
            "theta": theta_by_k[k],
            "nutrition_gain_med": round(float(np.nanmedian(h)), 1),
            "nutrition_gain_ci_lo": round(h_lo, 1),
            "nutrition_gain_ci_hi": round(h_hi, 1),
            "cost_savings_med": round(float(np.nanmedian(savings)), 1),
            "cost_savings_ci_lo": round(s_lo, 1),
            "cost_savings_ci_hi": round(s_hi, 1),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ablation_summary",
                    default="outputs/runs/eval_run/supplementary/ablation_portions/ablation_summary.csv")
    ap.add_argument("--sweep_dir", default="outputs/runs/eval_run/substitutions_k3",
                    help="Folder with tradeoff_theta_<θ>_k<k>/ runs and figures_theta_sweep/.")
    ap.add_argument("--theta", default="1", help="One θ for every k (default 1), or 'knee' for each k's knee θ.")
    ap.add_argument("--out_dir", default="outputs/runs/eval_run/supplementary/primary_outcomes_cis")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rdi_alignment_rows(Path(args.ablation_summary))).to_csv(out_dir / "rdi_alignment_cis.csv", index=False)
    print(f"Wrote {out_dir / 'rdi_alignment_cis.csv'}")

    rows = knee_rows(Path(args.sweep_dir), args.theta)
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "substitution_knee_cis.csv", index=False)
        print(f"Wrote {out_dir / 'substitution_knee_cis.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
