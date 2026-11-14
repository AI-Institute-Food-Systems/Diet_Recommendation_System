#!/usr/bin/env python3
"""Nutrition metrics of CVAE versus GPT meals (`make gpt-evaluate`).

Reads experiment_results.json from generate_meals.py and the nutrient data of the run
(<run_dir>/data_exports from `make eval`). Each meal's foods are looked up by FNDDS code, and the
metrics are computed per meal type with src/analysis_rdi_eval: energy, protein / fat /
carbohydrate shares of energy, AMDR compliance (all three shares within their ranges), MAR, MER
(with the moderation limits of Figure 3: sodium, saturated fat and total sugars) and diversity
(evenness of the gram amounts across the meal's foods, normalised Shannon entropy). Targets are
the daily RDI times the meal's share of daily energy (25 / 35 / 40%).

Writes to --out_dir (default: <results folder>/evaluation):
- per_meal_metrics_<cvae|gpt>_<meal type>.csv: one row per meal;
- summary_metrics.csv: means, SDs and medians per method and meal type;
- comparison.json and comparison.txt: per method, the number of meals and the meal-weighted
  means over all meal types (AMDR compliance, macronutrient shares, diversity).
No API access is needed.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyses._common import DEFAULT_MEAL_FRACTIONS, MEAL_TYPES  # noqa: E402
from experiments.gpt_comparison.common import meal_matrix  # noqa: E402
from src.analysis_rdi_eval.nutrition_metrics import compute_nutrition_metrics_comprehensive  # noqa: E402


def moderation_limits(nutrient_names) -> dict:
    """MER limits as in Figure 3 (src/pipeline_eval/visualization.py): sodium per day, saturated
    fat and sugars as a share of energy."""
    limits = {"sodium_mg": 2300.0, "sfa_pct_kcal": 10.0}
    if "added_sugars_g" in nutrient_names:
        limits["added_sugars_pct_kcal"] = 10.0
    elif "sugars_total_g" in nutrient_names:
        limits["sugars_total_pct_kcal"] = 25.0
    return limits


def load_nutrition(exports_dir: Path) -> dict:
    """Foods, nutrient matrix (per 100 g), nutrient names and daily RDI of a run."""
    nutrient_names = json.loads((exports_dir / "nutrient_names.json").read_text())
    return {
        "feature_names": [str(c).zfill(8) for c in json.loads((exports_dir / "feature_names.json").read_text())],
        "W_per100g": np.load(exports_dir / "nutrient_matrix.npy"),
        "nutrient_names": nutrient_names,
        "RDI": np.load(exports_dir / "RDI.npy"),
    }


def summary_row(model: str, meal_type: str, m: dict) -> dict:
    s = m["summary"]
    macro = s["macro_shares"]
    return {
        "model": model, "meal_type": meal_type, "n_meals": int(s["n_meals"]),
        "energy_kcal_mean": s["energy_kcal"]["mean"], "energy_kcal_std": s["energy_kcal"]["std"],
        "energy_kcal_median": s["energy_kcal"]["median"],
        "macro_protein_pct_mean": macro["protein_pct"]["mean"], "macro_protein_pct_std": macro["protein_pct"]["std"],
        "macro_fat_pct_mean": macro["fat_pct"]["mean"], "macro_fat_pct_std": macro["fat_pct"]["std"],
        "macro_carb_pct_mean": macro["carb_pct"]["mean"], "macro_carb_pct_std": macro["carb_pct"]["std"],
        "amdr_within_pct": s["amdr_within_pct"],
        "MAR_mean": s["mar"]["mean"], "MAR_std": s["mar"]["std"], "MAR_median": s["mar"]["median"],
        "MER_mean": s["mer"]["mean"], "MER_std": s["mer"]["std"], "MER_median": s["mer"]["median"],
        "diversity_mean": s["diversity"]["mean"], "diversity_std": s["diversity"]["std"],
        "diversity_median": s["diversity"]["median"],
    }


def overall(summary: pd.DataFrame, model: str) -> Dict[str, float]:
    """Meal-weighted means over the three meal types."""
    d = summary[summary["model"] == model]
    w = d["n_meals"].to_numpy(dtype=float)
    mean = lambda col: float(np.sum(d[col].to_numpy() * w) / w.sum())  # noqa: E731
    return {
        "n_meals": int(w.sum()),
        "n_meals_by_type": {r.meal_type: int(r.n_meals) for r in d.itertuples()},
        "amdr_compliant_pct": 100.0 * mean("amdr_within_pct"),
        "protein_pct": mean("macro_protein_pct_mean"),
        "fat_pct": mean("macro_fat_pct_mean"),
        "carb_pct": mean("macro_carb_pct_mean"),
        "diversity_pct": 100.0 * mean("diversity_mean"),
    }


def evaluate(results: dict, nutrition: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    unknown = {}
    limits = moderation_limits(nutrition["nutrient_names"])
    for model in ("cvae", "gpt"):
        X, C, unknown[model] = meal_matrix(results[f"{model}_meals"], nutrition["feature_names"])
        for mt in MEAL_TYPES:
            if X[mt].shape[0] == 0:
                continue
            m = compute_nutrition_metrics_comprehensive(
                X[mt], nutrition["W_per100g"], nutrition["nutrient_names"], nutrition["RDI"],
                DEFAULT_MEAL_FRACTIONS[mt], limits=limits)
            macro = np.asarray(m["macro_shares"], dtype=float)
            pd.DataFrame({
                "cluster": C[mt],
                "energy_kcal": np.asarray(m["energy_kcal"], dtype=float),
                "macro_protein_pct": macro[:, 0], "macro_fat_pct": macro[:, 1], "macro_carb_pct": macro[:, 2],
                "amdr_within": np.asarray(m["amdr_within"]).astype(int),
                "MAR": np.asarray(m["MAR"], dtype=float),
                "MER": np.asarray(m["MER"], dtype=float),
                "diversity": np.asarray(m["diversity"], dtype=float),
            }).to_csv(out_dir / f"per_meal_metrics_{model}_{mt}.csv", index=False)
            rows.append(summary_row(model, mt, m))
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)

    comparison = {model: overall(summary, model) for model in ("cvae", "gpt")}
    comparison["food_codes_not_in_run"] = unknown
    comparison["gpt_meals_requested"] = len(results["gpt_meals"])
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))

    lines = [f"{'':24s}{'CVAE':>10s}{'GPT':>10s}"]
    for key, label in [("n_meals", "Meals evaluated"), ("amdr_compliant_pct", "AMDR compliant (%)"),
                       ("protein_pct", "Protein (% energy)"), ("fat_pct", "Fat (% energy)"),
                       ("carb_pct", "Carbohydrate (% energy)"), ("diversity_pct", "Diversity (%)")]:
        c, g = comparison["cvae"][key], comparison["gpt"][key]
        fmt = "{:>10,d}" if key == "n_meals" else "{:>10.1f}"
        lines.append(f"{label:24s}" + fmt.format(c) + fmt.format(g))
    lines.append(f"GPT meals requested: {len(results['gpt_meals']):,}")
    (out_dir / "comparison.txt").write_text("\n".join(lines) + "\n")
    return comparison


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="experiment_results.json from generate_meals.py.")
    ap.add_argument("--run_dir", default="outputs/runs/eval_run", help="Run folder with data_exports/ from `make eval`.")
    ap.add_argument("--out_dir", default=None, help="Default: <folder of --results>/evaluation.")
    args = ap.parse_args()
    # Empty-meal edge cases in the metric code give harmless divide/empty-slice warnings.
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    results_path = Path(args.results)
    results = json.loads(results_path.read_text())
    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent / "evaluation"
    evaluate(results, load_nutrition(Path(args.run_dir) / "data_exports"), out_dir)
    print((out_dir / "comparison.txt").read_text(), end="")
    print(f"Written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
