#!/usr/bin/env python3
"""Archetype table (S1 Table 6): for each of the 34 archetypes, the number of meals, the number
of foods, the top food categories and the largest absolute Cohen's d.

- Meals: meals with the archetype's cluster label (outputs/clustering/cached_clustering.pkl).
- Foods: foods eaten in at least one of those meals (the preprocessed meal matrix that
  clustering reads, data/processed/real_meals/dataframes/<meal>_bci_filtered.pkl).
- Top categories: up to --top_n WWEIA food categories whose grams are significantly higher in
  the archetype than in the other meals (FDR-controlled hurdle tests of `make clustering`),
  ranked by Cohen's d (outputs/clustering/feature_contribution_report.csv).
- Max |Cohen's d|: the largest absolute effect size among the archetype's significant features.

Inputs come from `make preprocess clustering`; `make sup-clustering` runs it.
Writes <out_dir>/archetype_table.csv.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MEAL_TYPES = ["Breakfast", "Lunch", "Dinner"]


def category_label(feature: str) -> str:
    """grams_sweetened_beverages -> Sweetened beverages."""
    return feature.removeprefix("grams_").replace("_", " ").capitalize()


def archetype_rows(meal_type: str, labels: np.ndarray, meals: pd.DataFrame, report: pd.DataFrame,
                   archetypes: list, names: dict, top_n: int) -> list:
    if len(labels) != len(meals):
        raise ValueError(f"{meal_type}: {len(labels)} cluster labels but {len(meals)} meals; "
                         "rerun `make clustering` after `make preprocess`.")
    present = meals.to_numpy() > 0
    rows = []
    for cid in archetypes:
        in_cluster = labels == cid
        feats = report[(report["meal_type"] == meal_type.lower()) & (report["cluster_id"] == cid)]
        enriched = feats[feats["feature"].str.startswith("grams_") & (feats["direction"] == "HIGH")]
        top = enriched.sort_values("effect_size", ascending=False)["feature"].head(top_n)
        rows.append({
            "meal_type": meal_type,
            "cluster_id": int(cid),
            "cluster_name": names[meal_type.lower()].get(str(cid), f"Cluster {cid}"),
            "n_meals": int(in_cluster.sum()),
            "n_foods": int(present[in_cluster].any(axis=0).sum()),
            "top_categories": "; ".join(category_label(f) for f in top),
            "max_abs_cohens_d": float(feats["effect_size"].abs().max()) if len(feats) else float("nan"),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clustering_dir", default="outputs/clustering")
    ap.add_argument("--meals_dir", default="data/processed/real_meals/dataframes")
    ap.add_argument("--archetypes", default="config/cluster_filter_default.json")
    ap.add_argument("--names", default="config/cluster_names.json")
    ap.add_argument("--top_n", type=int, default=5, help="Food categories listed per archetype.")
    ap.add_argument("--out_dir", default="outputs/runs/eval_run/supplementary/archetype_table")
    args = ap.parse_args()

    clustering_dir = Path(args.clustering_dir)
    with open(clustering_dir / "cached_clustering.pkl", "rb") as f:  # trusted pipeline output
        clusters = pickle.load(f)["meal_type_clusters"]
    report = pd.read_csv(clustering_dir / "feature_contribution_report.csv")
    archetypes = json.loads(Path(args.archetypes).read_text())
    names = json.loads(Path(args.names).read_text(encoding="utf-8"))

    rows = []
    for mt in MEAL_TYPES:
        labels = np.asarray(clusters[mt.lower()]["labels"])
        meals = pd.read_pickle(Path(args.meals_dir) / f"{mt.lower()}_bci_filtered.pkl")
        rows += archetype_rows(mt, labels, meals, report, archetypes[mt], names, args.top_n)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "archetype_table.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(table.drop(columns="cluster_id").to_string(index=False, float_format="%.2f"))
    print(f"Written to {out_dir / 'archetype_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
