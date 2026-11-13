"""Helpers shared by the supplementary analyses."""
from __future__ import annotations

import json
import pickle
import zlib
from pathlib import Path
from typing import Dict

import numpy as np

MEAL_TYPES = ["Breakfast", "Lunch", "Dinner"]
MEAL_TYPE_MAPPING = {"Breakfast": 0, "Lunch": 1, "Dinner": 2}
# Share of daily energy per meal used for the per-meal RDI targets.
DEFAULT_MEAL_FRACTIONS = {"Breakfast": 0.25, "Lunch": 0.35, "Dinner": 0.40}


def load_exports(exports_dir: str | Path) -> dict:
    """Arrays that `make eval` writes to <run>/data_exports."""
    d = Path(exports_dir)
    return {
        "real_by_type": pickle.loads((d / "real_meals_by_type.pkl").read_bytes()),
        "gen_by_type": pickle.loads((d / "generated_meals_by_type.pkl").read_bytes()),
        "clusters_by_type": pickle.loads((d / "clusters_by_meal.pkl").read_bytes()),
        "feature_names": json.loads((d / "feature_names.json").read_text()),
        "nutrient_names": json.loads((d / "nutrient_names.json").read_text()),
        "code2name": json.loads((d / "food_code_to_name.json").read_text()),
        "food_to_group": json.loads((d / "food_to_group.json").read_text()),
        "W_per100g": np.load(d / "nutrient_matrix.npy"),
        "RDI": np.load(d / "RDI.npy"),
    }


def stable_seed(seed: int, *parts: str) -> int:
    """Seed offset that is identical in every process (Python's str hash() is salted)."""
    return int(seed) + zlib.crc32("|".join(parts).encode("utf-8")) % 10000


def subsample_per_cluster(clusters: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask keeping at most `cap` random meals per cluster (cap <= 0 keeps all)."""
    keep = np.zeros(clusters.shape[0], dtype=bool)
    for cid in np.unique(clusters):
        idx = np.where(clusters == cid)[0]
        if cap <= 0 or idx.size <= cap:
            keep[idx] = True
        else:
            keep[rng.choice(idx, size=cap, replace=False)] = True
    return keep


def cluster_level_medians(dev: np.ndarray, clusters: np.ndarray) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for cid in np.unique(clusters):
        m = clusters == cid
        if np.any(m):
            out[int(cid)] = float(np.median(dev[m]))
    return out


def cluster_reductions(real_med: Dict[int, float], other_med: Dict[int, float]) -> np.ndarray:
    """Per-cluster (real - other) / real, for clusters present in both."""
    vals = []
    for cid, r in real_med.items():
        o = other_med.get(cid)
        if o is None or not np.isfinite(r) or r == 0 or not np.isfinite(o):
            continue
        vals.append((r - o) / r)
    return np.asarray(vals, dtype=np.float64)


def median_percent_reduction(real_med: Dict[int, float], other_med: Dict[int, float]) -> float:
    """Median over clusters of (real - other) / real (a fraction, not a percentage)."""
    vals = cluster_reductions(real_med, other_med)
    return float(np.median(vals)) if vals.size else float("nan")


def portion_only_generator(exports: dict, meal_fractions: Dict[str, float], seed: int):
    """Generator without a model, used only to re-portion existing food sets."""
    from src.core.generator import Generator

    nutrient_names = exports["nutrient_names"]
    RDI = exports["RDI"]
    return Generator(
        model=None,
        feature_names=exports["feature_names"],
        pair_table={},
        meal_type_mapping=MEAL_TYPE_MAPPING,
        rng_seed=int(seed),
        nutrients_per_gram=(exports["W_per100g"].astype(np.float32) / 100.0),
        nutrient_names=nutrient_names,
        rdi_daily={str(n): float(RDI[i]) for i, n in enumerate(nutrient_names)},
        meal_energy_split={MEAL_TYPE_MAPPING[m]: float(meal_fractions[m]) for m in MEAL_TYPES},
        food_to_group=exports["food_to_group"],
        code2name=exports["code2name"],
        debug=False,
    )


def reportion(genr, present: np.ndarray, meal_type: str) -> np.ndarray:
    """Grams for each food-presence row, from the RDI-per-kcal portion optimizer."""
    genr._last_meals = np.full(present.shape[0], MEAL_TYPE_MAPPING[meal_type], dtype=int)
    return genr._portion_rdi_per_kcal(present, jitter_sigma=0.0)
