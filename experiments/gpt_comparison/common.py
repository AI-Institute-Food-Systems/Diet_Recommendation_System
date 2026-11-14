"""Shared pieces of the GPT comparison: run inputs, the CVAE generator and the allowed foods."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from analyses._common import MEAL_TYPE_MAPPING, MEAL_TYPES, load_exports

MEAL_NAMES = {idx: name for name, idx in MEAL_TYPE_MAPPING.items()}

# Per-meal targets given to GPT and used by the CVAE portioning (share of a 2,000 kcal day).
MEAL_ENERGY_SPLIT = {0: 0.25, 1: 0.35, 2: 0.40}
PROTEIN_MIN_G = {0: 20.0, 1: 25.0, 2: 28.0}
FIBER_MIN_G = {0: 7.5, 1: 8.5, 2: 9.5}
SODIUM_MAX_MG = {0: 450.0, 1: 700.0, 2: 800.0}
MIN_SOLID_ITEMS = {0: 2, 1: 3, 2: 3}
MAX_BEVERAGE_GRAMS = {0: 500, 1: 400, 2: 400}


def load_env_file(path: Path) -> None:
    """Read KEY=VALUE lines from a .env file into os.environ (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def load_pair_table(model_dir: Path, raw_cfg: dict) -> Dict[int, Tuple[int, int]]:
    """(meal index, cluster id) for each pair code, as `make eval` reads it."""
    path = model_dir / "pair_table.json"
    table = json.loads(path.read_text()) if path.exists() else raw_cfg.get("pair_table", {})
    if isinstance(table, list):
        table = dict(enumerate(table))
    return {int(k): (int(v[0]), int(v[1])) for k, v in table.items()}


def build_generator(run_dir: Path, model_dir: Path, seed: int):
    """The CVAE meal generator with the settings of `make eval`, plus the run's exports.

    Inputs come from `make cvae_train` (model_dir) and `make eval` (run_dir/data_exports).
    """
    from cli.evaluate import load_trained_cvae
    from src.core.generator import Generator

    exports = load_exports(run_dir / "data_exports")
    model, _cfg, raw_cfg = load_trained_cvae(model_dir, 1.0, 0.9)
    feature_names = [str(c).zfill(8) for c in json.loads((model_dir / "feature_names.json").read_text())]
    if feature_names != [str(c).zfill(8) for c in exports["feature_names"]]:
        raise ValueError(f"{model_dir} and {run_dir}/data_exports use different foods; rerun `make eval`.")

    nutrient_names = exports["nutrient_names"]
    RDI = exports["RDI"]
    energy_idx = nutrient_names.index("energy_kcal")
    rdi_per_kcal = {n: RDI[i] / RDI[energy_idx] for i, n in enumerate(nutrient_names)
                    if i != energy_idx and RDI[i] > 0}

    gen = Generator(
        model=model,
        feature_names=feature_names,
        pair_table=load_pair_table(model_dir, raw_cfg),
        meal_type_mapping=dict(MEAL_TYPE_MAPPING),
        rng_seed=seed,
        rdi_daily=RDI,
        food_to_group=exports["food_to_group"],
        max_beverage_grams=600,
        max_beverage_grams_by_meal=dict(MAX_BEVERAGE_GRAMS),
        expand_by_subcategory=False,
        nutrients_per_gram=exports["W_per100g"] / 100.0,
        nutrient_names=nutrient_names,
        rdi_per_kcal=rdi_per_kcal,
        nutrient_weights={},
        meal_energy_split=dict(MEAL_ENERGY_SPLIT),
        energy_name="energy_kcal",
        bev_energy_frac_max=0.25,
        debug=False,
    )
    return gen, exports


def allowed_foods(gen, exports: dict, pair_code: int) -> List[str]:
    """Foods the CVAE may use for this archetype (its decoder gate) that have nutrient data."""
    allowed = gen._gate[pair_code] < 0.5
    has_nutrients = exports["W_per100g"].sum(axis=1) > 0
    return [code for code, ok in zip(gen.feature_names, allowed & has_nutrients) if ok]


def meal_matrix(meals: List[dict], feature_names: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int]:
    """Gram matrices and cluster ids per meal type from successful meals; also the number of
    food codes that are not in feature_names (they are left out)."""
    col = {str(c).zfill(8): i for i, c in enumerate(feature_names)}
    rows = {mt: [] for mt in MEAL_TYPES}
    clusters = {mt: [] for mt in MEAL_TYPES}
    unknown = 0
    for meal in meals:
        if not meal["success"]:
            continue
        x = np.zeros(len(feature_names))
        for code, grams in meal["foods"].items():
            j = col.get(str(code).zfill(8))
            if j is None:
                unknown += 1
            else:
                x[j] += float(grams)
        mt = MEAL_NAMES[int(meal["meal_idx"])]
        rows[mt].append(x)
        clusters[mt].append(int(meal["cluster_id"]))
    X = {mt: np.array(rows[mt]).reshape(-1, len(feature_names)) for mt in MEAL_TYPES}
    C = {mt: np.array(clusters[mt], dtype=int) for mt in MEAL_TYPES}
    return X, C, unknown
