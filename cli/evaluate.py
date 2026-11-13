#!/usr/bin/env python3
r"""Generate meals with the trained CVAE and compare them with real meals (Figure 3).

`make eval` runs (default Makefile variables, paths relative to the repository root)

    python3 cli/evaluate.py --project_root . \
        --model_dir outputs/runs/eval_run/presence_only_model --out_dir outputs/runs/eval_run \
        --portion_strategy rdi_per_kcal --max_beverages 1 --presence_to_grams rdi_per_kcal \
        --top_k 10 --min_prob 0.10 --deterministic --force_regen \
        --price_map data/prices/price_map.csv --seed 42 --cluster_filter config/cluster_filter_default.json

It loads the archetype meals (outputs/clustering/<meal>/cluster_*_training_data.pkl), generates
one meal per real meal of each archetype with src/core/generator.py, portions it toward the
per-meal RDI targets (rdi_per_kcal), aligns real and generated meals by archetype, and writes
under --out_dir:

- data_exports/: real and generated meals, nutrient matrix, RDI and name maps. Without
  --force_regen a later run reuses them instead of generating again (`make eval FORCE_REGEN=0`).
- tables/: cluster_counts.csv, cluster_counts_totals.csv, cost_health_pair_metrics.csv.
- figures/: the Figure 3 panels (src/pipeline_eval/visualization.py).

stdout and stderr are redirected to logs/evaluate/evaluate.log under the working directory.
"""
from __future__ import annotations

# Must be before importing pyplot
import matplotlib as _mpl
_mpl.use("Agg")

# The script unpickles files from the clustering and run output folders; use trusted files only.

import argparse
import json
import os
from pathlib import Path
from collections import Counter
import sys
import warnings
from typing import Dict, List, Tuple, Optional

# Ensure repository root is importable when running as a script
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# Suppress CUDA NUMA node warnings and TensorFlow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage to avoid CUDA warnings

# Additional TensorFlow logging suppression
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def _load_cluster_name_map() -> dict:
    """Load cluster names from config/cluster_names.json.
    Returns mapping: { meal_type_lower: { cluster_id:int -> name:str } }.
    """
    try:
        p = Path("config/cluster_names.json")
        if not p.exists():
            return {}
        j = json.loads(p.read_text(encoding="utf-8"))
        out = {}
        for mt, d in j.items():
            # keys are strings of ints in the file
            out[mt.lower()] = {int(k): str(v) for k, v in d.items()}
        return out
    except Exception:
        return {}


_CLUSTER_NAME_MAP = _load_cluster_name_map()


def _get_cluster_name(meal_type: str, cluster_id: int) -> str:
    """Archetype name from config/cluster_names.json; fallback to "Cluster {id}"."""
    d = _CLUSTER_NAME_MAP.get(str(meal_type).lower()) or {}
    return d.get(int(cluster_id), f"Cluster {int(cluster_id)}")

# Import manager classes
from src.pipeline_eval.evaluation import EvaluationManager
from src.pipeline_eval.visualization import VisualizationManager, create_spider_plots

import numpy as np
import pandas as pd

try:
    import scipy.sparse as sp
    _SP_AVAILABLE = True
except Exception:
    _SP_AVAILABLE = False

# --- Evaluation helpers ---
from src.analysis_rdi_eval.io_utils import (
    set_random_seeds,
    validate_data_contracts,
)

# --- CVAE model classes (needed to load the trained model) ---
from src.utils.multi_meal_cvae import MultiMealCVAEConfig, Log1pZScore
from src.utils.multi_meal_presence_cvae import MultiMealPresenceOnlyCVAE, PresenceOnlyCVAELossLayer, sampling
from src.utils.helpers import setup_logger as _setup_logger

# --- logger redirection (match preprocessing style) ---
class _StreamToLogger:
    def __init__(self, logger: logging.Logger, level: int):
        self._logger = logger
        self._level = level

    def write(self, buf: str):
        for line in buf.rstrip().splitlines():
            self._logger.log(self._level, line.rstrip())

    def flush(self):
        pass


MEAL_TYPES = ["Breakfast", "Lunch", "Dinner"]
MEAL_TO_IDX = {"Breakfast": 0, "Lunch": 1, "Dinner": 2}

# kcal factors used whenever we need to synthesize energy
KCAL_SPEC_CANON = {"protein_g": 4.0, "carb_g": 4.0, "fat_g": 9.0, "sfa_g": 9.0}

# --- Serving sizes & fast presence->grams mapping ---

def load_serving_sizes_any(project_root: Path) -> pd.DataFrame:
    """
    Load a serving size table. Expected columns:
      - 'Food code' (8-digit string or int)
      - 'Serving_g' (float, grams per serving)
    Falls back to 100 g if not found. Only reported in the log: the Generator does its own portioning.
    """
    # Try csv then pkl
    csv_path = project_root / "data" / "mappings" / "food_servings.csv"
    pkl_path = project_root / "data" / "mappings" / "food_servings.pkl"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif pkl_path.exists():
        df = pd.read_pickle(pkl_path)
    else:
        # minimal fallback
        return pd.DataFrame({"Food code": [], "Serving_g": []})

    if "Food code" not in df.columns:
        raise ValueError("Serving size table must contain 'Food code' column.")
    # pick a column for grams
    grams_col = None
    for cand in ["Serving_g", "serving_g", "Serving (g)", "serving_grams"]:
        if cand in df.columns:
            grams_col = cand
            break
    if grams_col is None:
        raise ValueError("Serving size table must contain a grams column like 'Serving_g'.")

    df = df.copy()
    df["Food code"] = df["Food code"].astype(str).str.zfill(8)
    df = df.rename(columns={grams_col: "Serving_g"})
    return df[["Food code", "Serving_g"]]


def build_serving_vector(serving_df: pd.DataFrame, feature_codes: list[str], default_grams: float = 100.0) -> np.ndarray:
    """
    Build a vector 'S' of length = len(feature_codes) where S[j] is grams per serving for that food.
    Missing foods default to 'default_grams'.
    """
    S = np.full(len(feature_codes), float(default_grams), dtype=np.float32)
    if serving_df is not None and not serving_df.empty:
        m = serving_df.set_index("Food code")["Serving_g"]
        for j, code in enumerate(feature_codes):
            if code in m.index and np.isfinite(m.loc[code]):
                S[j] = float(m.loc[code])
    return S


def _looks_like_probabilities(X: np.ndarray) -> bool:
    """Heuristic: matrix is likely probabilities (not grams) if most entries are in [0,1]
    and per-row gram sums are tiny."""
    if X.size == 0:
        return False
    X = np.asarray(X, dtype=np.float32)
    frac_01 = np.mean((X >= 0.0) & (X <= 1.0))
    row_sums = np.sum(X, axis=1) if X.ndim == 2 else np.array([np.sum(X)])
    med_sum = float(np.median(row_sums)) if row_sums.size else 0.0
    # Almost everything in [0,1] and meals summing to less than a small grams threshold
    return frac_01 > 0.98 and med_sum < 20.0


def _coerce_presence_probs_to_grams(
    X_prob_by_type: Dict[str, np.ndarray],
    *,
    policy: str,                     # "rdi_per_kcal" only
    feature_names: List[str],
    code2name: Dict[str, str],
    W: np.ndarray,
    nutrient_names: List[str],
    RDI: np.ndarray,
    serving_vec: np.ndarray,
    food_to_group: Dict[str, str],
    price_vector: Optional[np.ndarray] = None,
    cost_weight: float = 0.0,
    seed: int = 42,
    top_k: int = 12,
    min_prob: float = 0.02,
    deterministic: bool = False,
) -> Dict[str, np.ndarray]:
    """Convert presence probabilities to grams for each meal type using rdi_per_kcal.

    Only used when cached generations are still probabilities. Matrices that already hold
    grams are returned unchanged. serving_vec, price_vector, cost_weight and seed are unused.
    """
    out: Dict[str, np.ndarray] = {}

    for mt, Xp in X_prob_by_type.items():
        if not _looks_like_probabilities(Xp):
            out[mt] = Xp  # already grams
            continue

        if policy != "rdi_per_kcal":
            raise ValueError(f"Unknown policy for presence_to_grams: {policy}")

        # rdi_per_kcal cached conversion: threshold + top_k → present mask, then portion with Generator
        P = np.clip(Xp, 1e-6, 1 - 1e-6)
        N, D = P.shape
        mask = (P >= float(min_prob)).astype(bool)
        if top_k is not None and top_k > 0 and top_k < D:
            for i in range(N):
                if not mask[i].any():
                    jmax = int(np.argmax(P[i]))
                    mask[i, jmax] = True
                sel = np.where(mask[i])[0]
                if sel.size > top_k:
                    keep = sel[np.argpartition(P[i, sel], -top_k)[-top_k:]]
                    row = np.zeros(D, dtype=bool)
                    row[keep] = True
                    mask[i] = row

        # Build rdi_per_kcal densities from RDI/day
        try:
            e_idx = nutrient_names.index("energy_kcal")
            daily_energy = float(RDI[e_idx])
        except ValueError:
            daily_energy = 2000.0
        rdi_per_kcal = {}
        if daily_energy and np.isfinite(daily_energy) and daily_energy > 0:
            for j, name in enumerate(nutrient_names):
                if name == "energy_kcal":
                    continue
                val = float(RDI[j]) if j < len(RDI) else 0.0
                if np.isfinite(val) and val > 0:
                    rdi_per_kcal[name] = val / daily_energy

        from src.core.generator import Generator as _Gen
        meal_map = {m: MEAL_TO_IDX[m] for m in MEAL_TYPES}
        nutr_per_g = W / 100.0
        gen = _Gen(
            model=None,
            feature_names=feature_names,
            pair_table={},
            meal_type_mapping=meal_map,
            nutrients_per_gram=nutr_per_g,
            nutrient_names=nutrient_names,
            rdi_per_kcal=rdi_per_kcal,
            rdi_daily={n: float(RDI[i]) for i, n in enumerate(nutrient_names)} if len(RDI) == len(nutrient_names) else {},
            meal_energy_split={0: 0.25, 1: 0.35, 2: 0.40},
            energy_name="energy_kcal",
            food_to_group=food_to_group,
            bev_energy_frac_max=0.25,
            code2name=code2name,
            debug=False,
        )
        meal_idx = np.full(N, MEAL_TO_IDX[mt], dtype=np.int32)
        grams = gen._portion_rdi_per_kcal(mask, min_grams=5, jitter_sigma=0.0 if deterministic else 0.10, meal_indices=meal_idx, use_energy_targets=True)
        out[mt] = grams

    return out


def load_food_nutrient_table_any(project_root: Path) -> pd.DataFrame:
    """
    Load data/mappings/food_to_nutrient.pkl if present, otherwise the .csv.
    Expects a 'Food code' column; values are per 100 g. Used here for food names and
    WWEIA food groups (nutrient values come from src/core/nutrition_harmonizer.py).
    """
    csv_path = project_root / "data" / "mappings" / "food_to_nutrient.csv"
    pkl_path = project_root / "data" / "mappings" / "food_to_nutrient.pkl"
    if pkl_path.exists():
        df = pd.read_pickle(pkl_path)
    elif csv_path.exists():
        try:
            df = pd.read_csv(csv_path, engine='python')
        except Exception as e:
            print(f"[WARN] CSV parsing failed: {e}. Falling back to PKL if available.")
            if pkl_path.exists():
                df = pd.read_pickle(pkl_path)
            else:
                raise FileNotFoundError(f"Missing nutrient mapping: {csv_path} or {pkl_path}")
    else:
        raise FileNotFoundError(f"Missing nutrient mapping: {csv_path} or {pkl_path}")
    if "Food code" not in df.columns:
        raise ValueError("food_to_nutrient table must contain a 'Food code' column.")
    return df


def build_foodgroup_map(food_df: pd.DataFrame, feature_codes: list[str]) -> Dict[str, str]:
    """Build {8-digit code -> canonical group} from the WWEIA main category of each food
    (the food-ingredient mapping, data/mappings/food_ingredient_map.pkl, has this column).
    Canonical groups: dairy, protein, grains, fruits, vegetables, beverages,
    alcoholic_beverages, snacks_sweets, mixed_dishes, condiments_sauces,
    sugars, fats_oils, baby_foods, other.
    """
    df = food_df.copy()
    if "Food code" not in df.columns:
        return {}
    df["Food code"] = df["Food code"].astype(str).str.zfill(8)

    # Candidate columns that may hold WWEIA main category descriptions
    cand_cols = [
        "WWEIA Main Category description",
        "WWEIA Main Category",
        "Main Category",
        "WWEIA_Main_Category",
    ]
    col = next((c for c in cand_cols if c in df.columns), None)
    if col is None:
        return {}

    def to_canonical(main_desc: str) -> str:
        s = str(main_desc or "").lower()
        if not s:
            return "other"
        if "alcohol" in s:
            return "alcoholic_beverages"
        if "milk" in s or "dairy" in s:
            return "dairy"
        if "meat" in s or "poultry" in s or "fish" in s or "egg" in s or "protein" in s or "seafood" in s:
            return "protein"
        if "grain" in s or "bread" in s or "cereal" in s or "rice" in s or "pasta" in s:
            return "grains"
        if "fruit" in s:
            return "fruits"
        if "vegetable" in s or "veg" in s:
            return "vegetables"
        if "beverage" in s or "drink" in s or "coffee" in s or "tea" in s or "water" in s or "juice" in s:
            return "beverages"
        if "mixed" in s or "sandwich" in s or "pizza" in s or "soup" in s:
            return "mixed_dishes"
        if "snack" in s or "sweet" in s or "dessert" in s or "candy" in s or "bakery" in s:
            return "snacks_sweets"
        if "condiment" in s or "sauce" in s:
            return "condiments_sauces"
        if "sugar" in s:
            return "sugars"
        if "fat" in s or "oil" in s:
            return "fats_oils"
        if "baby" in s or "infant" in s:
            return "baby_foods"
        return "other"

    groups = df[["Food code", col]].copy()
    groups["group"] = groups[col].map(to_canonical)
    present = groups[groups["Food code"].isin(feature_codes)]
    return dict(zip(present["Food code"], present["group"]))
def build_foodname_map(food_df: pd.DataFrame) -> Dict[str, str]:
    """
    Create {8-digit food code -> readable name}. Tries common name columns.
    Falls back to code itself if name missing.
    """
    df = food_df.copy()
    df["Food code"] = df["Food code"].astype(str).str.zfill(8)

    # Heuristic: likely name columns in USDA-style tables
    name_cols = [
        "Food name", "Main food description", "Description",
        "Food description", "Display Name", "Shrt_Desc", "long_desc", "short_desc"
    ]
    name_col = None
    for c in name_cols:
        if c in df.columns:
            name_col = c
            break

    if name_col is None:
        # if none exist, synthesize simple names
        return {code: code for code in df["Food code"]}

    names = df[name_col].astype(str).fillna("").str.strip()
    # Clean up multi-line names
    names = names.str.replace(r"\s+", " ", regex=True)
    return dict(zip(df["Food code"], names))


def fast_meal_nutrients_per_meal(X_grams: np.ndarray, W_per100g: np.ndarray, prefer_sparse: bool = True) -> np.ndarray:
    """
    Compute per-meal nutrients as (X/100) @ W_per100g with optional sparse acceleration.
    Returns a dense array (N x M).
    """
    if X_grams.size == 0:
        return np.zeros((0, W_per100g.shape[1]), dtype=np.float32)

    if prefer_sparse and _SP_AVAILABLE:
        X = sp.csr_matrix(X_grams, dtype=np.float32)
        W = sp.csr_matrix(W_per100g.astype(np.float32))
        N = (X.multiply(1/100.0)).dot(W).toarray()
        return N.astype(np.float32, copy=False)
    else:
        return ((X_grams.astype(np.float32, copy=False) / 100.0) @ W_per100g.astype(np.float32, copy=False))


# -----------------------
# Helpers: Data discovery
# -----------------------
def _clustering_pickles_root(project_root: Path) -> Path:
    """Root dir containing ``<meal_type>/cluster_*_training_data.pkl`` (written by `make clustering`)."""
    return project_root / "outputs" / "clustering"


def find_cluster_pickles(project_root: Path) -> Dict[str, List[Path]]:
    """Return {meal_type: [paths...]} for all cluster_*_training_data.pkl files."""
    out: Dict[str, List[Path]] = {mt: [] for mt in MEAL_TYPES}
    base = _clustering_pickles_root(project_root)
    for mt in MEAL_TYPES:
        d = base / mt.lower()
        if not d.exists():
            warnings.warn(f"Cluster dir missing for meal_type={mt}: {d}")
            continue
        for p in sorted(d.glob("cluster_*_training_data.pkl")):
            out[mt].append(p)
    return out


def load_cluster_df(pkl_path: Path) -> pd.DataFrame:
    """Load a single cluster dataframe and add cluster_id from filename."""
    df = pd.read_pickle(pkl_path)
    
    # Extract cluster ID from filename (e.g., cluster_0_training_data.pkl -> 0)
    cluster_id = int(pkl_path.stem.split('_')[1])
    
    # Add cluster_id column if not present
    if "cluster_id" not in df.columns:
        df = df.copy()
        df["cluster_id"] = cluster_id
    
    return df


def reindex_to_global(df: pd.DataFrame, global_codes: List[str]) -> pd.DataFrame:
    """Return df with only food-code columns reindexed (others dropped), filling missing with 0."""
    cols = [c for c in df.columns if c != "cluster_id"]
    df_grams = df[cols].copy()
    # Ensure string 8-digit format for matching
    df_grams.columns = [f"{int(c):08d}" if str(c).isdigit() else str(c) for c in df_grams.columns]
    aligned = pd.DataFrame(0.0, index=df.index, columns=global_codes, dtype=np.float32)
    common = [c for c in df_grams.columns if c in aligned.columns]
    aligned.loc[:, common] = df_grams[common].astype(np.float32).values
    aligned["cluster_id"] = df["cluster_id"].values
    return aligned


# --------------------------------------
# Helpers: Model loading & meal generate
# --------------------------------------
def _load_hdf5_model(file_path: Path, custom_objects: dict = None):
    """Load a model from HDF5 file (even if it has .keras extension)."""
    import tensorflow as tf
    import h5py
    import tempfile
    import shutil
    from src.utils.multi_meal_cvae import Log1pZScore
    
    try:
        # First try as Keras format
        return tf.keras.models.load_model(str(file_path), compile=False, custom_objects=custom_objects)
    except Exception as e1:
        print(f"Keras loading failed: {e1}")
        # If that fails, try as HDF5 by copying to .h5 extension
        try:
            with h5py.File(str(file_path), 'r') as f:
                # Check if it's a full model or just weights
                if 'model_config' in f.attrs or 'model_config' in f:
                    # It's a full model, try loading with .h5 extension
                    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                        shutil.copy2(str(file_path), tmp_file.name)
                        try:
                            # Add more custom objects for HDF5 loading
                            extended_custom_objects = custom_objects or {}
                            extended_custom_objects.update({
                                'TFOpLambda': tf.keras.layers.Lambda,
                                'CVAEModelWithMetrics': tf.keras.Model,
                                'Log1pZScore': Log1pZScore,
                            })
                            model = tf.keras.models.load_model(tmp_file.name, compile=False, custom_objects=extended_custom_objects, safe_mode=False)
                            return model
                        finally:
                            import os
                            os.unlink(tmp_file.name)
                else:
                    # It's just weights, return None to indicate we need the architecture
                    return None
        except Exception as e2:
            print(f"Warning: Could not load {file_path} as HDF5: {e2}")
            return None

def load_trained_cvae(model_dir: Path, generation_temp: float = 1.0, generation_nucleus_p: float = 0.8) -> Tuple[Optional[MultiMealPresenceOnlyCVAE], MultiMealCVAEConfig, dict]:
    """Load the trained presence-only CVAE; return (model or None, config_obj, raw_config_dict)."""
    presence_cfg_paths = [
        model_dir / "multi_meal_presence_cvae_config.json",
        model_dir / "presence_only_model" / "multi_meal_presence_cvae_config.json",
    ]
    
    for presence_cfg_path in presence_cfg_paths:
        if presence_cfg_path.exists():
            print("Loading presence-only CVAE model...")
            with open(presence_cfg_path, "r") as f:
                raw_cfg = json.load(f)
            
            # Extract input_dim from encoder_preproc if available
            input_dim = raw_cfg.get("input_dim", 0)
            if input_dim == 0 and "encoder_preproc" in raw_cfg:
                encoder_preproc = raw_cfg["encoder_preproc"]
                if "mean" in encoder_preproc:
                    input_dim = len(encoder_preproc["mean"])
            
            # Extract num_clusters from pair_table if available
            num_clusters = raw_cfg.get("num_clusters", 0)
            if num_clusters == 0 and "pair_table" in raw_cfg:
                num_clusters = len(raw_cfg["pair_table"])
            
            # choose the inner cvae subconfig if present; otherwise treat raw_cfg as that subconfig
            cvae_subcfg = raw_cfg.get("cvae_model", raw_cfg).copy()
            cvae_subcfg["input_dim"] = input_dim
            cvae_subcfg["num_clusters"] = num_clusters
            
            cfg = MultiMealCVAEConfig({"cvae_model": cvae_subcfg})
            
            # Add generation parameters to config
            cfg.generation_temp = generation_temp
            cfg.generation_nucleus_p = generation_nucleus_p
            
            try:
                # Try loading from the same directory as the config
                model_dir_to_try = presence_cfg_path.parent
                # Add custom objects for loading
                from src.utils.multi_meal_presence_cvae import CondDrop
                custom_objects = {'Log1pZScore': Log1pZScore, 'CondDrop': CondDrop}
                
                # Try the original load method first
                try:
                    model = MultiMealPresenceOnlyCVAE.load(
                        str(model_dir_to_try),
                        raw_config=raw_cfg,
                        custom_objects=custom_objects
                    )
                    print(f"Loaded presence-only model from: {model_dir_to_try}")
                except Exception as e1:
                    print(f"Original load failed ({e1}), trying HDF5 fallback...")
                    # Fallback: manually load HDF5 files
                    encoder_path = model_dir_to_try / "encoder.keras"
                    decoder_path = model_dir_to_try / "decoder.keras"
                    cvae_path = model_dir_to_try / "cvae.keras"
                    
                    if encoder_path.exists() and decoder_path.exists() and cvae_path.exists():
                        from src.utils.multi_meal_presence_cvae import CondDrop
                        extended = {
                            'Log1pZScore': Log1pZScore,
                            'PresenceOnlyCVAELossLayer': PresenceOnlyCVAELossLayer,
                            'sampling': sampling,
                            'CondDrop': CondDrop,
                        }
                        model = MultiMealPresenceOnlyCVAE(cfg, raw_cfg)  # construct shell
                        model.encoder = _load_hdf5_model(encoder_path, extended)
                        model.decoder = _load_hdf5_model(decoder_path, extended)
                        model.cvae = _load_hdf5_model(cvae_path, extended)
                        
                        if model.encoder is None or model.decoder is None or model.cvae is None:
                            raise Exception("Could not load model components as HDF5")
                        
                        # Load calibration if available
                        calib_path = model_dir_to_try / "calibration.json"
                        if calib_path.exists():
                            with open(calib_path, "r") as f:
                                calib = json.load(f)
                            model.import_calibration(calib, freeze=True)
                        
                        print(f"Loaded presence-only model from HDF5 files: {model_dir_to_try}")
                    else:
                        raise e1
                        
            except Exception as e:
                raise RuntimeError(f"Could not load the presence-only model from {model_dir_to_try} "
                                   f"({type(e).__name__}: {e})") from e
            return model, cfg, raw_cfg
    
    raise FileNotFoundError(
        f"No multi_meal_presence_cvae_config.json in {model_dir} (run `make cvae_train` first).")


# ------------------------
# Caching of data exports
# ------------------------
def save_exports(
    out_dir: Path,
    real_by_type: Dict[str, np.ndarray],
    gen_by_type: Dict[str, np.ndarray],
    clusters_real_by_type: Dict[str, np.ndarray],
    clusters_gen_by_type: Dict[str, np.ndarray],
    W: np.ndarray,
    nutrient_names: List[str],
    RDI: np.ndarray,
    feature_names: List[str],
    food_to_group: Dict[str, str],
    code2name: Dict[str, str],
    gen_row_indices_by_type: Dict[str, np.ndarray] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Pickles
    pd.to_pickle(real_by_type, out_dir / "real_meals_by_type.pkl")
    pd.to_pickle(gen_by_type, out_dir / "generated_meals_by_type.pkl")
    pd.to_pickle(clusters_real_by_type, out_dir / "clusters_by_meal.pkl")
    pd.to_pickle(clusters_gen_by_type, out_dir / "gen_clusters_by_meal.pkl")
    if gen_row_indices_by_type is not None:
        pd.to_pickle(gen_row_indices_by_type, out_dir / "gen_row_indices_by_meal.pkl")

    # Arrays / JSON
    np.save(out_dir / "nutrient_matrix.npy", W)
    (out_dir / "nutrient_names.json").write_text(json.dumps(nutrient_names, indent=2))
    np.save(out_dir / "RDI.npy", RDI)
    (out_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    (out_dir / "food_to_group.json").write_text(json.dumps(food_to_group, indent=2))
    (out_dir / "food_code_to_name.json").write_text(json.dumps(code2name, ensure_ascii=False, indent=2))


def maybe_load_exports(out_dir: Path) -> Optional[Dict]:
    """Load an existing run's data_exports/ folder (`out_dir`); None if a required file is missing."""
    exports_dir = out_dir
    
    # Check if exports directory exists
    if not exports_dir.exists():
        print(f"[CACHE] Exports directory does not exist: {exports_dir}")
        return None
    
    files = {
        "real": exports_dir / "real_meals_by_type.pkl",
        "gen": exports_dir / "generated_meals_by_type.pkl",
        "clusters_real": exports_dir / "clusters_by_meal.pkl",
        "clusters_gen": exports_dir / "gen_clusters_by_meal.pkl",
        "nutrient_matrix": exports_dir / "nutrient_matrix.npy",
        "nutrient_names": exports_dir / "nutrient_names.json",
        "RDI": exports_dir / "RDI.npy",
        "feature_names": exports_dir / "feature_names.json",
        "food_to_group": exports_dir / "food_to_group.json",
        "code2name": exports_dir / "food_code_to_name.json",
        "gen_idx": exports_dir / "gen_row_indices_by_meal.pkl",
    }
    required = ["real", "gen", "clusters_real", "nutrient_matrix", "nutrient_names", "RDI", "feature_names"]
    missing_files = [k for k in required if not files[k].exists()]
    if missing_files:
        print(f"[CACHE] Missing required files: {missing_files}")
        return None

    loaded = {
        "real_meals": pd.read_pickle(files["real"]),
        "generated_meals": pd.read_pickle(files["gen"]),
        "clusters_real": pd.read_pickle(files["clusters_real"]),
        "clusters_gen": pd.read_pickle(files["clusters_gen"]) if files["clusters_gen"].exists() else None,
        "gen_row_indices_by_meal": pd.read_pickle(files["gen_idx"]) if files["gen_idx"].exists() else None,
        "W": np.load(files["nutrient_matrix"]),
        "nutrient_names": json.loads(files["nutrient_names"].read_text()),
        "RDI": np.load(files["RDI"]),
        "feature_names": json.loads(files["feature_names"].read_text()),
        "food_to_group": json.loads(files["food_to_group"].read_text()) if files["food_to_group"].exists() else {},
        "code2name": json.loads(files["code2name"].read_text()) if files["code2name"].exists() else None,
    }
    
    return loaded


def check_pair(model, gate, pair_code, train_pair_table, feature_names, top_k=10):
    """Check a pair to catch milk-only failures early.

    Prints the number of allowed foods and the top-k foods of 8 decoded samples. The latent
    samples come from NumPy's global RNG (np.random.randn).
    """
    import numpy as np
    
    allowed = (gate[pair_code] < 0.5)           # boolean mask over D
    n_allowed = int(allowed.sum())
    
    # Decode a few z's
    z = np.random.randn(8, model.latent_dim).astype(np.float32)
    Cb = np.full(8, pair_code, np.int32)
    Mb = np.full(8, train_pair_table[pair_code][0], np.int32)  # meal_idx
    P = model.predict_heads_from_z(z, Cb, Mb)[0]              # (8, D)

    top = np.argsort(P.mean(0))[::-1][:top_k]
    print(f"[check] pair={pair_code} allowed={n_allowed}  top@{top_k} feature idx={top}")
    
    return n_allowed, top


def _load_cluster_filter(path: Path | None) -> dict[str, set[int]]:
    """Returns {"__keep__": {meal: ids}, "__omit__": {meal: ids}}, or {} without a filter file."""
    if path is None or not path.exists():
        return {}
    j = json.loads(path.read_text())

    # Supported shapes:
    # (a) {"Breakfast":[...], "Lunch":[...], "Dinner":[...]}          -> keep lists
    # (b) {"keep": {"Breakfast":[...]}, "omit": {"Lunch":[...]}}      -> keep/omit
    # (c) {"omit": {...}} only                                        -> omit only

    def to_set(x): return set(int(v) for v in x)

    keep: dict[str, set[int]] = {}
    omit: dict[str, set[int]] = {}
    if "keep" in j or "omit" in j:
        if "keep" in j:
            for mt, lst in j["keep"].items():
                keep[mt] = to_set(lst)
        if "omit" in j:
            for mt, lst in j["omit"].items():
                omit[mt] = to_set(lst)
    else:
        # treat top-level mapping as keep
        for mt, lst in j.items():
            if mt in MEAL_TYPES:
                keep[mt] = to_set(lst)

    # If only OMITS given, we'll compute KEEP later when we know which clusters exist.
    return {"__keep__": keep, "__omit__": omit}


def _allowed_clusters_for_meal(pair_table: dict, meal_idx: int) -> set[int]:
    """Get the set of cluster IDs that exist in pair_table for a given meal type."""
    return {cid for (mi, cid) in pair_table.values() if mi == meal_idx}

def align_for_evaluation(
    real_by_type: Dict[str, np.ndarray],
    gen_by_type: Dict[str, np.ndarray],
    clusters_by_type_real: Dict[str, np.ndarray],
    clusters_by_type_gen: Optional[Dict[str, np.ndarray]] = None,
    gen_row_indices_by_meal: Optional[Dict[str, np.ndarray]] = None,
    drop_noise: bool = True,
    pair_table: Optional[Dict] = None,
    meal_type_mapping: Optional[Dict] = None,
    cluster_filter: Optional[Dict] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Return (real_aligned, gen_aligned, clusters_aligned) with identical lengths and
    row order per meal type. Robust to stale/mismatched cached indices.
    """
    # Fallback meal type mapping if not provided
    if meal_type_mapping is None:
        meal_type_mapping = {"Breakfast": 0, "Lunch": 1, "Dinner": 2}
    
    real_aligned, gen_aligned, clusters_aligned = {}, {}, {}
    mt_keys = [mt for mt in MEAL_TYPES if mt in real_by_type and mt in gen_by_type]

    for mt in mt_keys:
        Xr_all = real_by_type[mt]
        Xg     = gen_by_type[mt]
        Cr_all = clusters_by_type_real.get(mt)
        if Cr_all is None:
            raise ValueError(f"Missing clusters for real data in meal_type={mt}")

        # Apply cluster_filter keep/omit by meal unconditionally first
        if cluster_filter is not None:
            keep_map = cluster_filter.get("__keep__") or {}
            omit_map = cluster_filter.get("__omit__") or {}
            keep_set = set(keep_map.get(mt, []))
            omit_set = set(omit_map.get(mt, []))
            if keep_set:
                mask_keep = np.isin(Cr_all, list(keep_set))
            else:
                mask_keep = np.ones_like(Cr_all, dtype=bool)
            if omit_set:
                mask_omit = ~np.isin(Cr_all, list(omit_set))
            else:
                mask_omit = np.ones_like(Cr_all, dtype=bool)
            allowed_mask = mask_keep & mask_omit
            Xr_all = Xr_all[allowed_mask]
            Cr_all = Cr_all[allowed_mask]

        # Additionally gate by pair_table if provided
        meal_idx = meal_type_mapping.get(mt)
        if meal_idx is not None and pair_table is not None:
            allowed_clusters = _allowed_clusters_for_meal(pair_table, meal_idx)
            allowed_mask = np.isin(Cr_all, list(allowed_clusters))
            Xr_all = Xr_all[allowed_mask]
            Cr_all = Cr_all[allowed_mask]

        n_real = len(Cr_all)
        valid_real = (Cr_all != -1) if drop_noise else np.ones(n_real, dtype=bool)

        # --- try using cached indices first ---
        idx = None
        if gen_row_indices_by_meal is not None and mt in gen_row_indices_by_meal:
            idx_raw = np.asarray(gen_row_indices_by_meal[mt], dtype=int)

            # bounds-check against current real length
            in_bounds = (idx_raw >= 0) & (idx_raw < n_real)
            if not np.all(in_bounds):
                bad = int((~in_bounds).sum())
                print(f"[align] {mt}: dropping {bad} cached indices out of bounds; "
                      f"cache/data mismatch detected.")

            idx_raw = idx_raw[in_bounds]

            # optionally drop noise
            if drop_noise:
                keep = valid_real[idx_raw]
                idx = idx_raw[keep]
            else:
                idx = idx_raw

            # ensure length matches generated rows
            if len(idx) != len(Xg):
                print(f"[align] {mt}: cached indices {len(idx)} != generated {len(Xg)}; re-aligning by clusters. "
                      f"Tip: re-run with --force_regen if results look odd.")
                idx = None  # force rebuild

        # --- rebuild mapping if no usable cache ---
        if idx is None:
            # Prefer explicit gen clusters if provided and sized correctly
            Cg = None
            if clusters_by_type_gen is not None and mt in clusters_by_type_gen:
                Cg = np.asarray(clusters_by_type_gen[mt], dtype=int)
                if len(Cg) != len(Xg):
                    print(f"[align] {mt}: gen cluster series length {len(Cg)} != generated {len(Xg)}; ignoring.")
                    Cg = None

            # If no gen clusters, assume generator followed filtered real order
            if Cg is None:
                base = Cr_all[valid_real]
                if base.size == 0:
                    raise AssertionError(f"[align] {mt}: no valid real clusters to align against.")
                reps = int(np.ceil(len(Xg) / base.size))
                Cg = np.tile(base, reps)[:len(Xg)]
            
            # Validate cluster multiset matches real data; if not, auto-heal by trimming to min counts
            from collections import Counter
            Cr_valid = Cr_all[valid_real]
            cr, cg = Counter(Cr_valid.tolist()), Counter(Cg.tolist())
            if cr != cg:
                # Build a healed sequence Cg' that matches the real multiset by trimming extras
                healed = []
                need = cr.copy()
                for k in list(cg.keys()):
                    if k not in need:
                        need[k] = 0
                for k in Cg:
                    if need.get(int(k), 0) > 0:
                        healed.append(int(k))
                        need[int(k)] -= 1
                    # stop early if we have all required counts
                    if sum(need.values()) <= 0:
                        break
                # If still short, fill by repeating available clusters from Cr_valid order
                if sum(need.values()) > 0:
                    base = list(Cr_valid.tolist())
                    for k in base:
                        if need.get(int(k), 0) > 0:
                            healed.append(int(k))
                            need[int(k)] -= 1
                        if sum(need.values()) <= 0:
                            break
                Cg = np.asarray(healed[:len(Cr_valid)], dtype=int)

            # queues of real indices per cluster (preserve original order)
            per_cluster = {k: np.where((Cr_all == k) & valid_real)[0]
                           for k in np.unique(Cr_all) if (k != -1 or not drop_noise)}
            take_ptr = {k: 0 for k in per_cluster}
            take_idx = []
            for k in Cg:
                if k == -1 and drop_noise:
                    continue
                if k not in per_cluster or take_ptr[k] >= len(per_cluster[k]):
                    raise AssertionError(
                        f"[align] {mt}: not enough real rows to match cluster {k}. "
                        f"Hint: cache/data mismatch—try re-running with --force_regen."
                    )
                take_idx.append(per_cluster[k][take_ptr[k]])
                take_ptr[k] += 1
            idx = np.asarray(take_idx, dtype=int)

        # --- finalize aligned arrays ---
        Xr = Xr_all[idx]
        Cr = Cr_all[idx]

        # hard equalization (paranoia)
        if len(Xg) != len(Xr):
            n = min(len(Xg), len(Xr))
            Xg, Xr, Cr = Xg[:n], Xr[:n], Cr[:n]

        # If we have generated cluster ids, reorder Xg/Cg to match Cr exactly.
        if clusters_by_type_gen is not None and mt in clusters_by_type_gen:
            Cg_full = np.asarray(clusters_by_type_gen[mt], dtype=int)
            Cg = Cg_full[:len(Xg)]
            from collections import Counter, defaultdict, deque

            # First, require the same multiset (counts) of clusters
            if Counter(Cg.tolist()) != Counter(Cr.tolist()):
                # Auto-heal: trim/expand Cg to match Cr multiset
                need = Counter(Cr.tolist())
                healed = []
                for k in Cg:
                    if need.get(int(k), 0) > 0:
                        healed.append(int(k))
                        need[int(k)] -= 1
                    if sum(need.values()) <= 0:
                        break
                if sum(need.values()) > 0:
                    # fill remaining from Cr order
                    for k in Cr:
                        if need.get(int(k), 0) > 0:
                            healed.append(int(k))
                            need[int(k)] -= 1
                        if sum(need.values()) <= 0:
                            break
                Cg = np.asarray(healed[:len(Cr)], dtype=int)

            # Build queues of generated-row indices per cluster, then pop in Cr order
            buckets = defaultdict(deque)
            for j, k in enumerate(Cg):
                buckets[int(k)].append(j)

            perm = [buckets[int(k)].popleft() for k in Cr]
            Xg = Xg[perm]
            Cg = Cg[perm]

            # Final safety: row-wise identity must now hold
            if not np.array_equal(Cg, Cr):
                raise AssertionError(f"[align] {mt}: internal reorder failed (cluster identities disagree).")

        real_aligned[mt], gen_aligned[mt], clusters_aligned[mt] = Xr, Xg, Cr

        assert real_aligned[mt].shape[0] == gen_aligned[mt].shape[0] == clusters_aligned[mt].shape[0], \
            f"[align] shape mismatch for {mt}"

    return real_aligned, gen_aligned, clusters_aligned


def per_meal_deviation_from_rdi(
    X_grams: np.ndarray,
    meal_type: str,
    W_per100g: np.ndarray,
    nutrient_names: list[str],
    RDI_per_day: np.ndarray,
    meal_fraction_map: dict[str, float],
    include_nutrients: list[str] | None = None,
    mode: str = "log2",            # "abs" | "signed" | "log2"
    clip: float | None = 2.0       # for "signed": ±clip around 0; for "log2": ±clip in log2 space
) -> np.ndarray:
    """
    Returns one scalar per meal row.
      - mode="abs": mean(|coverage-1|)          (unsigned)
      - mode="signed": mean(coverage-1)         (negative=under, positive=over)
      - mode="log2": mean(log2(coverage))       (0=on target; ±1 = 2×/½×; symmetric)
    coverage = meal nutrient / (RDI_per_day * meal fraction), over nutrients with RDI > 0.
    Also imported by analyses/ (outlier, meal-split and portion-ablation checks).
    """
    if X_grams.size == 0:
        return np.zeros((0,), dtype=np.float32)

    N_meal = (X_grams.astype(np.float32, copy=False) / 100.0) @ W_per100g.astype(np.float32, copy=False)

    if include_nutrients is None:
        include = np.ones(len(nutrient_names), dtype=bool)
    else:
        include = np.array([n in include_nutrients for n in nutrient_names], dtype=bool)

    rdi = np.asarray(RDI_per_day, dtype=np.float32)
    include &= np.isfinite(rdi) & (rdi > 0)

    if not np.any(include):
        return np.zeros(N_meal.shape[0], dtype=np.float32)

    meal_frac = float(meal_fraction_map.get(meal_type, 0.33))
    targets = rdi[include] * meal_frac  # (S,)

    cov = N_meal[:, include] / np.maximum(targets[None, :], 1e-8)

    if mode == "log2":
        x = np.log2(np.clip(cov, 1e-8, np.inf))
        if clip is not None:
            x = np.clip(x, -float(clip), float(clip))
        dev = np.mean(x, axis=1).astype(np.float32)
    elif mode == "signed":
        x = cov - 1.0
        if clip is not None:
            x = np.clip(x, -float(clip), float(clip))
        dev = np.mean(x, axis=1).astype(np.float32)
    else:  # "abs"
        x = np.abs(cov - 1.0)
        if clip is not None:
            x = np.clip(x, 0.0, float(clip))
        dev = np.mean(x, axis=1).astype(np.float32)

    return dev


def ensure_values_comparable(W_per100g: np.ndarray, nutrient_names: list[str], RDI_per_day: np.ndarray):
    """Check that the harmonized W (per 100 g), nutrient names and RDI agree before generation.

    Rescales a per-gram table to per 100 g, raises a vitamin D RDI below 600 IU to 800 IU
    (in place), and prints a warning for nutrients whose units look off.
    """
    # 1) Basic shape alignment
    assert W_per100g.ndim == 2, "W must be 2D (D × P)"
    D, P = W_per100g.shape
    assert P == len(nutrient_names) == RDI_per_day.shape[0], \
        f"Mismatch: W columns={P}, names={len(nutrient_names)}, RDI={RDI_per_day.shape[0]}"

    # 2) Heuristic: catch per-gram tables posing as per-100g
    try:
        p_idx = nutrient_names.index("protein_g")
        # per-100g protein rarely has max ≤ 1.5 g; if so, it's likely per-gram
        if np.nanmax(W_per100g[:, p_idx]) <= 1.5:
            print("[FIX] Detected per-gram nutrient table; converting to per-100g by ×100.")
            W_per100g = W_per100g * 100.0
    except ValueError:
        pass

    # 3) Fix obviously wrong Vitamin D values before checking ratios
    vitamin_d_idx = None
    for i, name in enumerate(nutrient_names):
        if name == "vitamin_d_iu":
            vitamin_d_idx = i
            break
    
    if vitamin_d_idx is not None:
        rdi_iu = RDI_per_day[vitamin_d_idx]
        # If RDI is suspiciously low (like 274 IU instead of 800 IU), correct to standard value
        # Standard daily RDI is 20 mcg = 800 IU
        if rdi_iu < 600:  # Less than 75% of expected 800 IU
            RDI_per_day[vitamin_d_idx] = 800.0  # Set to correct value
            print(f"[DEBUG] Vitamin D correction: {rdi_iu} IU → 800.0 IU (corrected to standard RDI)")
    
    # 4) Unit sanity: warn if orders of magnitude look off against RDI
    #    (e.g., mg vs µg confusion). Warning only; src/core/nutrition_harmonizer.py unifies units.
    median_food = np.nanmedian(W_per100g, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        approx_ratio = median_food / np.maximum(RDI_per_day, 1e-8)  # per 100 g as a fraction of daily RDI
    suspicious = np.where(np.isfinite(approx_ratio) & ((approx_ratio > 10) | (approx_ratio < 1e-4)))[0]
    if suspicious.size:
        names = [nutrient_names[i] for i in suspicious]
        print(f"[WARN] Unusual per-100g→RDI scale for: {names[:8]}{'...' if len(names)>8 else ''} "
              f"(check units in mapping files).")

    return W_per100g, nutrient_names, RDI_per_day


# ---------- Cost/Health impact metrics and exports ----------
def _compute_cost_per_meal(X_grams: np.ndarray, price_per_100g: np.ndarray) -> np.ndarray:
    if X_grams.size == 0:
        return np.zeros((0,), dtype=np.float32)
    price_per_g = np.asarray(price_per_100g, dtype=np.float32) / 100.0
    return (X_grams.astype(np.float32) * price_per_g[None, :]).sum(axis=1).astype(np.float32)


def _find_energy_index(nutrient_names: list[str]) -> int:
    for i, n in enumerate(nutrient_names):
        if n.lower() in ("energy_kcal", "kcal", "energy"):
            return i
    return -1


def _winsorize(x: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> np.ndarray:
    if x.size == 0:
        return x
    lo_v = np.nanquantile(x, lo)
    hi_v = np.nanquantile(x, hi)
    return np.clip(x, lo_v, hi_v)


def _pareto_frontier(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return boolean mask for Pareto-efficient points (min x, max y)."""
    if x.size == 0:
        return np.zeros((0,), dtype=bool)
    order = np.argsort(x)
    best_y = -np.inf
    mask = np.zeros_like(x, dtype=bool)
    for idx in order:
        if y[idx] > best_y:
            mask[idx] = True
            best_y = y[idx]
    return mask


def export_cost_health_metrics(
    real_by_type: Dict[str, np.ndarray],
    gen_by_type: Dict[str, np.ndarray],
    clusters_by_type: Dict[str, np.ndarray],
    feature_names: list[str],
    nutrient_names: list[str],
    W_per100g: np.ndarray,
    RDI: np.ndarray,
    meal_fraction_map: Dict[str, float],
    price_map_csv: Optional[Path],
    out_dir: Path,
    *,
    exclude_noncaloric_beverages: bool = False,
    noncal_kcal_per100g_threshold: float = 2.0,
    noncal_price_per100g_threshold: float = 0.20,
    restaurant_overhead_usd: float = 1.0,
) -> Optional[Path]:
    """Compute and export per-pair cost/health metrics CSV (<out_dir>/tables/cost_health_pair_metrics.csv).

    One row per aligned (real, generated) meal pair. Costs use restaurant-style prices
    (price map x food-group multiplier + restaurant_overhead_usd per meal).
    Columns: meal_type, cluster_id, cost_real, cost_gen, delta_cost_meal,
             delta_cost_per_1k_kcal, rdi_dev_real, rdi_dev_gen, rdi_reduction_pct,
             cost_effectiveness, delta_cost_per_1k_kcal_plot, rdi_reduction_pct_plot,
             frontier_flag
    """
    try:
        if price_map_csv is None or not price_map_csv.exists():
            print("[COST] No price map available; skipping cost/health export.")
            return None
        price_df = pd.read_csv(price_map_csv)
        price_df["food_code"] = price_df["food_code"].astype(str).str.zfill(8)
        pmap = {str(r["food_code"]): float(r["price_per_100g_usd"]) if pd.notna(r["price_per_100g_usd"]) else np.nan
                for _, r in price_df.iterrows()}
        price_vec = np.array([pmap.get(str(c).zfill(8), np.nan) for c in feature_names], dtype=np.float32)
        # Impute with median
        if np.isnan(price_vec).any():
            med = float(np.nanmedian(price_vec)) if not np.isnan(np.nanmedian(price_vec)) else 0.0
            price_vec = np.where(np.isnan(price_vec), med, price_vec).astype(np.float32)

        # Optionally exclude noncaloric beverages from cost (e.g., plain water, diet soda, unsweetened coffee/tea)
        if exclude_noncaloric_beverages:
            energy_idx = _find_energy_index(nutrient_names)
            if energy_idx >= 0:
                energy_per100g = W_per100g[:, energy_idx].astype(np.float32)
                noncal_mask = (energy_per100g <= float(noncal_kcal_per100g_threshold)) & (price_vec <= float(noncal_price_per100g_threshold))
                if np.any(noncal_mask):
                    price_vec = price_vec.copy()
                    price_vec[noncal_mask] = 0.0

        # Restaurant multipliers per food category (and subcategory)
        mult_map = {
            "beverages": {"plain_water": 1.1, "coffee_tea": 4.0, "diet_beverages": 4.0, "juice": 3.5, "default": 3.0},
            "mixed_dishes": {"sandwiches": 2.4, "pizza": 1.7, "soups": 1.3, "mexican": 2.6, "default": 2.4},
            "protein": 2.8,
            "grains": 2.0,
            "dairy": 2.2,
            "fruits": 2.0,
            "vegetables": 2.0,
            "snacks_sweets": 2.5,
            "condiments_sauces": 1.2,
            "sugars": 1.2,
            "fats_oils": 1.2,
            "alcoholic_beverages": 3.5,
            "_default": 2.8
        }

        # Optional helper dictionaries serialized in exports
        code_to_name = {}
        food_group_map = {}
        try:
            exports_dir = out_dir / "data_exports"
            if (exports_dir / "food_code_to_name.json").exists():
                import json as _json
                code_to_name = _json.load(open(exports_dir / "food_code_to_name.json"))
            if (exports_dir / "food_to_group.json").exists():
                import json as _json
                food_group_map = _json.load(open(exports_dir / "food_to_group.json"))
        except Exception:
            pass

        def _infer_group_and_sub(code8: str, name: str) -> Tuple[str, str]:
            g = (food_group_map.get(code8) or "").strip().lower()
            s = (name or "").lower()
            if not g:
                if any(w in s for w in ["coffee", "tea", "water", "soda", "juice"]):
                    g = "beverages"
                elif any(w in s for w in ["pizza", "sandwich", "soup", "burrito", "taco", "enchilada", "bowl", "plate"]):
                    g = "mixed_dishes"
                elif any(w in s for w in ["chicken", "beef", "pork", "turkey", "fish", "shrimp", "salmon", "egg"]):
                    g = "protein"
                elif any(w in s for w in ["bread", "rice", "pasta", "cereal", "cracker", "tortilla", "noodle"]):
                    g = "grains"
                elif any(w in s for w in ["yogurt", "cheese", "milk"]):
                    g = "dairy"
                elif any(w in s for w in ["apple", "banana", "grape", "fruit"]):
                    g = "fruits"
                elif any(w in s for w in ["salad", "broccoli", "spinach", "vegetable", "veg"]):
                    g = "vegetables"
                elif any(w in s for w in ["bar", "cookie", "chips", "snack", "dessert", "cake", "candy"]):
                    g = "snacks_sweets"
                else:
                    g = "_default"
            sub = ""
            if g == "beverages":
                if "water" in s:
                    sub = "plain_water"
                elif "diet" in s or "zero" in s:
                    sub = "diet_beverages"
                elif "coffee" in s or "tea" in s:
                    sub = "coffee_tea"
                elif "juice" in s:
                    sub = "juice"
            elif g == "mixed_dishes":
                if "sandwich" in s:
                    sub = "sandwiches"
                elif "pizza" in s:
                    sub = "pizza"
                elif "soup" in s:
                    sub = "soups"
                elif any(w in s for w in ["taco", "burrito", "enchilada", "mexican"]):
                    sub = "mexican"
            return g, sub

        # Effective price vector: per-100 g price times the category multiplier
        eff = price_vec.astype(np.float32)
        for j, code in enumerate(feature_names):
            code8 = str(code).zfill(8)
            name = code_to_name.get(code8, "")
            g, sub = _infer_group_and_sub(code8, name)
            m = mult_map.get(g)
            mult_val: float
            if isinstance(m, dict):
                mult_val = float(m.get(sub, m.get("default", mult_map.get("_default", 2.8))))
            else:
                mult_val = float(m if isinstance(m, (int, float)) else mult_map.get("_default", 2.8))
            eff[j] = eff[j] * float(np.clip(mult_val, 0.8, 6.0))

        rows = []
        energy_idx = _find_energy_index(nutrient_names)
        for mt in [m for m in ["Breakfast","Lunch","Dinner"] if m in real_by_type and m in gen_by_type]:
            Xr = real_by_type[mt]
            Xg = gen_by_type[mt]
            C  = clusters_by_type.get(mt)
            if C is None:
                continue
            n = min(len(Xr), len(Xg), len(C))
            Xr = Xr[:n]; Xg = Xg[:n]; C = C[:n]

            cost_r = _compute_cost_per_meal(Xr, eff)
            cost_g = _compute_cost_per_meal(Xg, eff)
            if restaurant_overhead_usd and restaurant_overhead_usd != 0.0:
                cost_r = cost_r + float(restaurant_overhead_usd)
                cost_g = cost_g + float(restaurant_overhead_usd)
            dcost_meal = (cost_g - cost_r)

            # Energy per meal for normalization
            if energy_idx >= 0:
                N_r = fast_meal_nutrients_per_meal(Xr, W_per100g)
                N_g = fast_meal_nutrients_per_meal(Xg, W_per100g)
                kcal_r = np.maximum(N_r[:, energy_idx], 1e-6)
                kcal_g = np.maximum(N_g[:, energy_idx], 1e-6)
                # Normalize delta cost by mean kcal (per 1000 kcal)
                kcal_mean = 0.5 * (kcal_r + kcal_g)
                dcost_per_1k = dcost_meal / (kcal_mean / 1000.0)
            else:
                dcost_per_1k = dcost_meal.copy()

            # Health deviation (use log2 metric centered at 0; convert to absolute distance magnitude)
            dev_r = np.abs(per_meal_deviation_from_rdi(Xr, mt, W_per100g, nutrient_names, RDI, meal_fraction_map, mode="log2"))
            dev_g = np.abs(per_meal_deviation_from_rdi(Xg, mt, W_per100g, nutrient_names, RDI, meal_fraction_map, mode="log2"))
            # Percent reduction (positive is improvement)
            rdi_reduction_pct = ((dev_r - dev_g) / np.maximum(dev_r, 1e-6)) * 100.0

            # Cost-effectiveness (improvement per $1 per 1k kcal)
            denom = np.where(np.abs(dcost_per_1k) < 1e-6, np.nan, dcost_per_1k)
            ce = rdi_reduction_pct / denom

            rows.append(pd.DataFrame({
                "meal_type": mt,
                "cluster_id": C,
                "cost_real": cost_r,
                "cost_gen": cost_g,
                "delta_cost_meal": dcost_meal,
                "delta_cost_per_1k_kcal": dcost_per_1k,
                "rdi_dev_real": dev_r,
                "rdi_dev_gen": dev_g,
                "rdi_reduction_pct": rdi_reduction_pct,
                "cost_effectiveness": ce,
            }))

        if not rows:
            return None
        df = pd.concat(rows, ignore_index=True)
        # Winsorize deltas for stable plotting exports
        df["delta_cost_per_1k_kcal_plot"] = _winsorize(df["delta_cost_per_1k_kcal"].to_numpy())
        df["rdi_reduction_pct_plot"] = _winsorize(df["rdi_reduction_pct"].to_numpy())
        # Frontier flag on plot-normalized axes
        mask = _pareto_frontier(df["delta_cost_per_1k_kcal_plot"].to_numpy(), df["rdi_reduction_pct_plot"].to_numpy())
        df["frontier_flag"] = mask.astype(bool)

        tables_dir = out_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        out_csv = tables_dir / "cost_health_pair_metrics.csv"
        df.to_csv(out_csv, index=False)
        print(f"[COST] Exported cost/health metrics → {out_csv}")
        return out_csv
    except Exception as e:
        print(f"[COST] Failed to compute/export cost/health metrics: {e}")
        return None


def compute_cluster_itemcount_stats(real_by_type: dict[str, np.ndarray],
                                    clusters_by_type: dict[str, np.ndarray]) -> dict[str, dict[int, dict]]:
    """
    Returns:
      { meal_type: { cluster_id: {"p25":int, "median":int, "p75":int, "n":int} } }
    """
    stats = {}
    for mt, Xr in real_by_type.items():
        C = np.asarray(clusters_by_type.get(mt, []), dtype=int)
        if Xr is None or C.size == 0:
            continue
        m = (C != -1)
        stats_mt = {}
        for k in np.unique(C[m]):
            mk = (C == k) & m
            if not np.any(mk):
                continue
            counts = (Xr[mk] > 0).sum(axis=1).astype(int)
            if counts.size:
                p25, med, p75 = np.percentile(counts, [25, 50, 75])
                stats_mt[int(k)] = {
                    "p25": int(round(p25)),
                    "median": int(round(med)),
                    "p75": int(round(p75)),
                    "n": int(counts.size),
                }
        stats[mt] = stats_mt
    return stats



# ------------------------
# Main orchestration
# ------------------------
def load_and_prepare_data(args, project_root):
    """Load and prepare all necessary data files"""
    nutrient_df = load_food_nutrient_table_any(project_root)
    code2name = build_foodname_map(nutrient_df)
    serving_df = load_serving_sizes_any(project_root)
    return nutrient_df, code2name, serving_df


def attach_spider_nutrition(data, project_root, feature_names):
    """Add the nutrient table and daily targets used by the Figure 3B spider plots to `data`.

    They come straight from the harmonised FNDDS table and the Healthy US-Style pattern,
    without the vitamin D adjustment that ensure_values_comparable() applies to the RDI used
    elsewhere. Both the fresh and the cached
    path call this, so `make eval` draws the same Figure 3B either way.
    """
    try:
        from src.core.nutrition_harmonizer import load_harmonized_nutrition
        from src.core.generator import Generator

        harmonized_data = load_harmonized_nutrition(
            str(project_root / "data" / "mappings" / "food_to_nutrient.csv"),
            str(project_root / "data" / "raw" / "food_patterns" / "healthy_US_style_food_pattern.csv"),
            use_standard_dri_overrides=False
        )
        nutrient_names_harmonized = harmonized_data["nutrient_names"]
        feature_ids_harmonized = harmonized_data["feature_ids"]
        # Align harmonized rows to the training features (as in the fresh data loading)
        W_per_g = Generator._align_nutrient_matrix_to_features(
            W_per100g=harmonized_data["W_per100g"],
            feature_ids=feature_ids_harmonized,
            feature_names=feature_names,
            nutrient_names=nutrient_names_harmonized,
        )
        data["W_harmonized"] = W_per_g * 100.0
        data["nutrient_names_harmonized"] = nutrient_names_harmonized
        data["RDI_harmonized"] = harmonized_data["RDI_per_day"]
        data["feature_ids_harmonized"] = feature_ids_harmonized
        print("[INFO] Loaded harmonized nutrition data for spider plots")
    except Exception as e:
        print(f"[WARN] Could not load harmonized nutrition data for spider plots: {e}")
        print("[WARN] Spider plots will use the basic nutrition data")


def load_cached_or_fresh_data(args, exports_dir, project_root, code2name, serving_df):
    """Load cached data if available, otherwise return None to trigger fresh loading"""
    if not args.force_regen:
        print(f"[CACHE] Checking for cached data in {exports_dir}")
        cached = maybe_load_exports(exports_dir)
        if cached is not None:
            print("✓ Using cached data exports.")
            data = cached
            code2name = data["code2name"] or build_foodname_map(load_food_nutrient_table_any(project_root))
            # these exist in cached exports
            feature_names = data.get("feature_names", [])
            if not feature_names:
                print("[CACHE] Missing feature_names in cached exports — auto-refreshing cache with fresh load.")
                return None  # trigger fresh load instead of forcing --force_regen
            # Infer presence-only from the data (don't force True)
            any_prob = any(_looks_like_probabilities(Xg) for Xg in data["generated_meals"].values())
            # For cached data, we don't need to rebuild serving_vec since it's not used in the Generator
            # (Generator uses its own realistic serving sizes)
            serving_vec = None

            if any_prob:
                # If cached gen looks like probabilities and user wants grams, convert now.
                want_policy = args.presence_to_grams  # argparse allows only "rdi_per_kcal"
                if want_policy != "none":
                    print("[CACHE] Detected presence-only cached generations in probability space; converting to grams...")
                    # Optional: prepare price vector aligned to feature_names
                    price_vector = None
                    if getattr(args, "cost_weight", 0.0) and args.cost_weight > 0.0:
                        try:
                            price_df = pd.read_csv(project_root / args.price_map)
                            # Expect columns: food_code, price_per_100g_usd
                            price_df["food_code"] = price_df["food_code"].astype(str).str.zfill(8)
                            price_map = {str(r["food_code"]): float(r["price_per_100g_usd"]) if pd.notna(r["price_per_100g_usd"]) else np.nan for _, r in price_df.iterrows()}
                            pv = np.array([price_map.get(str(fc).zfill(8), np.nan) for fc in feature_names], dtype=np.float32)
                            # Impute missing with median of available
                            if np.isnan(pv).any():
                                med = float(np.nanmedian(pv)) if not np.isnan(np.nanmedian(pv)) else 0.0
                                pv = np.where(np.isnan(pv), med, pv).astype(np.float32)
                            price_vector = pv
                            covered = np.mean(~np.isnan([price_map.get(str(fc).zfill(8), np.nan) for fc in feature_names])) * 100.0
                            print(f"[COST] Price coverage over features: {covered:.1f}% | median=${float(np.median(pv)):.2f}/100g")
                        except Exception as e:
                            print(f"[COST] Failed to load/align price_map '{args.price_map}': {e}. Proceeding without cost.")

                    data["generated_meals"] = _coerce_presence_probs_to_grams(
                            data["generated_meals"],
                            policy=want_policy,
                            feature_names=feature_names,
                            code2name=code2name,
                            W=data["W"],
                            nutrient_names=data["nutrient_names"],
                            RDI=data["RDI"],
                            serving_vec=serving_vec,
                            food_to_group=data.get("food_to_group", {}),
                            price_vector=price_vector,
                            cost_weight=getattr(args, "cost_weight", 0.0),
                            seed=args.seed,
                            top_k=args.top_k,
                            min_prob=args.min_prob,
                            deterministic=args.deterministic,
                        )
                else:
                    print("[INFO] Cached generated meals look like probabilities and --presence_to_grams=none. "
                          "Keeping them in probability space; nutrient-based evaluation/plots will be skipped by design.")
            
            attach_spider_nutrition(data, project_root, feature_names)
            return data
    else:
        print("[CACHE] --force_regen=True, skipping cache check")
    return None


def load_model_and_config(model_dir, args):
    """Load the trained CVAE model and configuration"""
    model, cfg, raw_cfg = load_trained_cvae(model_dir, args.generation_temp, args.generation_nucleus_p)
    print(f"[MODEL] type={type(model).__name__}")
    parts = {
        "decoder": getattr(model, "decoder", None),
        "cvae": getattr(model, "cvae", None),
        "cvae.decoder": getattr(getattr(model, "cvae", None), "decoder", None),
    }
    for name, obj in parts.items():
        if obj is not None:
            print(f"[MODEL] has {name}: {type(obj).__name__}")
    
    # Load feature names
    train_feats_path = None
    possible_paths = [
        model_dir / "feature_names.json",
        model_dir / "presence_only_model" / "feature_names.json",
    ]
    
    for path in possible_paths:
        if path.exists():
            train_feats_path = path
            break
            
    if train_feats_path is None:
        raise FileNotFoundError(
            f"feature_names.json not found in any of: {[str(p) for p in possible_paths]}. "
            f"Retrain with feature_names saved."
        )
        
    feature_names_trained = json.loads(train_feats_path.read_text())
    # normalize to 8-digit strings
    feature_names_trained = [str(c).zfill(8) for c in feature_names_trained]
    
    return model, cfg, raw_cfg, feature_names_trained


def run_evaluation_and_visualization(data, code2name, args, out_dir, project_root, exports_dir, figures_dir, cluster_filter=None):
    """Run the complete evaluation and visualization pipeline"""
    # Validate inputs
    data_for_validate = {
        "real_meals": data["real_meals"],
        "generated_meals": data["generated_meals"],
        "W": data["W"],
        "nutrient_names": data["nutrient_names"],
        "RDI": data["RDI"],
    }
    
    # Debug meal types
    print(f"[DEBUG] Real meal types: {list(data['real_meals'].keys())}")
    print(f"[DEBUG] Generated meal types: {list(data['generated_meals'].keys())}")
    
    if not validate_data_contracts(data_for_validate):
        print("Data validation failed; cannot proceed.")
        return 1
    
    # If any generated matrix still looks like probabilities, skip nutrient-based steps
    any_prob = any(_looks_like_probabilities(Xg) for Xg in data["generated_meals"].values())
    if any_prob:
        print("[WARN] Generated data appear to be presence probabilities; "
              "nutrient-based evaluation is skipped. "
              "To evaluate nutrients, re-run with cached conversion via "
              "--presence_to_grams=assign (default) or --presence_to_grams=servings.")
        return 0

    # Use harmonized data for spider plots if available, otherwise fall back to basic data
    if "W_harmonized" in data and "nutrient_names_harmonized" in data and "RDI_harmonized" in data:
        print("[INFO] Using harmonized nutrition data for spider plots (includes vitamins/minerals)")
        W_for_spiders = data["W_harmonized"]
        nutrient_names_for_spiders = data["nutrient_names_harmonized"]
        RDI_for_spiders = data["RDI_harmonized"]
    else:
        print("[INFO] Using basic nutrition data for spider plots (limited nutrients)")
        W_for_spiders = data["W"]
        nutrient_names_for_spiders = data["nutrient_names"]
        RDI_for_spiders = data["RDI"]
    
    processed = {
        "real_meals": data["real_meals"],
        "generated_meals": data["generated_meals"],
        "clusters_real": data.get("clusters_real", data.get("clusters")),   # fresh runs use "clusters"
        "clusters_gen": data.get("clusters_gen"),
        "W": data["W"],  # Keep original for other evaluations
        "nutrient_names": data["nutrient_names"],
        "RDI": data["RDI"],
        "W_harmonized": W_for_spiders,  # For spider plots
        "nutrient_names_harmonized": nutrient_names_for_spiders,  # For spider plots
        "RDI_harmonized": RDI_for_spiders,  # For spider plots
        "feature_names": data["feature_names"],
        "food_to_group": data["food_to_group"],
        "gen_row_indices_by_meal": data["gen_row_indices_by_meal"],
        "meal_fraction_map": {"Breakfast": 0.25, "Lunch": 0.35, "Dinner": 0.40},
        "limits": {"SFA_pct_kcal": 10.0, "sodium_mg": 2300.0},
        "kcal_spec": KCAL_SPEC_CANON,
    }

    # Initialize managers
    evaluation_manager = EvaluationManager(processed, args)
    visualization_manager = VisualizationManager(processed, code2name)



    # Run nutrient profile sanity check
    evaluation_manager.run_nutrient_profile_sanity_check(out_dir, project_root)

    # Align data for evaluation
    real_aligned, gen_aligned, clusters_aligned = align_for_evaluation(
        real_by_type=data["real_meals"],
        gen_by_type=data["generated_meals"],
        clusters_by_type_real=data.get("clusters_real", data.get("clusters")),
        clusters_by_type_gen=data.get("clusters_gen"),
        gen_row_indices_by_meal=data.get("gen_row_indices_by_meal"),
        drop_noise=True,
        pair_table=data.get("pair_table"),
        meal_type_mapping=data.get("meal_type_mapping"),
        cluster_filter=cluster_filter,
    )

    # Run bootstrap evaluation
    bootstrap_results = evaluation_manager.run_bootstrap_evaluation(
        real_aligned, 
        gen_aligned, 
        clusters_aligned, 
        None
    )

    # Print the bootstrap summary (log only)
    evaluation_manager.create_bootstrap_summary(bootstrap_results)

    # Emit per-cluster counts (real, generated, assessed/aligned)
    try:
        tables_dir = out_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        mt_seq = [mt for mt in MEAL_TYPES if mt in data.get("real_meals", {}) and mt in real_aligned]
        clusters_gen_raw = data.get("clusters_gen")
        # Fresh generation stores real-meal clusters under "clusters", cached exports under "clusters_real".
        clusters_real_raw = data.get("clusters_real", data.get("clusters"))
        for mt in mt_seq:
            # Raw counts from exports
            cr_raw = np.asarray(clusters_real_raw[mt], dtype=int)
            real_raw_counts = Counter([int(c) for c in cr_raw if c != -1])
            gen_raw_counts = {}
            if isinstance(clusters_gen_raw, dict) and mt in clusters_gen_raw and clusters_gen_raw[mt] is not None:
                cg_raw = np.asarray(clusters_gen_raw[mt], dtype=int)
                gen_raw_counts = Counter([int(c) for c in cg_raw if c != -1])

            # Assessed counts (aligned, equal-length real/gen used in evaluation)
            ca = np.asarray(clusters_aligned[mt], dtype=int)
            assessed_counts = Counter([int(c) for c in ca if c != -1])

            cluster_ids = sorted(set(real_raw_counts.keys()) | set(gen_raw_counts.keys()) | set(assessed_counts.keys()))
            for cid in cluster_ids:
                cname = _get_cluster_name(mt.lower(), int(cid))
                rows.append({
                    "meal_type": mt,
                    "cluster_id": int(cid),
                    "cluster_name": cname,
                    "real_count_raw": int(real_raw_counts.get(cid, 0)),
                    "gen_count_raw": int(gen_raw_counts.get(cid, assessed_counts.get(cid, 0))),
                    "assessed_count": int(assessed_counts.get(cid, 0)),
                })

        if rows:
            df_counts = pd.DataFrame(rows).sort_values(["meal_type", "cluster_id"]).reset_index(drop=True)
            df_counts.to_csv(tables_dir / "cluster_counts.csv", index=False)

            # Totals summary
            totals = df_counts.groupby("meal_type")[
                ["real_count_raw", "gen_count_raw", "assessed_count"]
            ].sum().reset_index()
            totals.loc[totals.shape[0]] = ["ALL"] + list(df_counts[["real_count_raw","gen_count_raw","assessed_count"]].sum())
            totals.to_csv(tables_dir / "cluster_counts_totals.csv", index=False)

            print(f"✓ Cluster count table → {tables_dir / 'cluster_counts.csv'}")
            print(f"✓ Totals summary → {tables_dir / 'cluster_counts_totals.csv'}")
    except Exception as e:
        print(f"[WARN] Failed to write cluster count tables: {e}")

    # Plot main figure
    visualization_manager.plot_main_figure(
        real_aligned, 
        gen_aligned, 
        None,  # adj_by_type - no RDI adjustment
        clusters_aligned, 
        figures_dir
    )

    # Export cost/health metrics CSV for downstream cost-aware visuals
    try:
        metrics_csv = export_cost_health_metrics(
            real_by_type=real_aligned,
            gen_by_type=gen_aligned,
            clusters_by_type=clusters_aligned,
            feature_names=processed["feature_names"],
            nutrient_names=processed["nutrient_names"],
            W_per100g=processed["W"],
            RDI=processed["RDI"],
            meal_fraction_map=processed["meal_fraction_map"],
            price_map_csv=(project_root / args.price_map) if getattr(args, "price_map", None) else None,
            out_dir=out_dir,
        )
        if metrics_csv:
            print(f"[COST] Metrics ready: {metrics_csv}")
    except Exception as e:
        print(f"[COST] Failed to export cost/health metrics: {e}")

    # Plot cluster deviation boxplots
    visualization_manager.plot_cluster_deviation_boxplots(
        real_aligned, 
        gen_aligned, 
        clusters_aligned, 
        figures_dir
    )
    
    # Create comprehensive spider plots
    print("Creating comprehensive spider plots...")
    create_spider_plots(
        real_data=real_aligned,
        gen_data=gen_aligned,
        W=processed["W_harmonized"],
        nutrient_names=processed["nutrient_names_harmonized"],
        RDI=processed["RDI_harmonized"],
        meal_fraction_map=processed["meal_fraction_map"],
        food_to_group=processed["food_to_group"],
        feature_names=processed["feature_names"],
        output_dir=figures_dir,
        limits=processed.get("limits")  # Pass moderation limits
    )
    print("✓ Spider plots created successfully")

    # Figure 3A: the training-stage overview, coloured by each archetype's change in RDI deviation
    from src.utils.cluster_overview import draw_figure3a
    fig3a = draw_figure3a(out_dir, figures_dir)
    if fig3a:
        print(f"✓ Figure 3A → {fig3a}")

    print(f"✓ Evaluation complete. Results saved to {out_dir}")
    return 0


def load_fresh_data_and_generate(project_root, model_dir, args, nutrient_df, code2name, serving_df, exports_dir, cluster_filter=None):
    """Load fresh data from cluster files and generate meals"""
    # 1) Discover and load ALL cluster frames
    paths_by_mt = find_cluster_pickles(project_root)
    frames_by_mt: Dict[str, List[pd.DataFrame]] = {}
    real_by_type: Dict[str, np.ndarray] = {}
    clusters_real_by_type: Dict[str, np.ndarray] = {}
    for mt, paths in paths_by_mt.items():
        frs = []
        for p in paths:
            try:
                frs.append(load_cluster_df(p))
            except Exception as e:
                warnings.warn(f"Skipping {p}: {e}")
        if frs:
            frames_by_mt[mt] = frs

    if not frames_by_mt:
        print("No cluster frames found. Aborting.")
        return None

    # 2) Load model first to get trained feature order
    model, cfg, raw_cfg, feature_names_trained = load_model_and_config(model_dir, args)
    if model is None:
        print("[ERROR] Could not load a generative model; fresh generation cannot proceed.")
        return None
    
    # Use trained order as the canonical feature space for everything
    feature_names = feature_names_trained

    # 3) Reindex each frame to the trained feature set and concatenate per meal type
    clusters_real_by_type: Dict[str, np.ndarray] = {}
    for mt, frs in frames_by_mt.items():
        aligned_frames = [reindex_to_global(df, feature_names) for df in frs]
        big = pd.concat(aligned_frames, ignore_index=True)
        Cr_full = big["cluster_id"].to_numpy(dtype=np.int32)
        Xr_full = big.drop(columns=["cluster_id"]).to_numpy(dtype=np.float32)

        # Apply cluster filter (KEEP/OMIT)
        keep_cfg = cluster_filter.get("__keep__", {}) if cluster_filter else {}
        omit_cfg = cluster_filter.get("__omit__", {}) if cluster_filter else {}

        if keep_cfg or omit_cfg:
            # Decide the keep set for this meal
            if mt in keep_cfg and len(keep_cfg[mt]):
                keep_set = keep_cfg[mt]
            else:
                # Start from all observed (excluding noise)
                observed = set(int(c) for c in np.unique(Cr_full) if int(c) >= 0)
                # Remove any omits
                if mt in omit_cfg and len(omit_cfg[mt]):
                    observed -= set(int(x) for x in omit_cfg[mt])
                keep_set = observed
            mask = np.isin(Cr_full, list(keep_set))
            Cr_use = Cr_full[mask]
            Xr_use = Xr_full[mask]
        else:
            # No filter provided; still drop noise
            mask = (Cr_full != -1)
            Cr_use = Cr_full[mask]
            Xr_use = Xr_full[mask]

        clusters_real_by_type[mt] = Cr_use
        real_by_type[mt] = Xr_use

    cluster_item_stats = compute_cluster_itemcount_stats(real_by_type, clusters_real_by_type)

    # 4) Harmonized nutrient table (per 100 g) and RDI from src/core/nutrition_harmonizer.py
    from src.core.nutrition_harmonizer import load_harmonized_nutrition
    
    harmonized_data = load_harmonized_nutrition(
        str(project_root / "data" / "mappings" / "food_to_nutrient.csv"),
        str(project_root / "data" / "raw" / "food_patterns" / "healthy_US_style_food_pattern.csv"),
        use_standard_dri_overrides=False
    )
    
    # Extract harmonized data
    W_harmonized = harmonized_data["W_per100g"]  # (D_harmonized, P)
    nutrient_names = harmonized_data["nutrient_names"]  # (P,)
    RDI_harmonized = harmonized_data["RDI_per_day"]  # (P,)
    feature_ids_harmonized = harmonized_data["feature_ids"]  # (D_harmonized,)
    
    # Validate data integrity and ensure comparability
    W_harmonized, nutrient_names, RDI_harmonized = ensure_values_comparable(
        W_harmonized, nutrient_names, RDI_harmonized
    )
    
    from src.core.generator import Generator
    # Align W_harmonized rows to trained feature_names EXACTLY (robust to id type/format)
    # Returns per-gram; convert back to per-100g for the rest of this script.
    W_per_g = Generator._align_nutrient_matrix_to_features(
        W_per100g=W_harmonized,
        feature_ids=feature_ids_harmonized,
        feature_names=feature_names,
        nutrient_names=nutrient_names,
    )
    W = W_per_g * 100.0
    mapped_count = W.shape[0] - int((W == 0).all(axis=1).sum())
    print(f"[DEBUG] Feature mapping: {mapped_count}/{len(feature_names)} features mapped to harmonized data")
    non_zero_count = int((W != 0).sum())
    print(f"[DEBUG] W matrix: {non_zero_count}/{W.size} non-zero values")
    if non_zero_count == 0:
        raise ValueError("[ERROR] W matrix is all zeros after alignment — check feature ids/name formats.")
    
    serving_vec = build_serving_vector(serving_df, feature_names, default_grams=100.0)  # (D,)
    print(f"[DEBUG] External serving_vec stats: min={serving_vec.min():.1f}, max={serving_vec.max():.1f}, median={np.median(serving_vec):.1f}")
    print("[DEBUG] Note: Generator will use its own realistic serving sizes, not this external vector")
    print(f"[DEBUG] Harmonized nutrients: {len(nutrient_names)} nutrients loaded")
    print(f"[DEBUG] Available nutrients: {nutrient_names[:10]}{'...' if len(nutrient_names) > 10 else ''}")

    # --- Sanity on REAL meals energy using canonical energy name ---
    if "energy_kcal" in nutrient_names:
        eidx = nutrient_names.index("energy_kcal")
        for mt, Xr in real_by_type.items():
            N = fast_meal_nutrients_per_meal(Xr, W, prefer_sparse=True)
            kcals = N[:, eidx]
            print(f"[SANITY] {mt:9s} kcal  min={kcals.min():.1f}  "
                  f"median={np.median(kcals):.1f}  max={kcals.max():.1f}  n={len(kcals)}")
    else:
        print("[SANITY] 'energy_kcal' not found; kcal checks skipped.")

    # 5) Use harmonized RDI vector (already aligned to nutrient_names)
    RDI = RDI_harmonized

    # 6) Food groups from the WWEIA main categories of the food-ingredient mapping
    food_to_group = build_foodgroup_map(
        pd.read_pickle(project_root / "data" / "mappings" / "food_ingredient_map.pkl"), feature_names
    )

    # 7) Generate meals (model already loaded above)

    # Build data structures for Generator
    print("[GENERATOR] Building pair_table and meal_type_mapping...")
    
    # Try to load the training-time pair table / meal mapping
    pair_table = {}
    meal_type_mapping = {}

    # Look for explicit JSONs saved at train time
    for cand in [
        model_dir / "pair_table.json",
    ]:
        if cand.exists():
            try:
                j = json.loads(cand.read_text())
                # support both {pair_code: [meal_idx, cluster_id]} and list-of-lists
                if isinstance(j, dict):
                    pair_table = {int(k): (int(v[0]), int(v[1])) for k, v in j.items()}
                elif isinstance(j, list):
                    pair_table = {i: (int(v[0]), int(v[1])) for i, v in enumerate(j)}
                print(f"[GENERATOR] Loaded training pair_table with {len(pair_table)} pairs from {cand}")
                break
            except Exception as e:
                print(f"[WARN] Could not parse {cand}: {e}")

    # meal_type_mapping too (if present)
    for cand in [
        model_dir / "meal_type_mapping.json",
    ]:
        if cand.exists():
            try:
                j = json.loads(cand.read_text())
                meal_type_mapping = {str(k): int(v) for k, v in j.items()}
                print(f"[GENERATOR] Loaded meal_type_mapping from {cand}")
                break
            except Exception as e:
                print(f"[WARN] Could not parse {cand}: {e}")

    # Fallback to config if not found in JSON files
    if not pair_table and isinstance(raw_cfg, dict):
        # presence-only configs often embed this
        cfg_pairs = raw_cfg.get("pair_table")
        if cfg_pairs:
            # tolerate dict or list formats
            if isinstance(cfg_pairs, dict):
                pair_table = {int(k): (int(v[0]), int(v[1])) for k, v in cfg_pairs.items()}
            else:  # list of [meal_idx, cluster_id]
                pair_table = {i: (int(v[0]), int(v[1])) for i, v in enumerate(cfg_pairs)}
        mt_map = raw_cfg.get("meal_type_mapping")
        if mt_map:
            meal_type_mapping = {str(k): int(v) for k, v in mt_map.items()}
    
    # fallback if missing in config:
    if not pair_table:
        print("[GENERATOR] pair_table missing in config; inferring from filtered real frames (fallback).")
        # Build pair_table from filtered real data, then densify like the runner
        pair_table_raw = {}
        pair_code = 0
        for mt in MEAL_TYPES:
            if mt not in clusters_real_by_type: 
                continue
            m_idx = MEAL_TO_IDX[mt]
            uniq = np.unique(clusters_real_by_type[mt])
            uniq = [int(c) for c in uniq if int(c) >= 0]
            for cid in uniq:
                pair_table_raw[pair_code] = (m_idx, cid)
                pair_code += 1
        
        # Densify pair codes to 0..K-1 like the runner does
        all_clusters = set()
        for mt in MEAL_TYPES:
            if mt in clusters_real_by_type:
                all_clusters.update(clusters_real_by_type[mt])
        all_clusters = sorted([c for c in all_clusters if c >= 0])
        
        # Create densified mapping: original_cluster_id -> new_pair_code
        cluster_to_pair = {cid: i for i, cid in enumerate(all_clusters)}
        pair_table = {}
        for orig_pair_code, (m_idx, orig_cid) in pair_table_raw.items():
            if orig_cid in cluster_to_pair:
                new_pair_code = cluster_to_pair[orig_cid]
                pair_table[new_pair_code] = (m_idx, orig_cid)
        
        print(f"[GENERATOR] Generated densified pair_table with {len(pair_table)} pairs, max cluster_id: {max(v[1] for v in pair_table.values())}")
        print(f"[GENERATOR] Cluster mapping: {cluster_to_pair}")
    
    if not meal_type_mapping:
        meal_type_mapping = {mt: MEAL_TO_IDX[mt] for mt in MEAL_TYPES if mt in real_by_type}
    
    print(f"[GENERATOR] Using {len(pair_table)} pairs across {len(meal_type_mapping)} meal types")
    
    # 1) Load gate and check exact shape agreement
    try:
        gate = model.decoder.get_layer("pair_allowed_gate").get_weights()[0]  # shape (K, D) or (K+1, D)
        K, D = gate.shape
        print(f"[GENERATOR] Gate shape: K={K}, D={D}")
    except Exception as e:
        print(f"[ERROR] Could not load pair_allowed_gate: {e}")
        return None

    # MUST load the *training-time* metadata that the runner used
    train_pair_table = None
    train_features = None
    
    # Try to load training-time pair table
    for cand in [
        model_dir / "pair_table.json",
    ]:
        if cand.exists():
            try:
                with open(cand, "r") as f:
                    train_pair_table = {int(k): tuple(v) for k, v in json.load(f).items()}
                print(f"[GENERATOR] Loaded training pair_table from {cand}")
                break
            except Exception as e:
                print(f"[WARN] Could not parse {cand}: {e}")

    # Try to load training-time feature names
    for cand in [
        model_dir / "feature_names.json",
    ]:
        if cand.exists():
            try:
                with open(cand, "r") as f:
                    train_features = json.load(f)
                print(f"[GENERATOR] Loaded training feature_names from {cand}")
                break
            except Exception as e:
                print(f"[WARN] Could not parse {cand}: {e}")

    # Use training metadata if available, otherwise fall back to current
    if train_pair_table is not None:
        pair_table = train_pair_table
        print(f"[GENERATOR] Using training pair_table with {len(pair_table)} pairs")
    
    if train_features is not None:
        feature_names = train_features
        print(f"[GENERATOR] Using training feature_names with {len(feature_names)} features")

    # Hard guards for alignment
    if len(feature_names) != D:
        raise ValueError(f"feature order mismatch: {len(feature_names)} vs gate cols {D}")
    
    # pair_table keys must be a subset of 0..K-1 (keep original training indices)
    bad_keys = [k for k in pair_table.keys() if (int(k) < 0 or int(k) >= K)]
    if bad_keys:
        raise ValueError(f"pair_table contains out-of-range keys {bad_keys[:5]} for gate rows {K}")
    
    # (do NOT assert len(pair_table) == K - allows subset of gate capacity)
    
    print(f"[GENERATOR] ✓ Alignment verified: {len(pair_table)} pairs, {len(feature_names)} features")
    

    # Create Generator instance
    gen_kwargs = {
        "model": model,
        "feature_names": feature_names,
        "pair_table": pair_table,
        "meal_type_mapping": meal_type_mapping,
        "cluster_item_stats": cluster_item_stats,  # Pass cluster stats for item count
        "rng_seed": args.seed,
        "rdi_daily": RDI_harmonized,  # Use harmonized RDI from local var
        # Beverage capping parameters
        "food_to_group": food_to_group,
        "max_beverage_grams": 600,  # Fallback limit
        "max_beverage_grams_by_meal": {
            0: 500,  # Breakfast: 450-500g
            1: 400,  # Lunch: 350-450g  
            2: 400,  # Dinner: 350-450g
        },
        # Subcategory expansion parameters
        "expand_by_subcategory": False  # Keep off by default for runner consistency
    }
    
    # Add nutrient parameters for rdi_per_kcal portioning
    if args.portion_strategy == "rdi_per_kcal":
        # Convert RDI per day to per-kcal targets
        rdi_per_kcal = {}
        if "energy_kcal" in nutrient_names:
            energy_idx = nutrient_names.index("energy_kcal")
            daily_energy = RDI[energy_idx]  # Daily energy target
            for i, nutrient in enumerate(nutrient_names):
                if i != energy_idx and RDI[i] > 0:  # Skip energy itself and zero nutrients
                    rdi_per_kcal[nutrient] = RDI[i] / daily_energy
        
        gen_kwargs.update({
            "nutrients_per_gram": W_per_g,  # Use aligned W (already in per-gram units)
            "nutrient_names": nutrient_names,
            "rdi_per_kcal": rdi_per_kcal,
            "nutrient_weights": {},  # Default equal weights
            "meal_energy_split": {0: 0.25, 1: 0.35, 2: 0.40},  # breakfast, lunch, dinner - aligned to 25/35/40%
            "energy_name": "energy_kcal",
            "bev_energy_frac_max": 0.25,  # Max 25% of meal energy from beverages
            "debug": True  # Enable debug output to show initial grams
        })
    
    gen = Generator(**gen_kwargs)
    
    # Generate meals for each meal type using Generator
    gen_by_type: Dict[str, np.ndarray] = {}
    clusters_gen_by_type: Dict[str, np.ndarray] = {}
    # Row indices are not cached (generation order != real order), so this stays empty
    # except for meal types whose generation fails; align_for_evaluation re-aligns by cluster.
    gen_row_indices_by_type: Dict[str, np.ndarray] = {}
    
    for mt in MEAL_TYPES:
        if mt not in real_by_type: 
            continue
            
        print(f"[GENERATOR] Generating meals for {mt}...")
        
        Cr_use = clusters_real_by_type[mt]

        # Map original cluster_id -> densified pair_code
        pc_for_cid = {cid: pc for pc, (mi, cid) in pair_table.items() if mi == MEAL_TO_IDX[mt]}
        
        # Filter real data to only include clusters that exist in the (possibly truncated) pair_table
        valid_mask = np.isin(Cr_use, list(pc_for_cid.keys()))
        if not valid_mask.any():
            print(f"[WARN] {mt}: no overlapping clusters with pair_table; skipping.")
            continue
            
        # Debug: show cluster mapping
        available_clusters = sorted(pc_for_cid.keys())
        real_clusters = sorted(np.unique(Cr_use))
        print(f"[GENERATOR] {mt}: available clusters in pair_table: {available_clusters}")
        print(f"[GENERATOR] {mt}: real clusters: {real_clusters}")
        print(f"[GENERATOR] {mt}: filtering {len(Cr_use)} -> {valid_mask.sum()} rows")
            
        # Real meals per archetype (only the counts are needed: one generated meal per real meal)
        Cr_filtered = Cr_use[valid_mask]

        # Generate samples per pair code (like the runner does)
        # Group real data by cluster to get counts per pair
        cluster_counts = Counter(Cr_filtered)
        remaining = int(getattr(args, "limit_total", 0) or 0)
        
        all_meals = []
        all_clusters = []
        

        # Generate samples for each pair code individually (like runner)
        for cluster_id, count in cluster_counts.items():
            if remaining == 0 and int(getattr(args, "limit_total", 0) or 0) > 0:
                break
            if cluster_id not in pc_for_cid:
                continue
                
            pair_code = pc_for_cid[cluster_id]
            
            # Get proper cluster name
            cluster_name = _get_cluster_name(mt, cluster_id)
            print(f"[GENERATOR] {mt}: generating {count} samples for {cluster_name} (pair {pair_code})")
            
            # Check this pair to catch alignment issues early
            try:
                n_allowed, top_features = check_pair(model, gate, pair_code, pair_table, feature_names)
                if n_allowed < 10:
                    print(f"[WARN] {cluster_name}: Only {n_allowed} items allowed - may cause milk-only generation")
            except Exception as e:
                print(f"[WARN] {cluster_name}: Pair check failed: {e}")
            
            try:
                # Determine generation count under test cap and oversampling
                gen_count = int(count) * max(int(getattr(args, "n_per_real", 1) or 1), 1)
                if int(getattr(args, "limit_total", 0) or 0) > 0:
                    gen_count = min(gen_count, remaining)
                if gen_count <= 0:
                    continue

                # Generate samples for this specific pair
                df_pair = gen.generate(
                    n=gen_count,
                    meal=mt,
                    pair_codes=[pair_code] * gen_count,  # All samples use same pair code
                    temperature=args.temperature,
                    nucleus_p=args.nucleus_p,
                    max_beverages=args.max_beverages,
                    diversity_lambda=args.diversity_lambda,
                    portion_strategy=args.portion_strategy,
                    min_grams=5,
                    jitter_sigma=0.15,
                    return_tidy=args.return_tidy,
                    seed=args.seed
                )
                
                # Show sample meals for this cluster
                if len(df_pair) > 0:
                    X_pair = df_pair[feature_names].values.astype(np.float32)
                    debug_n = min(2, len(df_pair))  # Show 2 samples per cluster
                    print(f"  [{cluster_name}] Sample meals:")
                    for i_dbg in range(debug_n):
                        grams = X_pair[i_dbg]
                        nz = np.where(grams > 0)[0]
                        if nz.size == 0:
                            print(f"    meal#{i_dbg}: (no items)")
                            continue
                        print(f"    meal#{i_dbg}: {nz.size} items")
                        for j in nz:
                            code = feature_names[j]
                            name = code2name.get(code, code)
                            category = food_to_group.get(code, "Unknown")
                            print(f"      - {name} [{code}] ({category}) grams={grams[j]:.1f}")

                all_meals.append(df_pair)
                all_clusters.extend([cluster_id] * len(df_pair))
                if int(getattr(args, "limit_total", 0) or 0) > 0:
                    remaining = max(0, remaining - len(df_pair))
                
            except Exception as e:
                print(f"[ERROR] {mt}: Generation failed for cluster {cluster_id}: {e}")
                continue
        
        if not all_meals:
            print(f"[WARN] {mt}: No meals generated successfully")
            continue
            
        # Combine all generated meals for this meal type
        df_meals = pd.concat(all_meals, ignore_index=True)

        try:
            # Align generated DataFrame to full feature space; fill missing items with zeros
            df_meals_aligned = df_meals.reindex(columns=feature_names, fill_value=0.0)
            X_gen = df_meals_aligned.values.astype(np.float32)
            C_gen = df_meals['cluster_id'].values.astype(np.int32)

            # Extra asserts and logging
            assert X_gen.shape[1] == len(feature_names), f"Generated D mismatch feature space: {X_gen.shape[1]} vs {len(feature_names)}"
            assert np.isfinite(X_gen).all(), "Non-finite grams in X_gen"
            assert (X_gen >= 0).all(), "Negative grams in X_gen"
            
            gen_by_type[mt] = X_gen
            clusters_gen_by_type[mt] = C_gen
            # NOTE: do not cache row indices; generation order != real order per cluster.


            # Per-cluster samples already shown above during generation

            counts = []
            for k in sorted(np.unique(C_gen)):
                mk = (C_gen == k)
                if not mk.any(): 
                    continue
                gen_n = int(mk.sum())
                target = cluster_item_stats.get(mt, {}).get(int(k), {}).get("median")
                obs    = int(np.median((X_gen[mk] > 0).sum(axis=1)))
                counts.append((int(k), gen_n, target, obs))
            print(f"[{mt}] per-cluster rows/targets: " +
                  ", ".join([f"k={k} gen={g} targetN={t} obsN={o}" for k,g,t,o in counts]))

            if "energy_kcal" in nutrient_names:
                kcal = fast_meal_nutrients_per_meal(X_gen, W, True)[:, nutrient_names.index("energy_kcal")]
                if np.median(X_gen.sum(axis=1)) < 150 or np.median(kcal) < 200:
                    print(f"[WARN] {mt}: low median grams/kcal; check item-count stats and assigner.")
                    
        except Exception as e:
            print(f"[ERROR] {mt}: Generation failed: {e}")
            # Create empty arrays for failed meal types
            gen_by_type[mt] = np.array([]).reshape(0, len(feature_names))
            clusters_gen_by_type[mt] = np.array([], dtype=np.int32)
            gen_row_indices_by_type[mt] = np.array([], dtype=int)

    # ===== SANITY CHECKS =====
    print("\n=== Running Sanity Checks ===")
    
    # 1. Determinism check (with fixed seed)
    print("→ Checking determinism...")
    try:
        # Generate a small sample twice with same seed using Generator
        test_mt = list(gen_by_type.keys())[0] if gen_by_type else None
        if test_mt:
            # Generate small sample twice with same seed
            df1 = gen.generate(n=5, meal=test_mt, seed=42)
            df2 = gen.generate(n=5, meal=test_mt, seed=42)
            
            X1 = df1[feature_names].values
            X2 = df2[feature_names].values
            
            if np.allclose(X1, X2):
                print("✓ Generation is deterministic with fixed seed")
            else:
                print("⚠ Generation is not deterministic. Ensure identical seed, temperature, nucleus_p, "
                      "and that any stochastic selection/refinement is disabled to reproduce exactly.")
    except Exception as e:
        print(f"⚠ Determinism check failed: {e}")
    
    # 2. Item-count fidelity by cluster
    print("→ Checking item-count fidelity...")
    for mt, Xg in gen_by_type.items():
        C = clusters_gen_by_type[mt]
        stats = cluster_item_stats[mt]
        for k in np.unique(C):
            mk = (C == k)
            if not mk.any(): continue
            want = stats.get(int(k), {}).get("median")
            got = int(np.median((Xg[mk] > 0).sum(axis=1)))
            if want is not None:
                if abs(got - want) <= 1:
                    print(f"✓ {mt} cluster {k}: items {got} ≈ target {want}")
                else:
                    print(f"⚠ {mt} cluster {k}: items {got} != target {want} (diff: {abs(got-want)})")
    
    # 3. Energy sanity vs meal fraction
    print("→ Checking energy sanity...")
    if "energy_kcal" in nutrient_names:
        eidx = nutrient_names.index("energy_kcal")
        frac = {"Breakfast": 0.25, "Lunch": 0.35, "Dinner": 0.40}
        for mt, Xg in gen_by_type.items():
            kcal = fast_meal_nutrients_per_meal(Xg, W, True)[:, eidx]
            tgt = 2000 * frac[mt]
            med = float(np.median(kcal)) if kcal.size else 0
            if 0.5*tgt <= med <= 1.5*tgt:
                print(f"✓ {mt}: median kcal {med:.1f} in plausible range around {tgt:.0f}")
            else:
                print(f"⚠ {mt}: median kcal {med:.1f} outside plausible range around {tgt:.0f}")
    
    # 4. No crazy outliers
    print("→ Checking for outliers...")
    for mt, Xg in gen_by_type.items():
        # Check for non-finite values
        if not np.isfinite(Xg).all():
            print(f"❌ {mt}: Found non-finite grams in generated meals")
        else:
            print(f"✓ {mt}: All grams are finite")
        
        # Check for excessive per-food grams
        if Xg.max() > 1000:
            print(f"⚠ {mt}: Some foods have >1000g (max: {Xg.max():.1f}g)")
        else:
            print(f"✓ {mt}: Per-food grams reasonable (max: {Xg.max():.1f}g)")
        
        # Check for excessive kcal
        if "energy_kcal" in nutrient_names:
            eidx = nutrient_names.index("energy_kcal")
            kcals = fast_meal_nutrients_per_meal(Xg, W, True)[:, eidx]
            if (kcals > 10000).any():
                print(f"⚠ {mt}: Some meals exceed 10k kcal (max: {kcals.max():.0f} kcal)")
            else:
                print(f"✓ {mt}: Meal kcal reasonable (max: {kcals.max():.0f} kcal)")
    
    # 5. Feature-space mapping correctness
    print("→ Checking feature-space mapping...")
    if all(len(c) == 8 for c in feature_names):
        print("✓ feature_names are 8-digit strings")
    else:
        print("❌ feature_names must be 8-digit strings")
    
    # 6. Nutrition breakdown sanity check
    print("→ Checking nutrition breakdown for example meals...")
    try:
        # Use the nutrition data that was already loaded
        # W and RDI are already available from the main data loading section
        
        # Show breakdown for each meal type
        for mt in MEAL_TYPES:
            if mt in gen_by_type and len(gen_by_type[mt]) > 0:
                print(f"\n--- {mt} Nutrition Breakdown ---")
                try:
                    # Generate a small sample for breakdown analysis
                    df_breakdown, table_breakdown = gen.show_generated_breakdown(
                        n=3,  # Show 3 example meals
                        meal_label=mt,
                        W=W,
                        nutrient_names=nutrient_names,
                        RDI_day=RDI,
                        portion_strategy=args.portion_strategy,
                        max_rows=3,
                        seed=args.seed
                    )
                    
                    # Print the breakdown table
                    print(table_breakdown.to_string(index=False))
                    
                    # Quick summary
                    if len(table_breakdown) > 0:
                        avg_energy = table_breakdown['energy_kcal'].mean()
                        avg_protein = table_breakdown['protein_g'].mean()
                        print(f"  Average: {avg_energy:.1f} kcal, {avg_protein:.1f}g protein")
                        
                except Exception as e:
                    print(f"⚠ {mt}: Nutrition breakdown failed: {e}")
            else:
                print(f"⚠ {mt}: No generated meals available for nutrition breakdown")
                
    except Exception as e:
        print(f"⚠ Nutrition breakdown sanity check failed: {e}")
    
    print("=== Sanity Checks Complete ===\n")


    # Fresh generation always produces grams, so run kcal sanity checks
    if "energy_kcal" in nutrient_names:
        eidx = nutrient_names.index("energy_kcal")
        for mt, Xg in gen_by_type.items():
            if Xg.size == 0:
                continue
            Ng = fast_meal_nutrients_per_meal(Xg, W, prefer_sparse=True)
            kcals = Ng[:, eidx]
            # Check for extreme values (explosion)
            extreme_mask = kcals > 10000  # More than 10k kcal is unreasonable
            if np.any(extreme_mask):
                print(f"[WARNING] {mt}: {np.sum(extreme_mask)} meals have >10k kcal (max: {kcals.max():.0f})")
                # Cap extreme values for sanity
                kcals_capped = np.clip(kcals, 0, 10000)
                print(f"[SANITY] GEN  {mt:9s} kcal  min={kcals_capped.min():.1f}  "
                      f"median={np.median(kcals_capped):.1f}  max={kcals_capped.max():.1f}  n={len(kcals_capped)}")
            else:
                print(f"[SANITY] GEN  {mt:9s} kcal  min={kcals.min():.1f}  "
                      f"median={np.median(kcals):.1f}  max={kcals.max():.1f}  n={len(kcals)}")
            
            # Check for collapse (too low kcal)
            if kcals.size and np.median(kcals) < 150:
                print(f"[WARN] {mt}: median kcal {np.median(kcals):.1f} looks too low. "
                      f"Check presence_to_grams / portion assignment.")
            
            # Check for zero-gram meals
            if (Xg.sum(axis=1) == 0).any():
                print(f"❌ {mt}: Found zero-gram meals after portioning")

    # 8) Save exports for evaluation
    save_exports(
        exports_dir,  # Save to the output directory, not project root
        real_by_type,
        gen_by_type,
        clusters_real_by_type,
        clusters_gen_by_type,
        W,
        nutrient_names,
        RDI,
        feature_names,
        food_to_group,
        code2name,
        gen_row_indices_by_type,
    )
    print(f"✓ Exports saved to {exports_dir}")
    
    return {
        "real_meals": real_by_type,
        "generated_meals": gen_by_type,
        "clusters": clusters_real_by_type,
        "clusters_gen": clusters_gen_by_type,
        "W": W,
        "nutrient_names": nutrient_names,
        "RDI": RDI,
        "feature_names": feature_names,
        "food_to_group": food_to_group,
        "gen_row_indices_by_meal": gen_row_indices_by_type,
        "pair_table": pair_table,
        "meal_type_mapping": meal_type_mapping,
    }


def run_pipeline(args):
    # Initialize file logger and redirect stdout/stderr
    try:
        from pathlib import Path as _P
        _log_path = _P.cwd() / 'logs' / 'evaluate' / 'evaluate.log'
        os.makedirs(_log_path.parent, exist_ok=True)
        _logger = _setup_logger('evaluate', str(_log_path))
        sys.stdout = _StreamToLogger(_logger, logging.INFO)
        sys.stderr = _StreamToLogger(_logger, logging.ERROR)
    except Exception:
        pass

    set_random_seeds(args.seed)

    project_root = Path(args.project_root).resolve()
    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    # Keep data exports in the output directory
    exports_dir = out_dir / "data_exports"
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Log policy settings for clarity
    print(f"[INFO] Cached presence→grams policy: {args.presence_to_grams} | "
          f"Generator portioning: {args.portion_strategy}")

    # Auto-discover cluster filter if needed
    cluster_filter_path = Path(args.cluster_filter) if args.cluster_filter else None
    if cluster_filter_path is None:
        for cand in [Path(args.model_dir)/"filters.json",
                     Path(args.model_dir)/"presence_only_model"/"filters.json",
                     Path(args.project_root)/"config"/"cluster_filter_default.json",
                     Path("config/cluster_filter_default.json")]:
            if cand.exists():
                cluster_filter_path = cand
                break
    cluster_filter = _load_cluster_filter(cluster_filter_path)

    # Load and prepare data
    nutrient_df, code2name, serving_df = load_and_prepare_data(args, project_root)
    # Check for cached data or load fresh
    data = load_cached_or_fresh_data(args, exports_dir, project_root, code2name, serving_df)

    if data is None:
        # Load fresh data and generate meals
        data = load_fresh_data_and_generate(project_root, model_dir, args, nutrient_df, code2name, serving_df, exports_dir, cluster_filter)
        if data is None:
            return 1
        attach_spider_nutrition(data, project_root, data["feature_names"])

    # Check if we have any generated data to evaluate
    if not data.get("generated_meals") or all(len(v) == 0 for v in data["generated_meals"].values()):
        print("[ERROR] No generated meals available; skipping evaluation.")
        return 1

    # Run evaluation and visualization
    return run_evaluation_and_visualization(data, code2name, args, out_dir, project_root, exports_dir, figures_dir, cluster_filter)


def main():
    ap = argparse.ArgumentParser(description="Generate CVAE meals per cluster and run evaluation/figures.")
    ap.add_argument("--project_root", type=str, default=".", help="Repo root containing outputs/ and data/ dirs.")
    ap.add_argument("--model_dir", type=str, required=False, default="outputs/runs/eval_run/presence_only_model",
                    help="Path to trained CVAE dir (with config + weights).")
    ap.add_argument("--out_dir", type=str, required=False, default="outputs/runs/eval_run",
                    help="Directory to write cached exports + figures.")
    ap.add_argument("--force_regen", action="store_true", help="Rebuild exports even if cache exists.")
    ap.add_argument("--n_per_real", type=int, default=1, help="Number of synthetic meals to generate per real meal (oversampling).")
    ap.add_argument("--limit_total", type=int, default=0, help="Cap on total generated meals per meal type for quick tests (0 = no cap).")
    ap.add_argument("--generation_temp", type=float, default=1.0, help="Temperature for generation sampling.")
    ap.add_argument("--generation_nucleus_p", type=float, default=0.9, help="Nucleus sampling parameter for generation.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument("--presence_to_grams", type=str, default="rdi_per_kcal", choices=["rdi_per_kcal"], help="How to convert presence probabilities to grams (rdi_per_kcal only).")
    ap.add_argument("--top_k", type=int, default=10, help="Top-k items to consider for portion assignment.")
    ap.add_argument("--min_prob", type=float, default=0.1, help="Minimum probability threshold for item selection.")
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic portion assignment.")
    
    # Generator arguments
    ap.add_argument("--portion_strategy", type=str, default="rdi_per_kcal", choices=["servings","assign","rdi_per_kcal"], help="Portioning strategy for Generator")
    ap.add_argument("--temperature", type=float, default=1.0, help="Temperature for Generator sampling")
    ap.add_argument("--nucleus_p", type=float, default=1.0, help="Nucleus sampling parameter for Generator")
    ap.add_argument("--max_beverages", type=int, default=1, help="Maximum beverages per meal for Generator")
    ap.add_argument("--diversity_lambda", type=float, default=1.0, help="Diversity lambda for Generator")
    ap.add_argument("--return_tidy", action="store_true", help="Return tidy format with items list for Generator")
    ap.add_argument("--cluster_filter", type=str, default=None,
                    help="Path to cluster filter JSON (clusters to KEEP/OMIT per meal). If omitted, tries model_dir filters and config/cluster_filter_default.json")
    
    # Cost inputs
    ap.add_argument("--price_map", type=str, default="data/prices/price_map.csv",
                    help="CSV with columns: food_code, food_name, price_per_100g_usd")
    ap.add_argument("--cost_weight", type=float, default=0.0,
                    help="Linear cost penalty weight in Stage-2 refinement (USD per 100g scaling)")

    args = ap.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
