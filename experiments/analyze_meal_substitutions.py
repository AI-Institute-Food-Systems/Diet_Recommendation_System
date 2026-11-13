#!/usr/bin/env python3
"""
Food-substitution analysis (Figure 4, S1 Tables 14-15).

For each meal type, every generated meal is compared with the real meals that pass the
calipers (exactly --require_k_substitution swapped foods, k = max(#added, #removed), else
at most --max_food_diff; energy within --kcal_caliper; item count within --items_caliper).
The --k_nearest most similar candidates (Jaccard/cosine blend) are scored:
  ΔRDI  = reduction in mean |nutrient / target - 1| over nutrients with a target, where the
          target is the daily RDI times the meal's energy share (0.25 / 0.35 / 0.40);
  ΔCost = generated - real cost (portion-based restaurant pricing with
          --use_restaurant_portion_pricing, otherwise --price_map_csv per-100 g prices
          times built-in food-category multipliers);
  score = w*ΔRDI% + (1-w)*savings% - w*cost-increase%, with w = θ / (1 + θ).
Candidates with ΔRDI < 0 or a negative score are dropped, category gates and the
mixed-dish penalties apply, and one winner is kept per generated meal.

Inputs (--data_dir, a run folder or its data_exports/): generated_meals_by_type.pkl,
real_meals_by_type.pkl, nutrient_matrix.npy, nutrient_names.json, RDI.npy,
feature_names.json, food_code_to_name.json, food_to_group.json.
Outputs (--output_dir): substitution_matches.csv, substitution_summary.csv,
substitution_stats_by_k.csv, substitution_stats_by_k_overall.csv,
substitution_stats_overall.csv, bootstrap_results.csv, soft_constraint_X.json,
substitution_examples_detailed.csv/.txt.

`make substitutions` and `make theta_sweep` run it once per k and θ, e.g.
    python -m experiments.analyze_meal_substitutions \
        --data_dir outputs/runs/eval_run \
        --output_dir outputs/runs/eval_run/substitutions_k3/tradeoff_theta_0p5_k1 \
        --require_k_substitution 1 --theta 0.5 --mixed_cross_alpha 0.10 --mixed_cross_beta 0.5 \
        $(SUB_ARGS)    # see the Makefile
`make sup-substitutions` runs it with the script defaults (S1 Tables 14-15, S1 section 3.5).
"""

import argparse
import logging
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys
from src.utils.helpers import setup_logger as _setup_logger

class _StreamToLogger:
    def __init__(self, logger: logging.Logger, level: int):
        self._logger = logger
        self._level = level
    def write(self, buf: str):
        for line in buf.rstrip().splitlines():
            self._logger.log(self._level, line.rstrip())
    def flush(self):
        pass
from tqdm import tqdm
try:
    # Shared restaurant pricing layer
    from experiments.costs.restaurant_pricing import (
        estimate_restaurant_costs_for_meals as _rest_costs_for_meals,
        estimate_restaurant_cost as _rest_cost,
    )
except Exception:
    _rest_costs_for_meals = None  # type: ignore
    _rest_cost = None  # type: ignore

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Global cache for food name lookups
_food_name_cache = {}

def get_food_name_cached(food_code: str, code2name: Dict[str, str]) -> str:
    """Get food name with caching to avoid repeated dictionary lookups."""
    if food_code not in _food_name_cache:
        _food_name_cache[food_code] = code2name.get(food_code, food_code)
    return _food_name_cache[food_code]

def preload_food_name_cache(feature_names: List[str], code2name: Dict[str, str]) -> None:
    """Preload all food names into cache for maximum efficiency."""
    print("  Preloading food name cache...")
    for food_code in feature_names:
        if food_code not in _food_name_cache:
            _food_name_cache[food_code] = code2name.get(food_code, food_code)
    print(f"  Preloaded {len(_food_name_cache)} food names into cache")

def clear_food_name_cache():
    """Clear the food name cache."""
    _food_name_cache.clear()

def get_cache_stats():
    """Get cache statistics."""
    return {
        'cache_size': len(_food_name_cache),
        'cached_items': list(_food_name_cache.keys())[:10]  # First 10 items as sample
    }

def save_nutrient_cache(gen_nutrients: np.ndarray, real_nutrients: np.ndarray, 
                       gen_deviations: np.ndarray, real_deviations: np.ndarray,
                       meal_type: str, cache_dir: Path) -> None:
    """Save precomputed nutrients to disk."""
    cache_dir.mkdir(exist_ok=True)
    
    np.save(cache_dir / f'{meal_type}_gen_nutrients.npy', gen_nutrients)
    np.save(cache_dir / f'{meal_type}_real_nutrients.npy', real_nutrients)
    np.save(cache_dir / f'{meal_type}_gen_deviations.npy', gen_deviations)
    np.save(cache_dir / f'{meal_type}_real_deviations.npy', real_deviations)
    
    print(f"  Saved nutrient cache for {meal_type}")

def load_nutrient_cache(meal_type: str, cache_dir: Path, expected_gen_shape: Tuple[int, int], expected_real_shape: Tuple[int, int]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load precomputed nutrients from disk with shape validation."""
    try:
        gen_nutrients = np.load(cache_dir / f'{meal_type}_gen_nutrients.npy')
        real_nutrients = np.load(cache_dir / f'{meal_type}_real_nutrients.npy')
        gen_deviations = np.load(cache_dir / f'{meal_type}_gen_deviations.npy')
        real_deviations = np.load(cache_dir / f'{meal_type}_real_deviations.npy')
        
        # Validate shapes match current dataset
        if (gen_nutrients.shape != expected_gen_shape or 
            real_nutrients.shape != expected_real_shape or
            gen_deviations.shape != (expected_gen_shape[0],) or
            real_deviations.shape != (expected_real_shape[0],)):
            print(f"  Cache shape mismatch for {meal_type}, recomputing...")
            print(f"    Expected gen: {expected_gen_shape}, got: {gen_nutrients.shape}")
            print(f"    Expected real: {expected_real_shape}, got: {real_nutrients.shape}")
            return None
        
        print(f"  Loaded nutrient cache for {meal_type}")
        return gen_nutrients, real_nutrients, gen_deviations, real_deviations
    except FileNotFoundError:
        return None

def load_data(data_dir: Path) -> Dict[str, Any]:
    """Load all required data files for the analysis."""
    print(f"Loading data from {data_dir}")
    data: Dict[str, Any] = {}
    # Fallback to data_exports if top-level files are missing
    root = data_dir
    de = data_dir / 'data_exports'
    gen_pkl = root / 'generated_meals_by_type.pkl'
    real_pkl = root / 'real_meals_by_type.pkl'
    if not gen_pkl.exists() and (de / 'generated_meals_by_type.pkl').exists():
        gen_pkl = de / 'generated_meals_by_type.pkl'
    if not real_pkl.exists() and (de / 'real_meals_by_type.pkl').exists():
        real_pkl = de / 'real_meals_by_type.pkl'
    # Load meal data
    data['gen_by_type'] = pickle.load(open(gen_pkl, 'rb'))
    data['real_by_type'] = pickle.load(open(real_pkl, 'rb'))
    
    # Load nutrient data
    nm_np = root / 'nutrient_matrix.npy'
    nn_js = root / 'nutrient_names.json'
    rdi_np = root / 'RDI.npy'
    fn_js = root / 'feature_names.json'
    if not nm_np.exists() and (de / 'nutrient_matrix.npy').exists():
        nm_np = de / 'nutrient_matrix.npy'
    if not nn_js.exists() and (de / 'nutrient_names.json').exists():
        nn_js = de / 'nutrient_names.json'
    if not rdi_np.exists() and (de / 'RDI.npy').exists():
        rdi_np = de / 'RDI.npy'
    if not fn_js.exists() and (de / 'feature_names.json').exists():
        fn_js = de / 'feature_names.json'
    data['nutrient_matrix'] = np.load(nm_np)
    data['nutrient_names'] = json.load(open(nn_js, 'r'))
    data['rdi_targets'] = np.load(rdi_np)
    data['feature_names'] = json.load(open(fn_js, 'r'))
    
    # Load food name mapping
    c2n_js = root / 'food_code_to_name.json'
    if not c2n_js.exists() and (de / 'food_code_to_name.json').exists():
        c2n_js = de / 'food_code_to_name.json'
    data['code2name'] = json.load(open(c2n_js, 'r'))
    # Optional food_to_group mapping (for category multipliers)
    try:
        ftg_js = root / 'food_to_group.json'
        if not ftg_js.exists() and (de / 'food_to_group.json').exists():
            ftg_js = de / 'food_to_group.json'
        data['food_to_group'] = json.load(open(ftg_js, 'r')) if ftg_js.exists() else {}
    except Exception:
        data['food_to_group'] = {}
    
    print("Loaded data:")
    print(f"  Generated meals: {sum(len(meals) for meals in data['gen_by_type'].values())} total")
    print(f"  Real meals: {sum(len(meals) for meals in data['real_by_type'].values())} total")
    print(f"  Foods: {len(data['feature_names'])}")
    print(f"  Nutrients: {len(data['nutrient_names'])}")
    
    return data


def calculate_food_differences(gen_meal: np.ndarray, real_meal: np.ndarray, 
                             feature_names: List[str], 
                             portion_abs_threshold_g: float = 10.0,
                             portion_rel_threshold: float = 0.20) -> Tuple[int, int, int, List[str], List[str], List[Tuple[str, float, float]]]:
    """
    Calculate food differences between generated and real meals.
    
    Args:
        gen_meal: Generated meal vector (n_foods,)
        real_meal: Real meal vector (n_foods,)
        feature_names: List of food codes
        portion_abs_threshold_g, portion_rel_threshold: a food present in both meals counts
            as a changed portion when |Δg| > max(abs threshold, rel threshold * real grams)

    Returns:
        (n_added, n_removed, n_changed_portions, added_foods, removed_foods, changed_foods)
    """
    # Find foods present in each meal (portion > 0)
    gen_present = gen_meal > 0
    real_present = real_meal > 0
    
    # Foods only in generated (added)
    added_mask = gen_present & ~real_present
    added_foods = [feature_names[i] for i in np.where(added_mask)[0]]
    n_added = len(added_foods)
    
    # Foods only in real (removed)
    removed_mask = real_present & ~gen_present
    removed_foods = [feature_names[i] for i in np.where(removed_mask)[0]]
    n_removed = len(removed_foods)
    
    # Foods in both but with different portions (count change if Δg > max(abs, rel))
    both_present = gen_present & real_present
    portion_diff = np.abs(gen_meal - real_meal)
    rel_thresh = portion_rel_threshold * np.maximum(real_meal, 1e-6)
    abs_thresh = np.full_like(portion_diff, float(portion_abs_threshold_g))
    change_thresh = np.maximum(abs_thresh, rel_thresh)
    changed_mask = both_present & (portion_diff > change_thresh)
    
    changed_foods = []
    for i in np.where(changed_mask)[0]:
        food_code = feature_names[i]
        old_portion = real_meal[i]
        new_portion = gen_meal[i]
        changed_foods.append((food_code, old_portion, new_portion))
    
    n_changed_portions = len(changed_foods)
    
    return n_added, n_removed, n_changed_portions, added_foods, removed_foods, changed_foods


def find_best_substitutions(gen_meals: np.ndarray, real_meals: np.ndarray,
                          W: np.ndarray, RDI: np.ndarray, nutrient_names: List[str],
                          feature_names: List[str], code2name: Dict[str, str], meal_type: str, 
                          max_food_diff: int = 2, debug: bool = False, show_all_matches: bool = False,
                          cache_dir: Optional[Path] = None, use_cache: bool = False,
                          k_nearest: int = 3, kcal_caliper: float = 0.05, items_caliper: int = 1,
                          jaccard_weight: float = 0.7,
                          portion_abs_threshold_g: float = 10.0,
                          portion_rel_threshold: float = 0.20,
                          price_per_100g: Optional[np.ndarray] = None,
                          no_cost_increase: bool = False,
                          budget_cap: Optional[float] = None,
                          theta: float = 0.5,
                          selection_mode: str = 'tradeoff',
                          tie_band_pct: float = 0.01,
                          restaurant_overhead_usd: float = 0.0,
                          use_restaurant_portion_pricing: bool = False,
                          food_to_group: Optional[Dict[str, str]] = None,
                          x_over_within_pct: Optional[float] = None,
                          cross_rel_uplift_alpha: Optional[float] = None,
                          cross_abs_margin_beta: Optional[float] = None,
                          category_first: bool = False,
                          category_only: bool = False,
                          category_universe: str = 'cluster',
                          require_k_substitution: Optional[int] = None,
                          force_within_category: bool = False,
                          omit_add_categories: Optional[set] = None,
                          omit_remove_categories: Optional[set] = None,
                          penalize_add_categories: Optional[set] = None,
                          gamma_penalize_add: float = 0.0,
                          mixed_cross_alpha: Optional[float] = None,
                          mixed_cross_beta: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Find substitution candidates and the winner for each generated meal of one meal type.

    Args:
        gen_meals: Generated meals matrix (n_gen, n_foods), grams
        real_meals: Real meals matrix (n_real, n_foods), grams
        W: Nutrient matrix (n_foods, n_nutrients), per 100 g
        RDI: Daily targets (n_nutrients,), scaled by the meal's energy share
        nutrient_names: List of nutrient names
        feature_names: List of food codes
        meal_type: 'Breakfast', 'Lunch' or 'Dinner'
        max_food_diff: Maximum k when require_k_substitution is None

    Returns:
        The selected substitution (winner) of each generated meal that has one
    """
    substitutions = []
    
    print(f"Finding substitutions for {meal_type}: {len(gen_meals)} generated vs {len(real_meals)} real meals")
    
    # Precompute meal fractions for efficiency
    meal_fractions = {'Breakfast': 0.25, 'Lunch': 0.35, 'Dinner': 0.40}
    meal_fraction = meal_fractions.get(meal_type, 0.33)
    rdi_meal = RDI * meal_fraction
    
    # Try to load precomputed nutrients from cache
    cached_nutrients = None
    if use_cache and cache_dir:
        expected_gen_shape = (len(gen_meals), W.shape[1])
        expected_real_shape = (len(real_meals), W.shape[1])
        cached_nutrients = load_nutrient_cache(meal_type, cache_dir, expected_gen_shape, expected_real_shape)
    
    if cached_nutrients is not None:
        gen_nutrients, real_nutrients, gen_deviations, real_deviations = cached_nutrients
        print("  Using cached nutrients")
        # Still need to compute RDI ratios for detailed analysis
        gen_rdi_ratios = gen_nutrients / (rdi_meal + 1e-8)
        real_rdi_ratios = real_nutrients / (rdi_meal + 1e-8)
    else:
        # Precompute nutrients for all meals to avoid repeated calculations
        print("  Precomputing nutrients for all meals...")
        gen_nutrients = (gen_meals / 100.0) @ W  # (n_gen, n_nutrients)
        real_nutrients = (real_meals / 100.0) @ W  # (n_real, n_nutrients)
        
        # Debug RDI targets (NaN-robust)
        if debug:
            try:
                print(f"    RDI targets: min={np.nanmin(RDI):.4f}, max={np.nanmax(RDI):.4f}, mean={np.nanmean(RDI):.4f}")
                print(f"    RDI meal targets: min={np.nanmin(rdi_meal):.4f}, max={np.nanmax(rdi_meal):.4f}, mean={np.nanmean(rdi_meal):.4f}")
                finite_meal = np.isfinite(rdi_meal)
                zeros = int(np.sum((rdi_meal == 0) & finite_meal))
                print(f"    Zero RDI targets: {zeros} out of {int(np.sum(finite_meal))}")
            except Exception:
                pass
        
        # Precompute RDI deviations for all meals
        gen_rdi_ratios = gen_nutrients / (rdi_meal + 1e-8)
        real_rdi_ratios = real_nutrients / (rdi_meal + 1e-8)
        
        valid_mask = np.isfinite(rdi_meal) & (rdi_meal > 0)
        gen_deviations = np.mean(np.abs(gen_rdi_ratios[:, valid_mask] - 1.0), axis=1)
        real_deviations = np.mean(np.abs(real_rdi_ratios[:, valid_mask] - 1.0), axis=1)
        
        # Save to cache if requested
        if use_cache and cache_dir:
            save_nutrient_cache(gen_nutrients, real_nutrients, gen_deviations, real_deviations, meal_type, cache_dir)
    
    # Candidate search (NumPy)
    xp = np

    n_real, n_foods = real_meals.shape
    # Precompute real presence and counts
    real_meals_xp = xp.asarray(real_meals)
    real_present = real_meals_xp > 0
    real_items = real_present.sum(axis=1)
    # Energy index
    energy_idx = None
    for i, n in enumerate(nutrient_names):
        if 'energy' in n.lower() and 'kcal' in n.lower():
            energy_idx = i
            break
    # Energy caliper can remain on CPU for simplicity
    real_energy = ((real_meals / 100.0) @ W[:, energy_idx]) if energy_idx is not None else None
    real_norms = xp.linalg.norm(real_meals_xp.astype(float), axis=1) + 1e-8
    real_costs = None
    price_g_xp = None
    if use_restaurant_portion_pricing and (_rest_costs_for_meals is not None):
        try:
            rc_np = _rest_costs_for_meals(real_meals, feature_names, code2name, overhead_usd=0.0)
            real_costs = xp.asarray(rc_np)
        except Exception:
            real_costs = None
    elif price_per_100g is not None:
        price_g = (price_per_100g.astype(float) / 100.0)
        price_g_xp = xp.asarray(price_g)
        real_costs = (real_meals_xp.astype(float) * price_g_xp).sum(axis=1)

    w = float(np.clip(jaccard_weight, 0.0, 1.0))

    # Helper for strict WWEIA same-category check using provided mapping
    def _cat_of(code: str) -> Optional[str]:
        if not food_to_group:
            return None
        c = str(code).zfill(8)
        val = food_to_group.get(c)
        if val is None:
            return None
        return str(val).strip().lower().replace(" ", "_") or None

    def _is_same_category(added: List[str], removed: List[str]) -> Optional[int]:
        if not food_to_group:
            return None
        # If only portions change (no adds/removes), treat as within-category
        if (not added) and (not removed):
            return 1
        if not added or not removed:
            return 0
        rem_cats = list({ _cat_of(fc) for fc in removed if _cat_of(fc) is not None })
        add_cats = list({ _cat_of(fc) for fc in added if _cat_of(fc) is not None })
        if len(rem_cats) == 1 and len(add_cats) == 1 and rem_cats[0] == add_cats[0]:
            return 1
        return 0

    # Winner selection helper (unified policy)
    def _select_winner(pool: List[Dict[str, Any]], mode: str) -> Optional[Dict[str, Any]]:
        if not pool:
            return None
        # If forcing within-category, restrict to within when any finite-scored within candidates exist
        if force_within_category:
            finite_pool = [c for c in pool if np.isfinite(c.get('score', np.nan))]
            within_pool = [c for c in finite_pool if c.get('same_category', None) == 1]
            if within_pool:
                best_score = max([c.get('score', -np.inf) for c in within_pool])
                eps = max(1e-9, abs(best_score) * float(max(tie_band_pct, 0.0)))
                near_best = [c for c in within_pool if abs(c.get('score', -np.inf) - best_score) <= eps]
                if len(near_best) > 1:
                    near_best.sort(key=lambda c: (float(c.get('portion_shift_pct', np.inf)),
                                                  -float(c.get('dRDI_pct', c.get('improvement', -np.inf)*100.0)),
                                                  float(c.get('delta_cost', np.inf))))
                    return near_best[0]
                return max(within_pool, key=lambda c: c.get('score', -np.inf))
        if mode == 'dominance':
            dom = [c for c in pool if (c.get('improvement', 0) > 0) and (np.isfinite(c.get('delta_cost')) and c.get('delta_cost') <= 0)]
            cand_pool = dom if dom else pool
            if not cand_pool:
                return max(pool, key=lambda c: c.get('improvement', -np.inf))
            ps = np.array([float(x.get('portion_shift_pct', np.inf)) for x in cand_pool], dtype=float)
            if ps.size == 0:
                # Unified rule: min portion shift, then max ΔRDI, then prefer savings
                return min(cand_pool, key=lambda c: (float(c.get('portion_shift_pct', np.inf)), -float(c.get('improvement', -np.inf)), float(c.get('delta_cost', np.inf))))
            mps = float(np.nanmin(ps))
            band = mps * float(max(tie_band_pct, 0.0))
            near = [c for c in cand_pool if (float(c.get('portion_shift_pct', np.inf)) - mps) <= band]
            if len(near) > 1:
                near.sort(key=lambda c: (-float(c.get('improvement', -np.inf)), float(c.get('delta_cost', np.inf))))
                return near[0]
            return min(cand_pool, key=lambda c: float(c.get('portion_shift_pct', np.inf)))
        # tradeoff
        finite_cands = [c for c in pool if np.isfinite(c.get('score', np.nan))]
        # Enforce exact k_substitution STRICTLY: if none match, return None (no fallback)
        if (require_k_substitution is not None):
            k_pool = [c for c in finite_cands if int(c.get('k_substitution', -1)) == int(require_k_substitution)]
            if not k_pool:
                return None
            finite_cands = k_pool
        cand_pool = finite_cands if finite_cands else pool
        if not cand_pool:
            return max(pool, key=lambda c: c.get('similarity', 0.0))

        # Category-specific cross acceptance: if cross adds mixed_dishes, require extra uplift
        try:
            if (mixed_cross_alpha is not None) or (mixed_cross_beta is not None):
                within = [c for c in cand_pool if c.get('same_category', None) == 1]
                cross = [c for c in cand_pool if c.get('same_category', None) != 1]
                if within and cross:
                    cross_md = [c for c in cross if 'mixed_dishes' in set(c.get('added_cats', []))]
                    if cross_md:
                        best_within = max(within, key=lambda c: c.get('score', -np.inf))
                        best_cross_md = max(cross_md, key=lambda c: c.get('score', -np.inf))
                        s_within = float(best_within.get('score', -np.inf))
                        s_cross = float(best_cross_md.get('score', -np.inf))
                        alpha_md = float(mixed_cross_alpha or 0.0)
                        beta_md = float(mixed_cross_beta or 0.0)
                        threshold = (1.0 + alpha_md) * s_within + beta_md
                        if s_cross < threshold:
                            return best_within
                        return best_cross_md
        except Exception:
            pass

        # Apply cross-category acceptance policy
        # 1) Dual-margin rule if provided: accept cross only if
        #    S_cross >= S_within * (1 + alpha) + beta
        if (
            cross_rel_uplift_alpha is not None and isinstance(cross_rel_uplift_alpha, (int, float)) and
            cross_abs_margin_beta is not None and isinstance(cross_abs_margin_beta, (int, float))
        ):
            within = [c for c in cand_pool if c.get('same_category', None) == 1]
            cross = [c for c in cand_pool if c.get('same_category', None) != 1]
            if within and cross:
                best_within = max(within, key=lambda c: c.get('score', -np.inf))
                best_cross = max(cross, key=lambda c: c.get('score', -np.inf))
                s_within = float(best_within.get('score', -np.inf))
                s_cross = float(best_cross.get('score', -np.inf))
                threshold = (1.0 + float(cross_rel_uplift_alpha)) * s_within + float(cross_abs_margin_beta)
                return best_cross if s_cross >= threshold else best_within
            if within:
                return max(within, key=lambda c: c.get('score', -np.inf))
            if cross:
                return max(cross, key=lambda c: c.get('score', -np.inf))

        # 2) Otherwise the relative uplift rule (--cross_over_within_pct, 20 by default):
        #    accept cross only if S_cross >= S_within * (1 + pct / 100)
        if isinstance(x_over_within_pct, (int, float)) and x_over_within_pct is not None:
            within = [c for c in cand_pool if c.get('same_category', None) == 1]
            cross = [c for c in cand_pool if c.get('same_category', None) != 1]
            if within and cross:
                best_within = max(within, key=lambda c: c.get('score', -np.inf))
                best_cross = max(cross, key=lambda c: c.get('score', -np.inf))
                thr = (1.0 + float(x_over_within_pct)/100.0) * float(best_within.get('score', -np.inf))
                return best_cross if float(best_cross.get('score', -np.inf)) >= thr else best_within
            if within:
                return max(within, key=lambda c: c.get('score', -np.inf))
            if cross:
                return max(cross, key=lambda c: c.get('score', -np.inf))

        best_score = max([c.get('score', -np.inf) for c in cand_pool])
        eps = max(1e-9, abs(best_score) * float(max(tie_band_pct, 0.0)))
        near_best = [c for c in cand_pool if abs(c.get('score', -np.inf) - best_score) <= eps]
        if len(near_best) > 1:
            # Tie-break: prefer within-category; then minimal portion shift; then larger ΔRDI%; then smaller ΔCost
            near_best.sort(key=lambda c: (0 if c.get('same_category', 0)==1 else 1,
                                          float(c.get('portion_shift_pct', np.inf)),
                                          -float(c.get('dRDI_pct', c.get('improvement', -np.inf)*100.0)),
                                          float(c.get('delta_cost', np.inf))))
            return near_best[0]
        return max(cand_pool, key=lambda c: c.get('score', -np.inf))

    for gen_idx in range(len(gen_meals)):
        gen_meal = gen_meals[gen_idx]
        gen_meal_xp = xp.asarray(gen_meal)
        gen_present = gen_meal_xp > 0
        gen_items = int(gen_present.sum())
        gen_norm = float(xp.linalg.norm(gen_meal_xp.astype(float)) + 1e-8)
        # Energy on CPU (small)
        gen_energy = ((gen_meal / 100.0) @ W[:, energy_idx]) if energy_idx is not None else None
        gen_cost = None
        if real_costs is not None:
            if use_restaurant_portion_pricing and (_rest_cost is not None):
                try:
                    g_np = xp.asnumpy(gen_meal_xp) if xp is not np else gen_meal
                    gen_cost = float(_rest_cost(g_np.astype(float), feature_names, code2name, overhead_usd=0.0))
                except Exception:
                    gen_cost = None
            elif price_g_xp is not None:
                gen_cost = float(xp.asnumpy((gen_meal_xp.astype(float) * price_g_xp).sum())) if xp is not np else float((gen_meal.astype(float) * (price_g_xp if price_g_xp is not None else 0)).sum())

        # Candidate mask: k, kcal and item-count calipers
        base_mask = xp.ones((n_real,), dtype=bool)
        # k_substitution caliper
        added_counts = (gen_present & (~real_present)).sum(axis=1)
        removed_counts = ((~gen_present) & real_present).sum(axis=1)
        k_sub_vec = xp.maximum(added_counts, removed_counts)
        if require_k_substitution is not None:
            base_mask &= (k_sub_vec == int(require_k_substitution))
        else:
            base_mask &= (k_sub_vec <= int(max_food_diff))
        # kcal caliper
        if (energy_idx is not None) and (gen_energy is not None) and (gen_energy > 0):
            rel = np.abs(real_energy - gen_energy) / np.maximum(real_energy, 1e-8)  # CPU array
            base_mask &= xp.asarray(rel <= float(kcal_caliper))
        # items caliper
        base_mask &= (xp.abs(real_items - gen_items) <= int(items_caliper))

        mask = base_mask
        if not bool(xp.any(mask)):
            continue

        # Similarities (item overlap + cosine)
        inter = (real_present & gen_present).sum(axis=1).astype(float)
        union = (real_present | gen_present).sum(axis=1).astype(float)
        jacc = inter / xp.maximum(union, 1.0)
        dots = real_meals_xp.astype(float) @ gen_meal_xp.astype(float)
        cos = dots / (real_norms * gen_norm)
        base_sim = w * jacc + (1.0 - w) * cos
        # With a food-category map, add extra weight on item overlap (Jaccard) as a proxy
        # for preserving food categories: sim = 0.85 * base_sim + 0.15 * jacc
        try:
            if food_to_group:
                cat_sim = jacc
                alpha_cat = 0.15
                sim = (1.0 - alpha_cat) * base_sim + alpha_cat * cat_sim
            else:
                sim = base_sim
        except Exception:
            sim = base_sim
        sim = sim.astype(float)
        sim = xp.where(mask, sim, -xp.inf)
        # Top-k indices (only needed if not category_only)
        idxs_np = np.array([], dtype=int)
        if not category_only:
            k_eff = max(1, int(k_nearest))
            if bool(xp.all(~xp.isfinite(sim))):
                continue
            try:
                idxs = xp.argpartition(sim, -k_eff)[-k_eff:]
            except Exception:
                idxs = xp.argsort(sim)[-k_eff:]
            idxs = idxs[xp.argsort(sim[idxs])[::-1]]
            idxs_np = xp.asnumpy(idxs) if xp is not np else idxs

        gen_cands: List[Dict[str, Any]] = []

        # Optional (--category_first): enumerate same-category single-food swaps (foods seen in
        # the cluster's real meals, or all foods with --category_universe all) before the pool
        if category_first and food_to_group:
            # Determine cluster scope for food universe
            if str(category_universe).lower() == 'all':
                # Allow any food among all features (full 480)
                cluster_food_idxs = np.arange(n_foods)
            else:
                # No cluster labels are passed in, so this universe spans all real meals
                cluster_mask = xp.ones((n_real,), dtype=bool)
                cluster_mask_np = xp.asnumpy(cluster_mask) if xp is not np else cluster_mask
                real_meals_cluster = real_meals[cluster_mask_np]
                # Unique foods present anywhere in this cluster
                cluster_food_idxs = np.where((real_meals_cluster > 0).any(axis=0))[0]
            # Base deviations/cost for generated meal
            gen_base_dev = float(gen_deviations[gen_idx])
            gen_base_cost = None
            price_g_np = None
            if use_restaurant_portion_pricing and (_rest_cost is not None):
                try:
                    gen_base_cost = float(_rest_cost(gen_meal.astype(float), feature_names, code2name, overhead_usd=0.0))
                except Exception:
                    gen_base_cost = None
            elif price_per_100g is not None:
                price_g_np = (price_per_100g.astype(float) / 100.0)
                gen_base_cost = float((gen_meal.astype(float) * price_g_np).sum())
            # Iterate present foods in generated meal
            present_food_idxs = np.where(np.asanyarray(gen_meal) > 0)[0]
            for i_idx in present_food_idxs:
                src_code = str(feature_names[i_idx])
                src_cat = _cat_of(src_code)
                if not src_cat:
                    continue
                grams_i = float(gen_meal[i_idx])
                if grams_i <= 0:
                    continue
                # Candidate foods from same category in cluster
                same_cat_j = []
                for j_idx in cluster_food_idxs:
                    if j_idx == i_idx:
                        continue
                    cand_code = str(feature_names[j_idx])
                    if _cat_of(cand_code) == src_cat:
                        same_cat_j.append(j_idx)
                if not same_cat_j:
                    continue
                # Evaluate swaps, keep best per source food
                best_local: Optional[Dict[str, Any]] = None
                for j_idx in same_cat_j:
                    # Construct alternate generated meal: swap grams from i->j
                    gen_alt = np.array(gen_meal, dtype=float, copy=True)
                    gen_alt[j_idx] += grams_i
                    gen_alt[i_idx] = 0.0
                    # Compute deviation for gen_alt
                    gen_alt_n = (gen_alt / 100.0) @ W
                    # Use same valid RDI indices as earlier
                    valid_idx = np.isfinite(RDI) & (RDI > 0)
                    gen_alt_ratios = gen_alt_n / (rdi_meal + 1e-8)
                    gen_alt_dev = float(np.mean(np.abs(gen_alt_ratios[valid_idx] - 1.0)))
                    delta_rdi_pct = float((gen_base_dev - gen_alt_dev) * 100.0)
                    # Cost change
                    delta_cost_pct = float('nan')
                    gen_alt_cost = None
                    if gen_base_cost is not None:
                        if use_restaurant_portion_pricing and (_rest_cost is not None):
                            try:
                                gen_alt_cost = float(_rest_cost(gen_alt.astype(float), feature_names, code2name, overhead_usd=0.0))
                            except Exception:
                                gen_alt_cost = None
                        elif price_g_np is not None:
                            gen_alt_cost = float((gen_alt * price_g_np).sum())
                        if (gen_alt_cost is not None) and (gen_base_cost != 0):
                            delta_cost_pct = float((gen_alt_cost - gen_base_cost) / gen_base_cost * 100.0)
                    # Score
                    if (delta_rdi_pct < 0.0) and (not show_all_matches):
                        score_local = float('nan')
                    else:
                        score_local = float(delta_rdi_pct) - float(theta) * float(max(0.0, delta_cost_pct if np.isfinite(delta_cost_pct) else 0.0))
                        if (not show_all_matches) and np.isfinite(score_local) and (score_local < 0.0):
                            score_local = float('nan')
                    if not np.isfinite(score_local):
                        continue
                    # Portion/comp changes
                    l1_grams = float(np.sum(np.abs(gen_alt - gen_meal)))
                    real_grams_total = float(np.sum(gen_meal))  # relative to original gen meal
                    portion_shift_pct = (l1_grams / max(real_grams_total, 1e-6)) * 100.0
                    n_added = 1 if gen_alt[j_idx] > 0 and gen_meal[j_idx] == 0 else 0
                    n_removed = 1 if gen_meal[i_idx] > 0 and gen_alt[i_idx] == 0 else 0
                    composition_change_pct_real = (100.0 * float(max(n_added, n_removed)) / max(float((gen_meal > 0).sum()), 1.0))
                    cand = {
                        'gen_idx': gen_idx,
                        'real_idx': int(-1),  # synthetic
                        'n_added': int(n_added),
                        'n_removed': int(n_removed),
                        'n_changed_portions': int(0),
                        'total_food_diff': int(max(n_added, n_removed)),
                        'k_substitution': 1,
                        'portion_shift_pct': float(portion_shift_pct),
                        'composition_change_pct': float(composition_change_pct_real),
                        'composition_change_pct_real': float(composition_change_pct_real),
                        'composition_change_pct_union': float(composition_change_pct_real),
                        'added_foods': [str(feature_names[j_idx])],
                        'removed_foods': [str(feature_names[i_idx])],
                        'changed_foods': [],
                        'gen_deviation': float(gen_alt_dev),
                        'real_deviation': float(gen_base_dev),
                        'improvement': float((gen_base_dev - gen_alt_dev)),
                        'per_nutrient_changes': {},
                        'delta_cost': float((gen_alt_cost - gen_base_cost)) if (gen_alt_cost is not None and gen_base_cost is not None) else np.nan,
                        'delta_mar': float('nan'),
                        'delta_amdr': float('nan'),
                        'delta_mer': float('nan'),
                        'delta_ed': float('nan'),
                        'real_cost': float(gen_base_cost) if gen_base_cost is not None else np.nan,
                        'gen_cost': float(gen_alt_cost) if gen_alt_cost is not None else np.nan,
                        'dRDI_pct': float(delta_rdi_pct),
                        'dCost_pct': float(delta_cost_pct),
                        'delta_protein_g': float('nan'),
                        'delta_fiber_g': float('nan'),
                        'delta_sodium_mg': float('nan'),
                        'score': float(score_local),
                        'same_category': 1,
                        'similarity': float('nan'),
                        'selected': False,
                    }
                    if (best_local is None) or (cand['score'] > best_local['score']):
                        best_local = cand
                if best_local is not None:
                    gen_cands.append(best_local)
        for real_idx in (idxs_np if not category_only else []):
            s_val = float(xp.asnumpy(sim[real_idx])) if xp is not np else float(sim[real_idx])
            if not np.isfinite(s_val):
                continue
            real_meal = real_meals[real_idx]
            n_added, n_removed, n_changed, added_foods, removed_foods, changed_foods = calculate_food_differences(
                gen_meal, real_meal, feature_names,
                portion_abs_threshold_g=portion_abs_threshold_g,
                portion_rel_threshold=portion_rel_threshold
            )
            # Map foods to WWEIA categories and apply omit gates
            try:
                def _cat_of_local(code: str) -> Optional[str]:
                    if not food_to_group:
                        return None
                    c = str(code).zfill(8)
                    v = food_to_group.get(c)
                    return (str(v).strip().lower().replace(" ", "_") or None) if v is not None else None
                def _cat_or_other(code: str) -> str:
                    cat = _cat_of_local(code)
                    return cat if cat is not None else 'other'
                added_cats = { _cat_or_other(fc) for fc in added_foods }
                removed_cats = { _cat_or_other(fc) for fc in removed_foods }
            except Exception:
                added_cats = set()
                removed_cats = set()
            if omit_add_categories and (added_cats & set(omit_add_categories)):
                continue
            if omit_remove_categories and (removed_cats & set(omit_remove_categories)):
                continue
            improvement = float(real_deviations[real_idx] - gen_deviations[gen_idx])
            # Cost
            dcost = float('nan')
            real_cost = float('nan')
            if real_costs is not None and gen_cost is not None:
                rc = real_costs[real_idx]
                real_cost = float(xp.asnumpy(rc)) if xp is not np else float(rc)
                dcost = float(gen_cost - real_cost)
                if no_cost_increase and dcost > 1e-9:
                    continue
                if budget_cap is not None:
                    eff_gen_cost = float(gen_cost + float(restaurant_overhead_usd))
                    if eff_gen_cost > float(budget_cap):
                        continue
            # Portion shift and endpoints (lightweight; heavy per-nutrient deltas later if needed)
            real_present_vec = (real_meal > 0)
            real_grams_total = float(np.sum(real_meal))
            l1_grams = float(np.sum(np.abs(gen_meal - real_meal)))
            portion_shift_pct = (l1_grams / max(real_grams_total, 1e-6)) * 100.0
            # Composition change (two variants)
            real_count = int(real_present_vec.sum()) if real_present_vec is not None else 0
            union_count = int(((gen_meal > 0) | (real_meal > 0)).sum())
            composition_change_pct_real = (100.0 * float(max(n_added, n_removed)) / max(float(real_count), 1.0)) if real_count > 0 else float('nan')
            composition_change_pct_union = (100.0 * float(max(n_added, n_removed)) / max(float(union_count), 1.0)) if union_count > 0 else float('nan')
            # Per-nutrient detailed changes
            gen_ratios = gen_rdi_ratios[gen_idx]
            real_ratios = real_rdi_ratios[real_idx]
            per_nutrient_changes = {}
            for i, nutrient in enumerate(nutrient_names):
                dev_red = float(abs(real_ratios[i] - 1.0) - abs(gen_ratios[i] - 1.0))
                per_nutrient_changes[nutrient] = {
                    'gen_ratio': float(gen_ratios[i]),
                    'real_ratio': float(real_ratios[i]),
                    'improvement': dev_red
                }
            # Additional endpoints
            def _idx(name_substr: str) -> Optional[int]:
                for ii, nn in enumerate(nutrient_names):
                    if name_substr in nn.lower():
                        return ii
                return None
            gen_n = gen_nutrients[gen_idx]
            real_n = real_nutrients[real_idx]
            valid_idx = np.isfinite(RDI) & (RDI > 0)
            dmar = float(np.mean(np.clip(gen_ratios[valid_idx], 0.0, 1.0)) - np.mean(np.clip(real_ratios[valid_idx], 0.0, 1.0)))
            e_idx = _idx('energy_kcal') or _idx('energy')
            p_idx = _idx('protein_g') or _idx('protein')
            f_idx = _idx('fat_g') or _idx('fat, total')
            c_idx = _idx('carbohydrate_g') or _idx('carb')
            def amdr(nv: np.ndarray) -> float:
                if e_idx is None or p_idx is None or f_idx is None or c_idx is None:
                    return float('nan')
                kcal = nv[e_idx]
                if kcal <= 0:
                    return float('nan')
                frac_p = (nv[p_idx]*4.0)/kcal
                frac_f = (nv[f_idx]*9.0)/kcal
                frac_c = (nv[c_idx]*4.0)/kcal
                return float(((0.10 <= frac_p <= 0.35) + (0.20 <= frac_f <= 0.35) + (0.45 <= frac_c <= 0.65))/3.0)
            damdr = amdr(gen_n) - amdr(real_n)
            def mer_excess(r: np.ndarray) -> float:
                idxs = []
                for key in ['sugars', 'saturated', 'sodium']:
                    j = _idx(key)
                    if j is not None:
                        idxs.append(j)
                if not idxs:
                    return float('nan')
                exc = np.clip(r[idxs] - 1.0, 0.0, None)
                return float(np.nansum(exc))
            dmer = mer_excess(real_ratios) - mer_excess(gen_ratios)
            def ed(nv: np.ndarray, grams: np.ndarray) -> float:
                if e_idx is None:
                    return float('nan')
                total_g = float(np.sum(grams))
                return float(nv[e_idx]/max(total_g, 1e-6))
            ded = ed(gen_n, gen_meal) - ed(real_n, real_meal)
            prot_idx = _idx('protein_g') or _idx('protein')
            fib_idx = _idx('fiber_g') or _idx('fiber')
            sod_idx = _idx('sodium_mg') or _idx('sodium')
            d_prot = float(gen_n[prot_idx] - real_n[prot_idx]) if prot_idx is not None else float('nan')
            d_fiber = float(gen_n[fib_idx] - real_n[fib_idx]) if fib_idx is not None else float('nan')
            d_sodium = float(gen_n[sod_idx] - real_n[sod_idx]) if sod_idx is not None else float('nan')

            # Weighted health-vs-savings objective with theta: w = theta / (1 + theta)
            # score = w * ΔRDI% + (1-w) * Savings% - w * CostIncrease%
            # Ensures theta trades off health and savings in both directions
            if np.isfinite(improvement) and np.isfinite(real_cost) and float(real_cost) != 0 and np.isfinite(dcost):
                delta_rdi_pct = float(improvement) * 100.0
                delta_cost_pct = float(dcost) / float(real_cost) * 100.0
                # Health floor: require ΔRDI% ≥ 0 unless show_all_matches
                if (delta_rdi_pct < 0.0) and (not show_all_matches):
                    score = float('nan')
                else:
                    theta_w = float(theta) / float(1.0 + float(theta)) if (isinstance(theta, (int,float)) and float(theta) >= 0) else 0.0
                    savings_pct = float(max(0.0, -delta_cost_pct))
                    cost_increase_pct = float(max(0.0, delta_cost_pct))
                    score = theta_w * float(delta_rdi_pct) + (1.0 - theta_w) * savings_pct - theta_w * cost_increase_pct
                    # Disallow negative scores unless explicitly inspecting all matches
                    if (not show_all_matches) and np.isfinite(score) and (score < 0.0):
                        score = float('nan')
            else:
                delta_rdi_pct = float(improvement) * 100.0 if np.isfinite(improvement) else float('nan')
                delta_cost_pct = float('nan')
                score = float('nan')

            # Penalize additions from specified categories (e.g., mixed_dishes)
            try:
                if penalize_add_categories and any((c in set(penalize_add_categories)) for c in added_cats):
                    if np.isfinite(score):
                        score -= float(gamma_penalize_add)
            except Exception:
                pass

            # Same-category flag for selection heuristic
            same_cat_flag = _is_same_category(added_foods, removed_foods)

            if improvement > 0 or show_all_matches:
                gen_cands.append({
                    'gen_idx': gen_idx,
                    'real_idx': int(real_idx),
                    'n_added': int(n_added),
                    'n_removed': int(n_removed),
                    'n_changed_portions': int(n_changed),
                    'total_food_diff': int(n_added + n_removed + n_changed),
                    'k_substitution': int(max(n_added, n_removed)),
                    'portion_shift_pct': float(portion_shift_pct),
                    'composition_change_pct': float(composition_change_pct_real),
                    'composition_change_pct_real': float(composition_change_pct_real),
                    'composition_change_pct_union': float(composition_change_pct_union),
                    'added_foods': added_foods,
                    'removed_foods': removed_foods,
                    'changed_foods': changed_foods,
                    'gen_deviation': float(gen_deviations[gen_idx]),
                    'real_deviation': float(real_deviations[real_idx]),
                    'improvement': float(improvement),
                    'per_nutrient_changes': per_nutrient_changes,
                    'delta_cost': float(dcost),
                    'delta_mar': float(dmar),
                    'delta_amdr': float(damdr),
                    'delta_mer': float(dmer),
                    'delta_ed': float(ded),
                    'real_cost': (real_cost + float(restaurant_overhead_usd)) if (np.isfinite(dcost)) else np.nan,
                    'gen_cost': (gen_cost + float(restaurant_overhead_usd)) if (np.isfinite(dcost)) else np.nan,
                    # Percent encodings used by policy/reporting
                    'dRDI_pct': float(delta_rdi_pct),
                    'dCost_pct': float(delta_cost_pct),
                    'delta_protein_g': float(d_prot),
                    'delta_fiber_g': float(d_fiber),
                    'delta_sodium_mg': float(d_sodium),
                    'score': float(score),
                    'same_category': same_cat_flag if same_cat_flag is not None else np.nan,
                    'similarity': float(s_val),
                    'selected': False,
                    # sorted: set order changes between processes (salted str hash)
                    'added_cats': sorted(added_cats),
                    'removed_cats': sorted(removed_cats),
                })

        # Winner selection with recording best within vs cross scores
        if gen_cands:
            # Compute best within and best cross scores for diagnostics
            finite = [c for c in gen_cands if np.isfinite(c.get('score', np.nan))]
            within = [c for c in finite if c.get('same_category', None) == 1]
            cross = [c for c in finite if c.get('same_category', None) != 1]
            best_within_score = float(max([c.get('score', -np.inf) for c in within])) if within else float('nan')
            best_cross_score = float(max([c.get('score', -np.inf) for c in cross])) if cross else float('nan')
            for c in gen_cands:
                c['best_within_score'] = best_within_score
                c['best_cross_score'] = best_cross_score
            best = _select_winner(gen_cands, selection_mode)
            if best is not None:
                best['selected'] = True
                substitutions.append(best)

        # Periodic reductions every 50 generated meals (debug only)
        if debug and ((gen_idx + 1) % 50 == 0):
            try:
                sel = [s for s in substitutions if bool(s.get('selected', False))]
                if sel:
                    imp = np.array([s.get('improvement', np.nan) for s in sel], dtype=float)
                    dcost = np.array([s.get('delta_cost', np.nan) for s in sel], dtype=float)
                    pshift = np.array([s.get('portion_shift_pct', np.nan) for s in sel], dtype=float)
                    real_dev = np.array([s.get('real_deviation', np.nan) for s in sel], dtype=float)
                    gen_dev = np.array([s.get('gen_deviation', np.nan) for s in sel], dtype=float)
                    real_costs_dbg = np.array([s.get('real_cost', np.nan) for s in sel], dtype=float)
                    gen_costs_dbg = np.array([s.get('gen_cost', np.nan) for s in sel], dtype=float)
                    win_mask = (dcost <= 0) & (imp > 0)
                    win_rate = float(np.nanmean(win_mask.astype(float))) if len(win_mask) else float('nan')
                    med_rdi = float(np.nanmedian(imp)) if imp.size else float('nan')
                    med_cost = float(np.nanmedian(dcost)) if dcost.size else float('nan')
                    med_ps = float(np.nanmedian(pshift)) if pshift.size else float('nan')
                    # Medians and percent deltas; show orig -> new with Δ and %
                    med_real_dev = float(np.nanmedian(real_dev)) if real_dev.size else float('nan')
                    med_gen_dev = float(np.nanmedian(gen_dev)) if gen_dev.size else float('nan')
                    if np.isfinite(med_real_dev) and med_real_dev != 0:
                        med_rdi_pct = 100.0 * (med_rdi / med_real_dev)
                    else:
                        med_rdi_pct = float('nan')
                    med_real_cost = float(np.nanmedian(real_costs_dbg)) if real_costs_dbg.size else float('nan')
                    med_gen_cost = float(np.nanmedian(gen_costs_dbg)) if gen_costs_dbg.size else float('nan')
                    if np.isfinite(med_real_cost) and med_real_cost != 0:
                        med_cost_pct = 100.0 * med_cost / med_real_cost
                    else:
                        med_cost_pct = float('nan')
                    print(
                        f"  [progress] {meal_type}: processed {gen_idx+1} generated | winners={len(sel)} | "
                        f"win_rate={win_rate:.2f} | RDI dev: {med_real_dev:.3f} → {med_gen_dev:.3f} (Δ{med_rdi:+.3f}, {med_rdi_pct:+.1f}%) | "
                        f"Cost: {med_real_cost:.2f} → {med_gen_cost:.2f} (Δ{med_cost:+.2f}, {med_cost_pct:+.1f}%) | "
                        f"median portion_shift%={med_ps:.1f}"
                    )
            except Exception:
                pass

    print(f"  Found {len(substitutions)} substitutions with positive RDI improvement")
    return substitutions


def create_summary_table(substitutions_by_type: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """Create summary statistics table across meal types."""
    
    summary_data = []
    
    for meal_type, substitutions in substitutions_by_type.items():
        if not substitutions:
            summary_data.append({
                'Meal Type': meal_type,
                'Total Generated Meals': 0,
                'Substitutions Found': 0,
                '1-Food Difference': 0,
                '2-Food Difference': 0,
                'Avg RDI Improvement': 0.0,
                'Avg Energy Improvement': 0.0,
                'Avg Protein Improvement': 0.0,
                'Avg Fiber Improvement': 0.0,
                'Avg Sodium Improvement': 0.0
            })
            continue
        
        # Count food differences
        one_food_diff = sum(1 for s in substitutions if s['total_food_diff'] == 1)
        two_food_diff = sum(1 for s in substitutions if s['total_food_diff'] == 2)
        
        # Calculate average improvements
        avg_rdi_improvement = np.mean([s['improvement'] for s in substitutions])
        
        # Key nutrient improvements
        energy_improvements = []
        protein_improvements = []
        fiber_improvements = []
        sodium_improvements = []
        
        for s in substitutions:
            per_nutrient = s['per_nutrient_changes']
            for nutrient, changes in per_nutrient.items():
                if 'energy' in nutrient.lower():
                    energy_improvements.append(changes['improvement'])
                elif 'protein' in nutrient.lower():
                    protein_improvements.append(changes['improvement'])
                elif 'fiber' in nutrient.lower():
                    fiber_improvements.append(changes['improvement'])
                elif 'sodium' in nutrient.lower():
                    sodium_improvements.append(changes['improvement'])
        
        summary_data.append({
            'Meal Type': meal_type,
            'Total Generated Meals': len(substitutions),  # main() replaces this with the generated-meal count
            'Substitutions Found': len(substitutions),
            '1-Food Difference': one_food_diff,
            '2-Food Difference': two_food_diff,
            'Avg RDI Improvement': avg_rdi_improvement,
            'Avg Energy Improvement': np.mean(energy_improvements) if energy_improvements else 0.0,
            'Avg Protein Improvement': np.mean(protein_improvements) if protein_improvements else 0.0,
            'Avg Fiber Improvement': np.mean(fiber_improvements) if fiber_improvements else 0.0,
            'Avg Sodium Improvement': np.mean(sodium_improvements) if sodium_improvements else 0.0
        })
    
    return pd.DataFrame(summary_data)


def create_detailed_examples_table(substitutions_by_type: Dict[str, List[Dict[str, Any]]], 
                                 code2name: Dict[str, str], W: np.ndarray, 
                                 nutrient_names: List[str], RDI: np.ndarray, 
                                 feature_names: List[str], gen_meals_by_type: Dict[str, np.ndarray],
                                 real_meals_by_type: Dict[str, np.ndarray],
                                 n_examples: int = 10) -> pd.DataFrame:
    """Create detailed examples table with top substitutions."""
    
    examples_data = []
    
    for meal_type, substitutions in substitutions_by_type.items():
        if not substitutions:
            continue
        
        # Sort by improvement and take top N
        top_substitutions = sorted(substitutions, key=lambda x: x['improvement'], reverse=True)[:n_examples]
        
        for i, sub in enumerate(top_substitutions):
            # Get the actual meal data
            gen_meal = gen_meals_by_type[meal_type][sub['gen_idx']]
            real_meal = real_meals_by_type[meal_type][sub['real_idx']]
            
            # Format full meals
            gen_meal_str = []
            real_meal_str = []
            
            for j, food_code in enumerate(feature_names):
                gen_portion = gen_meal[j]
                real_portion = real_meal[j]
                
                if gen_portion > 0:
                    food_name = get_food_name_cached(food_code, code2name)
                    gen_meal_str.append(f"{food_name} ({gen_portion:.1f}g)")
                
                if real_portion > 0:
                    food_name = get_food_name_cached(food_code, code2name)
                    real_meal_str.append(f"{food_name} ({real_portion:.1f}g)")
            
            # Format food changes
            added_foods_str = []
            for food_code in sub['added_foods']:
                food_name = get_food_name_cached(food_code, code2name)
                portion = gen_meal[feature_names.index(food_code)]
                added_foods_str.append(f"{food_name} ({portion:.1f}g)")
            
            removed_foods_str = []
            for food_code in sub['removed_foods']:
                food_name = get_food_name_cached(food_code, code2name)
                portion = real_meal[feature_names.index(food_code)]
                removed_foods_str.append(f"{food_name} ({portion:.1f}g)")
            
            changed_foods_str = []
            for food_code, old_portion, new_portion in sub['changed_foods']:
                food_name = get_food_name_cached(food_code, code2name)
                changed_foods_str.append(f"{food_name}: {old_portion:.1f}g → {new_portion:.1f}g")
            
            # Get key nutrient changes
            per_nutrient = sub['per_nutrient_changes']
            energy_change = 0.0
            protein_change = 0.0
            fiber_change = 0.0
            sodium_change = 0.0
            
            for nutrient, changes in per_nutrient.items():
                if 'energy' in nutrient.lower():
                    energy_change = changes['improvement']
                elif 'protein' in nutrient.lower():
                    protein_change = changes['improvement']
                elif 'fiber' in nutrient.lower():
                    fiber_change = changes['improvement']
                elif 'sodium' in nutrient.lower():
                    sodium_change = changes['improvement']
            
            examples_data.append({
                'Meal Type': meal_type,
                'Rank': i + 1,
                'RDI Improvement': sub['improvement'],
                'Food Differences': sub['total_food_diff'],
                'Generated Meal': '; '.join(gen_meal_str) if gen_meal_str else 'Empty meal',
                'Real Meal': '; '.join(real_meal_str) if real_meal_str else 'Empty meal',
                'Added Foods': '; '.join(added_foods_str) if added_foods_str else 'None',
                'Removed Foods': '; '.join(removed_foods_str) if removed_foods_str else 'None',
                'Changed Portions': '; '.join(changed_foods_str) if changed_foods_str else 'None',
                'RDI Deviation (Real)': sub['real_deviation'],
                'RDI Deviation (Generated)': sub['gen_deviation'],
                'Energy Improvement': energy_change,
                'Protein Improvement': protein_change,
                'Fiber Improvement': fiber_change,
                'Sodium Improvement': sodium_change
            })
    
    return pd.DataFrame(examples_data)


def format_detailed_output_text(examples_df: pd.DataFrame, output_path: Path) -> None:
    """Create human-readable formatted text output."""
    
    with open(output_path, 'w') as f:
        f.write("MEAL SUBSTITUTION ANALYSIS - DETAILED EXAMPLES\n")
        f.write("=" * 60 + "\n\n")
        
        if examples_df.empty:
            f.write("No substitutions found with positive RDI improvements.\n")
            f.write("This could mean:\n")
            f.write("- Generated meals are not sufficiently different from real meals\n")
            f.write("- Generated meals do not show RDI improvements over real meals\n")
            f.write("- The criteria (max food differences, min improvement) are too strict\n\n")
            return
        
        for meal_type in examples_df['Meal Type'].unique():
            meal_examples = examples_df[examples_df['Meal Type'] == meal_type]
            
            f.write(f"{meal_type.upper()} SUBSTITUTIONS\n")
            f.write("-" * 40 + "\n\n")
            
            for _, row in meal_examples.iterrows():
                f.write(f"Rank {row['Rank']}: RDI Improvement = {row['RDI Improvement']:.3f}\n")
                f.write(f"Food Differences: {row['Food Differences']}\n")
                f.write(f"RDI Deviation: {row['RDI Deviation (Real)']:.3f} → {row['RDI Deviation (Generated)']:.3f}\n\n")
                
                f.write("REAL MEAL:\n")
                f.write(f"  {row['Real Meal']}\n\n")
                
                f.write("GENERATED MEAL:\n")
                f.write(f"  {row['Generated Meal']}\n\n")
                
                if row['Added Foods'] != 'None':
                    f.write(f"Added Foods: {row['Added Foods']}\n")
                if row['Removed Foods'] != 'None':
                    f.write(f"Removed Foods: {row['Removed Foods']}\n")
                if row['Changed Portions'] != 'None':
                    f.write(f"Changed Portions: {row['Changed Portions']}\n")
                
                f.write("Key Nutrient Improvements:\n")
                f.write(f"  Energy: {row['Energy Improvement']:+.3f}\n")
                f.write(f"  Protein: {row['Protein Improvement']:+.3f}\n")
                f.write(f"  Fiber: {row['Fiber Improvement']:+.3f}\n")
                f.write(f"  Sodium: {row['Sodium Improvement']:+.3f}\n")
                f.write("\n" + "="*60 + "\n\n")


def main():
    """Main function to run the meal substitution analysis."""
    try:
        from datetime import datetime
        log_path = Path.cwd() / 'logs' / 'substitutions' / 'substitutions.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _logger = _setup_logger('substitutions', str(log_path), file_mode='a')
        sys.stdout = _StreamToLogger(_logger, logging.INFO)
        sys.stderr = _StreamToLogger(_logger, logging.ERROR)
        _logger.info(
            "\n" + "=" * 60 + "\n[session] %s analyze_meal_substitutions %s",
            datetime.now().isoformat(),
            " ".join(sys.argv),
        )
    except Exception:
        pass
    
    parser = argparse.ArgumentParser(description='Analyze meal substitutions for RDI improvements')
    parser.add_argument('--data_dir', type=str, default='outputs/runs/eval_run',
                       help='Run folder (or its data_exports/) with the evaluation exports')
    parser.add_argument('--output_dir', type=str, default='substitution_analysis',
                       help='Output directory for results')
    parser.add_argument('--max_food_diff', type=int, default=2,
                       help='Upper bound on composition swaps; use --require_k_substitution to enforce exact K')
    parser.add_argument('--category_first', action='store_true',
                       help='Enumerate within-cluster, same-category food swaps first (k=1), then compare vs pool')
    parser.add_argument('--category_only', action='store_true',
                       help='Use only within-cluster, same-category food swaps (ignore real-meal pool)')
    parser.add_argument('--category_universe', type=str, default='cluster', choices=['cluster','all'],
                       help='Within-category enumeration universe: cluster or all foods')
    parser.add_argument('--require_k_substitution', type=int, default=None,
                       help='Require exactly this k_substitution (max(#added,#removed)); generated meals without such a candidate get no winner')
    parser.add_argument('--k_nearest', type=int, default=3,
                       help='Number of nearest real meals to match per generated meal')
    parser.add_argument('--kcal_caliper', type=float, default=0.05,
                       help='Relative kcal tolerance (± fraction) for matching')
    parser.add_argument('--items_caliper', type=int, default=1,
                       help='Max difference in item count for matching')
    parser.add_argument('--jaccard_weight', type=float, default=0.7,
                       help='Weight for presence Jaccard vs portion cosine (0..1)')
    parser.add_argument('--portion_change_abs_g', type=float, default=10.0,
                       help='Absolute grams threshold for portion change to count as edit')
    parser.add_argument('--portion_change_rel', type=float, default=0.20,
                       help='Relative threshold (fraction of original grams) for portion change')
    parser.add_argument('--price_map_csv', type=str, default='data/prices/price_map.csv',
                       help='CSV with columns food_code, price_per_100g_usd (others ignored)')
    parser.add_argument('--no_cost_increase', action='store_true',
                       help='Require delta cost ≤ 0 for accepted substitutions')
    parser.add_argument('--budget_cap', type=float, default=None,
                       help='Optional absolute budget cap in USD per meal for generated')
    parser.add_argument('--n_examples', type=int, default=15,
                       help='Number of detailed examples to show per meal type')
    parser.add_argument('--min_improvement', type=float, default=0.0,
                       help='Minimum RDI improvement threshold (raw, unitless)')
    parser.add_argument('--min_delta_rdi', type=float, default=None,
                       help='Minimum ΔRDI%% threshold (percent points), applied post-selection')
    parser.add_argument('--max_meals', type=int, default=None,
                       help='Maximum number of generated meals to process per meal type (for testing)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output to understand why no substitutions are found')
    parser.add_argument('--show_all_matches', action='store_true',
                       help='Show all matches even if they have negative RDI improvements')
    parser.add_argument('--show_cache_stats', action='store_true',
                       help='Show detailed cache statistics')
    parser.add_argument('--cache_nutrients', action='store_true',
                       help='Cache precomputed nutrients to disk for faster subsequent runs')
    parser.add_argument('--bootstraps', type=int, default=1000,
                       help='Number of bootstrap resamples for CIs')
    parser.add_argument('--price_noise_pct', type=float, default=0.0,
                       help='Optional ±pct noise to apply to prices for robustness (e.g., 10.0)')
    # Restaurant pricing
    parser.add_argument('--restaurant_overhead_usd', type=float, default=0.0,
                       help='Flat restaurant overhead added to each meal (cancels in ΔCost)')
    parser.add_argument('--use_restaurant_portion_pricing', action='store_true',
                       help='Use portion-based restaurant pricing layer for costs and deltas')
    parser.add_argument('--exclude_noncal_beverages', action='store_true',
                       help='Set price=0 for near-zero kcal beverages (e.g., water, diet soda, black coffee)')
    parser.add_argument('--noncal_kcal_per100g_threshold', type=float, default=2.0,
                       help='kcal/100g threshold for noncaloric beverages')
    parser.add_argument('--noncal_price_per100g_threshold', type=float, default=0.20,
                       help='price/100g threshold for noncal beverages exclusion')
    parser.add_argument('--theta', type=float, default=0.5,
                       help='Health-vs-savings weight: score = w*ΔRDI%% + (1-w)*savings%% - w*cost increase%%, '
                            'w = θ/(1+θ); 0 scores savings only, large θ favours health')
    parser.add_argument('--tie_band_pct', type=float, default=0.10,
                       help='Tie band as a fraction (e.g., 0.01=1%%) for near-ties on portion shift or score')
    parser.add_argument('--reuse_matches', action='store_true',
                       help='Reuse existing substitution_matches.csv to compute stats/bootstraps only (skip heavy matching).')
    parser.add_argument('--selection_mode', type=str, default='tradeoff', choices=['tradeoff','dominance'],
                       help='Winner selection: tradeoff score or dominance (ΔRDI>0 & ΔCost≤0 max ΔRDI).')
    parser.add_argument('--cross_over_within_pct', type=float, default=20.0,
                       help='Require cross-category score to exceed best within-category score by at least this percent to be chosen.')
    parser.add_argument('--cross_rel_uplift_alpha', type=float, default=None,
                       help='Relative uplift alpha for cross-category acceptance (S_cross >= (1+alpha)*S_within + beta).')
    parser.add_argument('--cross_abs_margin_beta', type=float, default=None,
                       help='Absolute margin beta for cross-category acceptance (S_cross >= (1+alpha)*S_within + beta).')
    parser.add_argument('--force_within_category', action='store_true',
                       help='If any within-category candidates exist for a meal, force winner to be chosen among them.')
    # Category gating and mixed-dishes guardrails
    parser.add_argument('--omit_add_categories', type=str, default='beverages,other',
                       help='Comma-separated main categories disallowed as additions')
    parser.add_argument('--omit_remove_categories', type=str, default='beverages,other',
                       help='Comma-separated main categories disallowed as removals')
    parser.add_argument('--penalize_add_categories', type=str, default='mixed_dishes',
                       help='Comma-separated categories: subtract a penalty when these are added')
    parser.add_argument('--gamma_penalize_add', type=float, default=4.0,
                       help='Penalty applied per penalized category added (tradeoff score units)')
    parser.add_argument('--mixed_cross_alpha', type=float, default=0.25,
                       help='Extra relative uplift needed to pick cross-category when cross adds mixed_dishes')
    parser.add_argument('--mixed_cross_beta', type=float, default=1.5,
                       help='Extra absolute uplift needed to pick cross-category when cross adds mixed_dishes')
    
    args = parser.parse_args()
    k_label = args.require_k_substitution if args.require_k_substitution is not None else f"<={args.max_food_diff}"
    print(f"analyze_meal_substitutions: theta={args.theta:g} k={k_label} -> {args.output_dir} "
          "(log: logs/substitutions/substitutions.log)", file=sys.__stdout__, flush=True)

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Reuse fast path: load existing matches and emit stats/bootstraps
    matches_path = output_dir / 'substitution_matches.csv'
    if args.reuse_matches and matches_path.exists():
        print(f"Reusing existing matches: {matches_path}")
        mdf = pd.read_csv(matches_path)
        # winners filter if present
        if 'selected' in mdf.columns:
            mdf_w = mdf[mdf['selected'] == True]
            if not mdf_w.empty:
                mdf = mdf_w
        # Emit stats and bootstrap from existing matches
        def _emit_stats(_mdf: pd.DataFrame) -> None:
            # Per meal × k
            rows = []
            kcol = 'k_substitution' if 'k_substitution' in _mdf.columns else ('total_food_diff' if 'total_food_diff' in _mdf.columns else None)
            for mt in sorted(_mdf['Meal Type'].dropna().unique()):
                sub = _mdf[_mdf['Meal Type'] == mt]
                if kcol is None or sub.empty:
                    continue
                for k in sorted(sub[kcol].dropna().unique()):
                    sk = sub[sub[kcol] == k]
                    win = float(((sk['delta_cost'] <= 0) & (sk['improvement'] > 0)).mean())
                    share_save = float((sk['delta_cost'] < 0).mean())
                    med_cost = float(np.nanmedian(sk['delta_cost'].to_numpy()))
                    med_rdi = float(np.nanmedian(sk['improvement'].to_numpy()))
                    # Percent changes
                    # RDI% is already a unitless deviation; report medians as percentage-points
                    med_rdi_pct = med_rdi * 100.0
                    if 'real_cost' in sk.columns:
                        base = sk['real_cost'].to_numpy(dtype=float)
                        d = sk['delta_cost'].to_numpy(dtype=float)
                        pct = np.where(np.isfinite(base) & (base != 0), 100.0 * d / base, np.nan)
                        med_cost_pct = float(np.nanmedian(pct))
                    else:
                        med_cost_pct = float('nan')
                    rows.append({'Meal Type': mt, 'k': int(k), 'n': int(len(sk)),
                                 'win_rate': win, 'share_savings': share_save,
                                 'median_dCost': med_cost, 'median_dCost_pct': med_cost_pct,
                                 'median_dRDI': med_rdi, 'median_dRDI_pct': med_rdi_pct})
            if rows:
                pd.DataFrame(rows).to_csv(output_dir / 'substitution_stats_by_k.csv', index=False)
            # Overall by k (across meals)
            rows_k = []
            if kcol is not None:
                for k in sorted(_mdf[kcol].dropna().unique()):
                    sk = _mdf[_mdf[kcol] == k]
                    win = float(((sk.get('delta_cost', np.nan) <= 0) & (sk.get('improvement', np.nan) > 0)).mean())
                    share_save = float((sk.get('delta_cost', np.nan) < 0).mean())
                    med_cost = float(np.nanmedian(sk.get('delta_cost', np.nan)))
                    med_rdi = float(np.nanmedian(sk.get('improvement', np.nan)))
                    rows_k.append({'k': int(k), 'n': int(len(sk)), 'win_rate': win, 'share_savings': share_save,
                                   'median_dCost': med_cost, 'median_dRDI': med_rdi})
            if rows_k:
                pd.DataFrame(rows_k).to_csv(output_dir / 'substitution_stats_by_k_overall.csv', index=False)
            # Portion reallocation medians with 95% CIs
            try:
                import numpy as _np
                def _ci(vals: _np.ndarray):
                    vals = vals.astype(float)
                    rng = _np.random.default_rng(42)
                    meds = []
                    for _ in range(int(args.bootstraps)):
                        idx = rng.integers(0, len(vals), len(vals))
                        meds.append(_np.nanmedian(vals[idx]))
                    meds = _np.array(meds)
                    return float(_np.nanmedian(vals)), float(_np.nanpercentile(meds, 2.5)), float(_np.nanpercentile(meds, 97.5))
                pr_rows = []
                if kcol is not None and 'portion_shift_pct' in _mdf.columns:
                    for mt in sorted(_mdf['Meal Type'].dropna().unique()):
                        sub = _mdf[_mdf['Meal Type']==mt]
                        for k in sorted(sub[kcol].dropna().unique()):
                            sk = sub[sub[kcol]==k]
                            if not sk.empty:
                                est, lo, hi = _ci(sk['portion_shift_pct'].to_numpy())
                                est2, lo2, hi2 = _ci(sk.get('composition_change_pct', pd.Series([np.nan]*len(sk))).to_numpy())
                                pr_rows.append({'Meal Type': mt, 'k': int(k), 'median_portion_shift_pct': est, 'ci_lo': lo, 'ci_hi': hi,
                                                'median_composition_change_pct': est2, 'ci_lo_comp': lo2, 'ci_hi_comp': hi2})
                if pr_rows:
                    pd.DataFrame(pr_rows).to_csv(output_dir / 'portion_reallocation_summary.csv', index=False)
            except Exception:
                pass
            # Overall
            over = _mdf.copy()
            overall = {
                'n_total': int(len(over)),
                'win_rate': float(((over['delta_cost'] <= 0) & (over['improvement'] > 0)).mean()),
                'share_savings': float((over['delta_cost'] < 0).mean()),
                'median_dCost': float(np.nanmedian(over['delta_cost'].to_numpy())),
                'median_dCost_pct': float(np.nanmedian(np.where(
                    np.isfinite(over.get('real_cost', np.nan)) & (over.get('real_cost', np.nan) != 0),
                    100.0 * over.get('delta_cost', np.nan) / over.get('real_cost', np.nan), np.nan
                ))) if 'real_cost' in over.columns else float('nan'),
                'median_dRDI': float(np.nanmedian(over['improvement'].to_numpy())),
                'median_dRDI_pct': float(np.nanmedian(over['improvement'].to_numpy()) * 100.0),
                # Base cost medians (no energy normalization)
                'median_cost_real': float(np.nanmedian(over.get('real_cost', np.nan))) if 'real_cost' in over.columns else float('nan'),
                'median_cost_gen': float(np.nanmedian(over.get('gen_cost', np.nan))) if 'gen_cost' in over.columns else float('nan'),
                'median_cost_delta': float(np.nanmedian(over.get('gen_cost', np.nan) - over.get('real_cost', np.nan))) if {'real_cost','gen_cost'}.issubset(over.columns) else float('nan'),
                'median_cost_delta_pct': float(np.nanmedian(np.where(
                    np.isfinite(over.get('real_cost', np.nan)) & (over.get('real_cost', np.nan) != 0),
                    100.0 * (over.get('gen_cost', np.nan) - over.get('real_cost', np.nan)) / over.get('real_cost', np.nan), np.nan
                ))) if 'real_cost' in over.columns and 'gen_cost' in over.columns else float('nan'),
            }
            pd.DataFrame([overall]).to_csv(output_dir / 'substitution_stats_overall.csv', index=False)
            # Log concise summary
            print("\n=== SUBSTITUTION STATS (reused matches) ===")
            print(f"Overall: n={overall['n_total']} | win_rate={overall['win_rate']:.2f} | median ΔCost={overall['median_dCost']:+.2f} ({overall['median_dCost_pct']:+.1f}%) | median ΔRDI={overall['median_dRDI']:+.3f} ({overall['median_dRDI_pct']:+.1f}pp)")
            if rows:
                by = pd.DataFrame(rows)
                for mt in by['Meal Type'].unique():
                    bt = by[by['Meal Type'] == mt]
                    s = ", ".join([f"k={int(r.k)} n={int(r.n)} win={r.win_rate:.2f} medΔ$={r.median_dCost:+.2f} ({getattr(r,'median_dCost_pct',float('nan')):+.1f}%) medΔRDI={r.median_dRDI:+.3f} ({getattr(r,'median_dRDI_pct',float('nan')):+.1f}pp)" for r in bt.itertuples()])
                    print(f"{mt}: {s}")
            # Overall by k
            if 'substitution_stats_by_k_overall.csv' in [p.name for p in output_dir.iterdir()]:
                byk = pd.read_csv(output_dir / 'substitution_stats_by_k_overall.csv')
            else:
                byk = pd.DataFrame()
            # Pretty tables with tabulate (if available)
            try:
                from tabulate import tabulate as _tab
                if rows:
                    cols = ["Meal Type","k","n","win_rate","share_savings","median_dCost","median_dCost_pct","median_dRDI","median_dRDI_pct"]
                    by_tab = by[cols].copy()
                    by_tab["win_rate"] = by_tab["win_rate"].map(lambda v: f"{v:.2f}")
                    by_tab["share_savings"] = by_tab["share_savings"].map(lambda v: f"{v:.2f}")
                    by_tab["median_dCost_pct"] = by_tab["median_dCost_pct"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "NA")
                    by_tab["median_dRDI_pct"] = by_tab["median_dRDI_pct"].map(lambda v: f"{v:+.1f}pp")
                    by_tab["median_dCost"] = by_tab["median_dCost"].map(lambda v: f"{v:+.2f}")
                    by_tab["median_dRDI"] = by_tab["median_dRDI"].map(lambda v: f"{v:+.3f}")
                    print("\nPer meal × k:")
                    print(_tab(by_tab.values.tolist(), headers=cols, tablefmt="github"))
                ov = pd.DataFrame([overall])[ ["n_total","win_rate","share_savings","median_dCost","median_dCost_pct","median_dRDI","median_dRDI_pct"] ].copy()
                ov["win_rate"] = ov["win_rate"].map(lambda v: f"{v:.2f}")
                ov["share_savings"] = ov["share_savings"].map(lambda v: f"{v:.2f}")
                ov["median_dCost_pct"] = ov["median_dCost_pct"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "NA")
                ov["median_dRDI_pct"] = ov["median_dRDI_pct"].map(lambda v: f"{v:+.1f}pp")
                ov["median_dCost"] = ov["median_dCost"].map(lambda v: f"{v:+.2f}")
                ov["median_dRDI"] = ov["median_dRDI"].map(lambda v: f"{v:+.3f}")
                print("\nOverall:")
                print(_tab(ov.values.tolist(), headers=list(ov.columns), tablefmt="github"))
                if not byk.empty:
                    byk_tab = byk.copy()
                    byk_tab["win_rate"] = byk_tab["win_rate"].map(lambda v: f"{v:.2f}")
                    byk_tab["share_savings"] = byk_tab["share_savings"].map(lambda v: f"{v:.2f}")
                    byk_tab["median_dCost"] = byk_tab["median_dCost"].map(lambda v: f"{v:+.2f}")
                    byk_tab["median_dRDI"] = byk_tab["median_dRDI"].map(lambda v: f"{v:+.3f}")
                    print("\nOverall by k:")
                    print(_tab(byk_tab.values.tolist(), headers=list(byk_tab.columns), tablefmt="github"))
            except Exception:
                pass
        def _print_examples(_mdf: pd.DataFrame) -> None:
            try:
                from tabulate import tabulate as _tab
            except Exception:
                _tab = None
            # winners only if flagged
            sel = _mdf[_mdf['selected'] == True] if 'selected' in _mdf.columns else _mdf
            # prefer k_substitution if present
            if 'k_substitution' in sel.columns:
                kcol = 'k_substitution'
            else:
                kcol = 'total_food_diff' if 'total_food_diff' in sel.columns else None
            if kcol is None:
                return
            print("\nExamples (top ΔRDI per k, up to 3):")
            for k in sorted(sel[kcol].dropna().unique()):
                sk = sel[sel[kcol] == k].sort_values('improvement', ascending=False).head(3)
                rows = []
                for _, r in sk.iterrows():
                    add_s = r.get('added_foods', '')
                    rem_s = r.get('removed_foods', '')
                    add_s = str(add_s) if pd.notna(add_s) else ''
                    rem_s = str(rem_s) if pd.notna(rem_s) else ''
                    rows.append([
                        int(r.get('gen_idx', -1)),
                        int(r.get('real_idx', -1)),
                        f"{r.get('improvement', np.nan):+.3f}",
                        f"{r.get('delta_cost', np.nan):+.2f}",
                        (add_s or '')[:40],
                        (rem_s or '')[:40],
                    ])
                if rows:
                    print(f"k={int(k)}:")
                    if _tab:
                        print(_tab(rows, headers=["gen_idx","real_idx","ΔRDI","Δ$","added","removed"], tablefmt="github"))
                    else:
                        for row in rows:
                            print(row)
        _emit_stats(mdf)
        _print_examples(mdf)
        # Sanity-check price map tails
        try:
            tail = mdf[['Meal Type','delta_cost','real_cost','gen_cost','k_substitution','added_foods','removed_foods']].copy()
            tail = tail.rename(columns={'delta_cost':'dCost','k_substitution':'k'})
            tail = tail.replace([np.inf,-np.inf], np.nan)
            q1 = float(np.nanpercentile(tail['dCost'].to_numpy(), 1))
            q99 = float(np.nanpercentile(tail['dCost'].to_numpy(), 99))
            print(f"\nΔCost sanity: p1={q1:+.2f}, p99={q99:+.2f}")
            # Print top/bottom 3 by ΔCost
            bot = tail.nsmallest(3, 'dCost')
            top = tail.nlargest(3, 'dCost')
            from tabulate import tabulate as _tab
            def _fmt(df):
                df2 = df.copy(); df2['dCost'] = df2['dCost'].map(lambda v: f"{v:+.2f}")
                df2['real_cost'] = df2['real_cost'].map(lambda v: f"{v:.2f}")
                df2['gen_cost'] = df2['gen_cost'].map(lambda v: f"{v:.2f}")
                df2['added_foods'] = df2['added_foods'].fillna('').map(lambda s: str(s)[:30])
                df2['removed_foods'] = df2['removed_foods'].fillna('').map(lambda s: str(s)[:30])
                return _tab(df2.values.tolist(), headers=list(df2.columns), tablefmt='github')
            print("Bottom ΔCost (most savings):\n" + _fmt(bot))
            print("Top ΔCost (most spend):\n" + _fmt(top))
        except Exception:
            pass
        # Bootstrap from matches (reuse logic from below)
        try:
            _np = np
            bs_rows = []
            rng = _np.random.default_rng(42)
            def _ci_of(vals: _np.ndarray):
                vals = vals.astype(float)
                meds = []
                for _ in range(int(args.bootstraps)):
                    idx = rng.integers(0, len(vals), len(vals))
                    meds.append(_np.nanmedian(vals[idx]))
                meds = _np.array(meds)
                return float(_np.nanmedian(vals)), float(_np.nanpercentile(meds, 2.5)), float(_np.nanpercentile(meds, 97.5))
            for mt in mdf['Meal Type'].unique():
                sub = mdf[mdf['Meal Type'] == mt]
                for k in sorted(sub['total_food_diff'].dropna().unique()):
                    sk = sub[sub['total_food_diff'] == k]
                    vals = sk['improvement'].to_numpy()
                    est, lo, hi = _ci_of(vals)
                    bs_rows.append({'meal_type': mt, 'k': int(k), 'metric': 'dRDI', 'est': est, 'ci_lo': lo, 'ci_hi': hi})
                    if 'delta_cost' in sk.columns:
                        vals = sk['delta_cost'].to_numpy()
                        est, lo, hi = _ci_of(vals)
                        bs_rows.append({'meal_type': mt, 'k': int(k), 'metric': 'dCost', 'est': est, 'ci_lo': lo, 'ci_hi': hi})
            if bs_rows:
                pd.DataFrame(bs_rows).to_csv(output_dir / 'bootstrap_results.csv', index=False)
        except Exception:
            pass
        print(f"Results (reused) saved to: {output_dir}")
        return

    # Clear cache at start of each full run
    clear_food_name_cache()
    
    # Load data
    data = load_data(data_dir)
    
    # Preload food name cache for maximum efficiency
    preload_food_name_cache(data['feature_names'], data['code2name'])
    
    # Process each meal type
    meal_types = ['Breakfast', 'Lunch', 'Dinner']
    substitutions_by_type = {}
    
    print(f"\nProcessing {len(meal_types)} meal types...")
    
    for meal_type in tqdm(meal_types, desc="Processing meal types"):
        if meal_type not in data['gen_by_type'] or meal_type not in data['real_by_type']:
            print(f"Warning: No data for {meal_type}")
            substitutions_by_type[meal_type] = []
            continue
        
        gen_meals = data['gen_by_type'][meal_type]
        real_meals = data['real_by_type'][meal_type]
        
        # Limit meals for testing if specified
        if args.max_meals and len(gen_meals) > args.max_meals:
            print(f"  Limiting to {args.max_meals} generated meals for {meal_type}")
            gen_meals = gen_meals[:args.max_meals]
        
        # Optional: load price vector for cost guardrail (with restaurant multipliers)
        price_per_100g = None
        try:
            pm = pd.read_csv(args.price_map_csv)
            if 'food_code' not in pm.columns:
                pm = pd.read_csv(args.price_map_csv, header=None)
                pm.columns = ['food_code','desc','price_per_100g_usd','note','source']
            pm['food_code'] = pm['food_code'].astype(str).str.zfill(8)
            price_col = 'price_per_100g_usd' if 'price_per_100g_usd' in pm.columns else pm.columns[2]
            pm[price_col] = pd.to_numeric(pm[price_col], errors='coerce')
            price_map = dict(zip(pm['food_code'], pm[price_col]))
            base = np.array([price_map.get(c, np.nan) for c in data['feature_names']], dtype=np.float32)
            if np.isnan(base).any():
                med = float(np.nanmedian(base))
                base = np.where(np.isnan(base), med, base).astype(np.float32)
            eff = base.astype(np.float32)
            # Optionally exclude noncaloric beverages by setting price to 0
            if args.exclude_noncal_beverages:
                # Identify energy index
                e_idx = None
                for i, n in enumerate(data['nutrient_names']):
                    if 'energy' in n.lower() and 'kcal' in n.lower():
                        e_idx = i; break
                if e_idx is not None:
                    energy_per100g = data['nutrient_matrix'][:, e_idx].astype(np.float32)
                    noncal_mask = (energy_per100g <= float(args.noncal_kcal_per100g_threshold)) & (eff <= float(args.noncal_price_per100g_threshold))
                    if np.any(noncal_mask):
                        eff = eff.copy(); eff[noncal_mask] = 0.0
            # Built-in food-category multipliers for the per-100 g prices, applied whenever
            # portion pricing is off
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
            # Helper to infer group/sub from name if group map missing
            def _infer_group_and_sub(code8: str, name: str) -> tuple[str, str]:
                g = (data.get('food_to_group', {}).get(code8) or '').strip().lower()
                s = (name or '').lower()
                if not g:
                    if any(w in s for w in ["coffee","tea","water","soda","juice"]):
                        g = "beverages"
                    elif any(w in s for w in ["pizza","sandwich","soup","burrito","taco","enchilada","bowl","plate"]):
                        g = "mixed_dishes"
                    elif any(w in s for w in ["chicken","beef","pork","turkey","fish","shrimp","salmon","egg"]):
                        g = "protein"
                    elif any(w in s for w in ["bread","rice","pasta","cereal","cracker","tortilla","noodle"]):
                        g = "grains"
                    elif any(w in s for w in ["yogurt","cheese","milk"]):
                        g = "dairy"
                    elif any(w in s for w in ["apple","banana","grape","fruit"]):
                        g = "fruits"
                    elif any(w in s for w in ["salad","broccoli","spinach","vegetable","veg"]):
                        g = "vegetables"
                    elif any(w in s for w in ["bar","cookie","chips","snack","dessert","cake","candy"]):
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
                    elif any(w in s for w in ["taco","burrito","enchilada","mexican"]):
                        sub = "mexican"
                return g, sub
            # Apply multipliers per item
            code2name = data.get('code2name', {})
            if not args.use_restaurant_portion_pricing:
                for j, code in enumerate(data['feature_names']):
                    code8 = str(code).zfill(8)
                    name = code2name.get(code8, '')
                    g, sub = _infer_group_and_sub(code8, name)
                    m = mult_map.get(g)
                    if isinstance(m, dict):
                        mult_val = float(m.get(sub, m.get('default', mult_map.get('_default', 2.8))))
                    else:
                        mult_val = float(m if isinstance(m, (int, float)) else mult_map.get('_default', 2.8))
                    eff[j] = eff[j] * float(np.clip(mult_val, 0.8, 6.0))
            price_per_100g = eff.astype(np.float32)
            if args.price_noise_pct and args.price_noise_pct > 0:
                rng = np.random.default_rng(123)
                noise = 1.0 + (args.price_noise_pct/100.0) * (rng.random(len(price_per_100g))*2.0 - 1.0)
                price_per_100g = (price_per_100g * noise).astype(np.float32)
        except Exception as e:
            if args.debug:
                print(f"[warn] price map not loaded: {e}")

        substitutions = find_best_substitutions(
            gen_meals, real_meals, data['nutrient_matrix'], data['rdi_targets'],
            data['nutrient_names'], data['feature_names'], data['code2name'], meal_type, args.max_food_diff, args.debug, args.show_all_matches,
            output_dir / 'nutrient_cache', args.cache_nutrients,
            k_nearest=args.k_nearest, kcal_caliper=args.kcal_caliper, items_caliper=args.items_caliper,
            jaccard_weight=args.jaccard_weight,
            portion_abs_threshold_g=args.portion_change_abs_g,
            portion_rel_threshold=args.portion_change_rel,
            price_per_100g=price_per_100g,
            no_cost_increase=args.no_cost_increase,
            budget_cap=args.budget_cap,
            theta=args.theta,
            selection_mode=args.selection_mode,
            tie_band_pct=args.tie_band_pct,
            restaurant_overhead_usd=args.restaurant_overhead_usd,
            use_restaurant_portion_pricing=args.use_restaurant_portion_pricing,
            food_to_group=data.get('food_to_group', {}),
            x_over_within_pct=args.cross_over_within_pct,
            cross_rel_uplift_alpha=args.cross_rel_uplift_alpha,
            cross_abs_margin_beta=args.cross_abs_margin_beta,
            category_first=args.category_first,
            category_only=args.category_only,
            category_universe=args.category_universe,
            require_k_substitution=args.require_k_substitution,
            force_within_category=args.force_within_category,
            omit_add_categories=set(x.strip().lower() for x in str(args.omit_add_categories).split(',') if x.strip()),
            omit_remove_categories=set(x.strip().lower() for x in str(args.omit_remove_categories).split(',') if x.strip()),
            penalize_add_categories=set(x.strip().lower() for x in str(args.penalize_add_categories).split(',') if x.strip()),
            gamma_penalize_add=float(args.gamma_penalize_add),
            mixed_cross_alpha=args.mixed_cross_alpha,
            mixed_cross_beta=args.mixed_cross_beta
        )
        if args.use_restaurant_portion_pricing:
            print('[info] Using portion-based restaurant pricing; the per-100 g category multipliers are not applied.')
        
        # Filter by minimum improvement (raw)
        if args.min_improvement > 0:
            substitutions = [s for s in substitutions if s['improvement'] >= args.min_improvement]
        # Optional post-filter by ΔRDI%
        if args.min_delta_rdi is not None:
            thr = float(args.min_delta_rdi)
            substitutions = [s for s in substitutions if float(s.get('dRDI_pct', s.get('improvement', 0.0)*100.0)) >= thr]
        
        substitutions_by_type[meal_type] = substitutions
    
    # Save the selected substitutions
    matches_rows = []
    # Helper for strict WWEIA main category lookup (no heuristics)
    ftg_map = data.get('food_to_group', {}) if isinstance(data.get('food_to_group', {}), dict) else {}
    def _cat_of(code: str) -> Optional[str]:
        if not code:
            return None
        c = str(code).zfill(8)
        val = ftg_map.get(c)
        if val is None:
            return None
        # Normalize spacing/case to be robust but do not infer non-WWEIA
        return str(val).strip().lower().replace(" ", "_") or None
    for meal_type, substitutions in substitutions_by_type.items():
        for s in substitutions:
            # Ensure k_substitution is computed consistently: max(#added, #removed)
            if 'k_substitution' in s:
                k_sub_val = int(s.get('k_substitution', 0))
            else:
                add_ct = len(s.get('added_foods', [])) if isinstance(s.get('added_foods', []), list) else 0
                rem_ct = len(s.get('removed_foods', [])) if isinstance(s.get('removed_foods', []), list) else 0
                k_sub_val = int(max(add_ct, rem_ct))
            # Compute strict same-category flag from added/removed foods using WWEIA main categories only
            add_list = s.get('added_foods', []) if isinstance(s.get('added_foods', []), list) else []
            rem_list = s.get('removed_foods', []) if isinstance(s.get('removed_foods', []), list) else []
            same_category_flag = np.nan
            if add_list and rem_list:
                rem_cats = list({ _cat_of(fc) for fc in rem_list if _cat_of(fc) is not None })
                add_cats = list({ _cat_of(fc) for fc in add_list if _cat_of(fc) is not None })
                if len(rem_cats) == 1 and len(add_cats) == 1 and rem_cats[0] == add_cats[0]:
                    same_category_flag = 1
                else:
                    same_category_flag = 0

            matches_rows.append({
                'Meal Type': meal_type,
                'gen_idx': s['gen_idx'],
                'real_idx': s['real_idx'],
                'total_food_diff': s['total_food_diff'],
                'k_substitution': k_sub_val,
                'improvement': s['improvement'],
                'score': s.get('score', np.nan),
                'best_within_score': s.get('best_within_score', np.nan),
                'best_cross_score': s.get('best_cross_score', np.nan),
                'dRDI_pct': s.get('dRDI_pct', float(s.get('improvement', np.nan))*100.0),
                'gen_deviation': s['gen_deviation'],
                'real_deviation': s['real_deviation'],
                'delta_cost': s.get('delta_cost', np.nan),
                'dCost_pct': s.get('dCost_pct', np.nan),
                'real_cost': s.get('real_cost', np.nan),
                'gen_cost': s.get('gen_cost', np.nan),
                'delta_mar': s.get('delta_mar', np.nan),
                'delta_amdr': s.get('delta_amdr', np.nan),
                'delta_mer': s.get('delta_mer', np.nan),
                'delta_ed': s.get('delta_ed', np.nan),
                'delta_protein_g': s.get('delta_protein_g', np.nan),
                'delta_fiber_g': s.get('delta_fiber_g', np.nan),
                'delta_sodium_mg': s.get('delta_sodium_mg', np.nan),
                'portion_shift_pct': s.get('portion_shift_pct', np.nan),
                'composition_change_pct': s.get('composition_change_pct', np.nan),
                'same_category': same_category_flag,
                'selected': bool(s.get('selected', False)),
                'added_foods': ';'.join(s.get('added_foods', [])),
                'removed_foods': ';'.join(s.get('removed_foods', [])),
                'changed_foods': ';'.join([cf[0] for cf in s.get('changed_foods', [])]),
                'added_cats': ';'.join(s.get('added_cats', [])) if isinstance(s.get('added_cats', []), list) else '',
                'removed_cats': ';'.join(s.get('removed_cats', [])) if isinstance(s.get('removed_cats', []), list) else ''
            })
    if matches_rows:
        mdf = pd.DataFrame(matches_rows)
        # Ensure k_substitution exists and is consistent
        if 'k_substitution' not in mdf.columns:
            def _k_sub(row):
                na = len(row['added_foods'].split(';')) if isinstance(row['added_foods'], str) and row['added_foods'] else 0
                nr = len(row['removed_foods'].split(';')) if isinstance(row['removed_foods'], str) and row['removed_foods'] else 0
                return int(max(na, nr))
            mdf['k_substitution'] = mdf.apply(_k_sub, axis=1)
        # Write matches
        mdf.to_csv(output_dir / 'substitution_matches.csv', index=False)

        # Compute data-driven X: 75th percentile of ΔCost% among same-category candidates
        if 'same_category' in mdf.columns and 'dCost_pct' in mdf.columns:
            sc = mdf[(mdf['same_category'] == 1) & np.isfinite(mdf['dCost_pct'])]['dCost_pct'].to_numpy(dtype=float)
            if sc.size > 0:
                X = float(np.nanpercentile(sc, 75))
                meta = {
                    'X_percentile': 75,
                    'X_value_pct': X,
                    'n_same_category': int(sc.size)
                }
                with open(output_dir / 'soft_constraint_X.json', 'w') as _f:
                    json.dump(meta, _f, indent=2)

    # Create summary table
    summary_df = create_summary_table(substitutions_by_type)
    summary_path = output_dir / 'substitution_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary table saved to: {summary_path}")
    # Correct denominator to total generated per meal type
    totals_by_type = {mt: (len(data['gen_by_type'].get(mt, [])) if mt in data['gen_by_type'] else 0) for mt in meal_types}
    if not summary_df.empty:
        for mt in summary_df['Meal Type'].unique():
            summary_df.loc[summary_df['Meal Type']==mt, 'Total Generated Meals'] = int(totals_by_type.get(mt, 0))
        summary_df.to_csv(summary_path, index=False)

    # Stats and bootstrap: save medians and rates with CIs (use k_substitution consistently)
    if matches_rows:
        import numpy as _np
        mdf = pd.DataFrame(matches_rows)
        # Per meal × k
        rows = []
        for mt in sorted(mdf['Meal Type'].dropna().unique()):
            sub = mdf[mdf['Meal Type'] == mt]
            for k in sorted(sub['k_substitution'].dropna().unique()):
                sk = sub[sub['k_substitution'] == k]
                win = float(((sk['delta_cost'] <= 0) & (sk['improvement'] > 0)).mean())
                share_save = float((sk['delta_cost'] < 0).mean())
                med_cost = float(np.nanmedian(sk['delta_cost'].to_numpy()))
                med_rdi = float(np.nanmedian(sk['improvement'].to_numpy()))
                # Percent changes
                if 'real_cost' in sk.columns:
                    base = sk['real_cost'].to_numpy(dtype=float)
                    d = sk['delta_cost'].to_numpy(dtype=float)
                    pct = np.where(np.isfinite(base) & (base != 0), 100.0 * d / base, np.nan)
                    med_cost_pct = float(np.nanmedian(pct))
                else:
                    med_cost_pct = float('nan')
                rows.append({'Meal Type': mt, 'k': int(k), 'n': int(len(sk)),
                             'win_rate': win, 'share_savings': share_save,
                             'median_dCost': med_cost, 'median_dCost_pct': med_cost_pct,
                             'median_dRDI': med_rdi, 'median_dRDI_pct': med_rdi * 100.0})
        if rows:
            pd.DataFrame(rows).to_csv(output_dir / 'substitution_stats_by_k.csv', index=False)
        # Overall by k (across meals)
        rows_k = []
        for k in sorted(mdf['k_substitution'].dropna().unique()):
            sk = mdf[mdf['k_substitution'] == k]
            win = float(((sk.get('delta_cost', np.nan) <= 0) & (sk.get('improvement', np.nan) > 0)).mean())
            share_save = float((sk.get('delta_cost', np.nan) < 0).mean())
            med_cost = float(np.nanmedian(sk.get('delta_cost', np.nan)))
            med_rdi = float(np.nanmedian(sk.get('improvement', np.nan)))
            rows_k.append({'k': int(k), 'n': int(len(sk)), 'win_rate': win, 'share_savings': share_save,
                           'median_dCost': med_cost, 'median_dRDI': med_rdi})
        if rows_k:
            pd.DataFrame(rows_k).to_csv(output_dir / 'substitution_stats_by_k_overall.csv', index=False)
        # Overall summary with base costs
        over = mdf.copy()
        overall = {
            'n_total': int(len(over)),
            'win_rate': float(((over['delta_cost'] <= 0) & (over['improvement'] > 0)).mean()),
            'share_savings': float((over['delta_cost'] < 0).mean()),
            'median_dCost': float(np.nanmedian(over['delta_cost'].to_numpy())),
            'median_dCost_pct': float(np.nanmedian(np.where(
                np.isfinite(over.get('real_cost', np.nan)) & (over.get('real_cost', np.nan) != 0),
                100.0 * over.get('delta_cost', np.nan) / over.get('real_cost', np.nan), np.nan
            ))) if 'real_cost' in over.columns else float('nan'),
            'median_dRDI': float(np.nanmedian(over['improvement'].to_numpy())),
            'median_dRDI_pct': float(np.nanmedian(over['improvement'].to_numpy()) * 100.0),
            'median_cost_real': float(np.nanmedian(over.get('real_cost', np.nan))) if 'real_cost' in over.columns else float('nan'),
            'median_cost_gen': float(np.nanmedian(over.get('gen_cost', np.nan))) if 'gen_cost' in over.columns else float('nan'),
        }
        pd.DataFrame([overall]).to_csv(output_dir / 'substitution_stats_overall.csv', index=False)
        # Bootstrap
        bs_rows = []
        rng = _np.random.default_rng(42)
        def _ci_of(vals: _np.ndarray):
            vals = vals.astype(float)
            meds = []
            for _ in range(int(args.bootstraps)):
                idx = rng.integers(0, len(vals), len(vals))
                meds.append(_np.nanmedian(vals[idx]))
            meds = _np.array(meds)
            return float(_np.nanmedian(vals)), float(_np.nanpercentile(meds, 2.5)), float(_np.nanpercentile(meds, 97.5))
        def _rate_ci(mask: _np.ndarray):
            mask = mask.astype(float)
            rates = []
            for _ in range(int(args.bootstraps)):
                idx = rng.integers(0, len(mask), len(mask))
                rates.append(_np.nanmean(mask[idx]))
            return float(_np.nanmean(mask)), float(_np.nanpercentile(rates, 2.5)), float(_np.nanpercentile(rates, 97.5))
        taus = [0.05, 0.10, 0.15]
        for mt in mdf['Meal Type'].unique():
            sub = mdf[mdf['Meal Type']==mt]
            for k in sorted(sub['k_substitution'].dropna().unique()):
                sk = sub[sub['k_substitution']==k]
                win_mask = (sk['delta_cost']<=0) & (sk['improvement']>0)
                est, lo, hi = _rate_ci(win_mask.to_numpy())
                bs_rows.append({'meal_type': mt, 'k': int(k), 'metric': 'win_rate', 'est': est, 'ci_lo': lo, 'ci_hi': hi})
                for tau in taus:
                    succ = (sk['delta_cost']<=0) & (sk['improvement']>=tau)
                    est, lo, hi = _rate_ci(succ.to_numpy())
                    bs_rows.append({'meal_type': mt, 'k': int(k), 'metric': f'success_rate_tau_{tau:.2f}', 'est': est, 'ci_lo': lo, 'ci_hi': hi})
                for mcol, mname in [('delta_mar','dMAR'),('delta_amdr','dAMDR'),('delta_mer','dMER'),('delta_ed','dED')]:
                    if mcol in sk.columns:
                        vals = sk[mcol].to_numpy()
                        est, lo, hi = _ci_of(vals)
                        bs_rows.append({'meal_type': mt, 'k': int(k), 'metric': mname, 'est': est, 'ci_lo': lo, 'ci_hi': hi})
                vals = sk['improvement'].to_numpy()
                est, lo, hi = _ci_of(vals)
                bs_rows.append({'meal_type': mt, 'k': int(k), 'metric': 'dRDI', 'est': est, 'ci_lo': lo, 'ci_hi': hi})
                if 'delta_cost' in sk.columns:
                    vals = sk['delta_cost'].to_numpy()
                    est, lo, hi = _ci_of(vals)
                    bs_rows.append({'meal_type': mt, 'k': int(k), 'metric': 'dCost', 'est': est, 'ci_lo': lo, 'ci_hi': hi})
        pd.DataFrame(bs_rows).to_csv(output_dir / 'bootstrap_results.csv', index=False)
    
    # Create detailed examples table
    examples_df = create_detailed_examples_table(
        substitutions_by_type, data['code2name'], data['nutrient_matrix'],
        data['nutrient_names'], data['rdi_targets'], data['feature_names'],
        data['gen_by_type'], data['real_by_type'], args.n_examples
    )
    examples_path = output_dir / 'substitution_examples_detailed.csv'
    examples_df.to_csv(examples_path, index=False)
    print(f"Detailed examples saved to: {examples_path}")
    
    # Create formatted text output
    text_path = output_dir / 'substitution_examples_detailed.txt'
    format_detailed_output_text(examples_df, text_path)
    print(f"Formatted text output saved to: {text_path}")
    
    # Print key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    total_substitutions = sum(len(subs) for subs in substitutions_by_type.values())
    print(f"Total substitutions found: {total_substitutions}")
    
    for meal_type, substitutions in substitutions_by_type.items():
        if substitutions:
            avg_improvement = np.mean([s['improvement'] for s in substitutions])
            print(f"{meal_type}: {len(substitutions)} substitutions, avg RDI improvement: {avg_improvement:.3f}")
        else:
            print(f"{meal_type}: No substitutions found")
    
    # Print cache statistics
    if args.show_cache_stats:
        cache_stats = get_cache_stats()
        print("\nCache Statistics:")
        print(f"  Total cached food names: {cache_stats['cache_size']}")
        if cache_stats['cached_items']:
            print(f"  Sample cached items: {', '.join(cache_stats['cached_items'])}")
        
        # Show cache hit rate estimation
        total_food_lookups = sum(len(meals) * len(data['feature_names']) for meals in data['gen_by_type'].values())
        cache_hit_rate = (cache_stats['cache_size'] / len(data['feature_names'])) * 100
        print(f"  Estimated cache hit rate: {cache_hit_rate:.1f}%")
        print(f"  Total potential lookups: {total_food_lookups}")
        print(f"  Unique food codes: {len(data['feature_names'])}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
