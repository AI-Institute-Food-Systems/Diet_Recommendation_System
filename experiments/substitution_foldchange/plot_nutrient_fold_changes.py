#!/usr/bin/env python3
"""
Figure 4A: a worked 1-hop / 2-hop substitution example (`make figure4a`).

With --sweep_dir, reads the k=1 and k=2 substitution_matches.csv at each k's knee θ
(figures_theta_sweep/optimal_theta_by_k.json); --matches_csv (repeatable) names the files
directly. select_k1_k2_same_base() looks for real meals with 4-5 foods that have both a k=1
and a k=2 selected winner that is healthier and not more expensive, and draws the
second-highest-scoring candidate (the best one when there is only one); --base
MealType:real_idx restricts the search to one real meal, and --pair MealType:gen_idx,real_idx
(once for k=1, once for k=2) skips it. Displayed costs use experiments/costs/restaurant_pricing.py
plus a $2.00 overhead; ΔRDI is the relative reduction of the real meal's RDI deviation.

Writes to --outdir:
  fixed_meal_composition.{svg,png}     the three meals with cost and ΔRDI (Figure 4A); the side
                                       salad, fruit cup and iced tea listed in every column are
                                       illustrative only (not part of the meals, costs or ΔRDI)
  figure_substitution_foldchange_two_panels.{svg,png}   generated / real nutrient ratios
  figure_meal_compositions.{svg,png}   the same meals with grams
  candidate_meals_top50.{csv,json}     the other candidate bases

    python -m experiments.substitution_foldchange.plot_nutrient_fold_changes \
        --sweep_dir outputs/runs/eval_run/substitutions_k3 \
        --data_dir outputs/runs/eval_run \
        --outdir outputs/runs/eval_run/substitutions_k3/figure4a
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
try:
    from experiments.costs.restaurant_pricing import (
        clean_name_display as rp_clean_name_display,
        portion_spec_for_label as rp_portion_spec_for_label,
        max_portions_for_label as rp_max_portions_for_label,
        estimate_restaurant_cost as rp_estimate_restaurant_cost,
    )
except Exception:
    rp_clean_name_display = None  # type: ignore
    rp_portion_spec_for_label = None  # type: ignore
    rp_max_portions_for_label = None  # type: ignore
    rp_estimate_restaurant_cost = None  # type: ignore

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.transforms import blended_transform_factory


# Canonical nutrient display names and groupings
NUTRIENT_LABELS: Dict[str, str] = {
    'energy_kcal': 'Energy (kcal)',
    'protein_g': 'Protein (g)',
    'fat_g': 'Total Fat (g)',
    'carb_g': 'Carbohydrate (g)',
    'fiber_g': 'Dietary Fiber (g)',
    'sugars_total_g': 'Total Sugars (g)',
    'sfa_g': 'Saturated Fatty Acids (g)',
    'mono_fat_g': 'Monounsaturated Fatty Acids (g)',
    'poly_fat_g': 'Polyunsaturated Fatty Acids (g)',
    'cholesterol_mg': 'Cholesterol (mg)',
    'sodium_mg': 'Sodium (mg)',
    'potassium_mg': 'Potassium (mg)',
    'calcium_mg': 'Calcium (mg)',
    'iron_mg': 'Iron (mg)',
    'magnesium_mg': 'Magnesium (mg)',
    'phosphorus_mg': 'Phosphorus (mg)',
    'zinc_mg': 'Zinc (mg)',
    'copper_mg': 'Copper (mg)',
    'selenium_mcg': 'Selenium (mcg)',
    'vitamin_a_mcg': 'Vitamin A (mcg)',
    'vitamin_e_mg': 'Vitamin E (mg)',
    'vitamin_d_iu': 'Vitamin D (IU)',
    'vitamin_c_mg': 'Vitamin C (mg)',
    'thiamin_mg': 'Thiamin (mg)',
    'riboflavin_mg': 'Riboflavin (mg)',
    'niacin_mg': 'Niacin (mg)',
    'vitamin_b6_mg': 'Vitamin B6 (mg)',
    'vitamin_b12_mcg': 'Vitamin B12 (mcg)',
    'choline_mg': 'Choline (mg)',
    'vitamin_k_mcg': 'Vitamin K (mcg)',
    'folate_mcg': 'Folate (mcg)',
}

# Group order for display: (smaller = earlier)
NUTRIENT_GROUP_INDEX: Dict[str, int] = {}
_groups = {
    'Energy': ['energy_kcal'],
    'Macronutrients: Protein': ['protein_g'],
    'Macronutrients: Fat': ['fat_g', 'sfa_g', 'mono_fat_g', 'poly_fat_g', 'cholesterol_mg'],
    'Macronutrients: Carbohydrate': ['carb_g', 'sugars_total_g', 'fiber_g'],
    'Minerals / Electrolytes': ['sodium_mg', 'potassium_mg', 'calcium_mg', 'iron_mg', 'magnesium_mg', 'phosphorus_mg', 'zinc_mg', 'copper_mg', 'selenium_mcg'],
    'Vitamins': ['vitamin_a_mcg', 'vitamin_c_mg', 'vitamin_d_iu', 'vitamin_e_mg', 'vitamin_k_mcg', 'thiamin_mg', 'riboflavin_mg', 'niacin_mg', 'vitamin_b6_mg', 'vitamin_b12_mcg', 'folate_mcg', 'choline_mg'],
}
for gi, (_, keys) in enumerate(_groups.items()):
    for k in keys:
        NUTRIENT_GROUP_INDEX[k] = gi

# Heuristic healthy / limit nutrients for ranking candidate examples. Tuples rather than sets:
# the candidate score sums over them, and a set's order (hence the rounding) changes with
# Python's hash seed from run to run.
HEALTHY_NUTRIENTS = (
    'fiber_g', 'vitamin_a_mcg', 'vitamin_c_mg', 'vitamin_d_iu', 'vitamin_e_mg',
    'vitamin_k_mcg', 'thiamin_mg', 'riboflavin_mg', 'niacin_mg', 'vitamin_b6_mg',
    'vitamin_b12_mcg', 'folate_mcg', 'choline_mg', 'potassium_mg', 'calcium_mg',
    'iron_mg', 'magnesium_mg', 'phosphorus_mg', 'zinc_mg', 'copper_mg', 'selenium_mcg',
)
BAD_NUTRIENTS = ('sugars_total_g', 'fat_g', 'sfa_g', 'cholesterol_mg')


def load_data(data_dir: Path) -> Dict:
    data = {}
    # Allow passing either the parent dir or the data_exports dir directly
    base = data_dir / 'data_exports' if (data_dir / 'data_exports').exists() else data_dir
    data['gen_by_type'] = pd.read_pickle(base / 'generated_meals_by_type.pkl')
    data['real_by_type'] = pd.read_pickle(base / 'real_meals_by_type.pkl')
    data['W'] = np.load(base / 'nutrient_matrix.npy')
    data['nutrient_names'] = json.load(open(base / 'nutrient_names.json', 'r'))
    data['feature_names'] = json.load(open(base / 'feature_names.json', 'r'))
    # RDI daily targets (vector aligned with nutrient_names)
    rdi_fp = base / 'RDI.npy'
    if rdi_fp.exists():
        try:
            data['RDI'] = np.load(rdi_fp)
        except Exception:
            data['RDI'] = None
    else:
        data['RDI'] = None
    # food names are optional but useful for titles
    code_map_fp = base / 'food_code_to_name.json'
    data['code2name'] = json.load(open(code_map_fp, 'r')) if code_map_fp.exists() else {}
    return data


def _align_rdi_units_with_names(rdi: Optional[np.ndarray], nutrient_names: List[str]) -> Optional[np.ndarray]:
    """Ensure RDI units match the nutrient vector units.

    Handles Vitamin D IU↔mcg mismatches by heuristics:
      - If names include 'vitamin_d_iu' but RDI looks like mcg (<200), convert to IU by ×40.
      - If names include 'vitamin_d_mcg' but RDI looks like IU (>200), convert to mcg by ÷40.
    Leaves other nutrients unchanged.
    """
    if rdi is None:
        return None
    out = np.array(rdi, dtype=float).copy()
    name_to_idx = {n: i for i, n in enumerate(nutrient_names)}
    idx_iu = name_to_idx.get('vitamin_d_iu', None)
    idx_mcg = name_to_idx.get('vitamin_d_mcg', None)
    try:
        if idx_iu is not None and np.isfinite(out[idx_iu]):
            # Typical RDI in IU is ~800; if value <200, likely mcg (e.g., 20)
            if out[idx_iu] < 200.0:
                out[idx_iu] = out[idx_iu] * 40.0
        if idx_mcg is not None and np.isfinite(out[idx_mcg]):
            # Typical RDI in mcg is ~20; if value >200, likely IU (e.g., 800)
            if out[idx_mcg] > 200.0:
                out[idx_mcg] = out[idx_mcg] / 40.0
    except Exception:
        pass
    return out
def compute_delta_rdi_percent_from_vectors(real_meal: np.ndarray, gen_meal: np.ndarray,
                                           W: np.ndarray, RDI: Optional[np.ndarray]) -> float:
    if RDI is None:
        return float('nan')
    try:
        rN = (real_meal / 100.0) @ W
        gN = (gen_meal / 100.0) @ W
        with np.errstate(divide='ignore', invalid='ignore'):
            r_ratio = np.where(RDI > 0, rN / RDI, np.nan)
            g_ratio = np.where(RDI > 0, gN / RDI, np.nan)
        r_dev = np.nanmean(np.abs(r_ratio - 1.0))
        g_dev = np.nanmean(np.abs(g_ratio - 1.0))
        if np.isfinite(r_dev) and r_dev > 0:
            return (r_dev - g_dev) / r_dev * 100.0
        return float('nan')
    except Exception:
        return float('nan')




def _parse_food_list(val: Optional[str]) -> List[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    s = str(val).strip()
    if not s:
        return []
    # Normalize codes that may appear as numeric strings with decimal (e.g., '51154100.0')
    out: List[str] = []
    for x in s.split(';'):
        x = x.strip()
        if not x:
            continue
        # If looks like a float number string, cast to int then back to str
        try:
            if x.replace('.', '', 1).isdigit():
                xv = float(x)
                if xv.is_integer():
                    x = str(int(xv))
        except Exception:
            pass
        out.append(x)
    return out


def _read_matches_any(matches_csv: Union[Path, List[Path]]) -> pd.DataFrame:
    """Read one or more matches CSV files and concatenate."""
    try:
        if isinstance(matches_csv, (list, tuple)):
            frames = []
            for p in matches_csv:
                frames.append(pd.read_csv(Path(p)))
            return pd.concat(frames, ignore_index=True)
        else:
            return pd.read_csv(Path(matches_csv))
    except Exception as e:
        raise e


def select_pairs(matches_csv: Union[Path, List[Path]], n_pairs: int) -> List[Tuple[str, int, int, Dict]]:
    """Fallback when select_k1_k2_same_base() finds nothing: the top selected winners with
    ΔRDI > 0, ΔCost <= 0 and k <= 2 (one per generated meal), ranked by ΔRDI then ΔCost.
    Uses the costs in the matches files as they are."""
    df = _read_matches_any(matches_csv)
    # winners with savings, small edits
    if 'selected' in df.columns:
        df = df[df['selected'] == True]
    df = df[(df.get('improvement', 0) > 0) & (df.get('delta_cost', 0) <= 0)]
    if 'k_substitution' in df.columns:
        df = df[df['k_substitution'] <= 2]
    elif 'total_food_diff' in df.columns:
        df = df[df['total_food_diff'] <= 2]
    # rank
    df = df.sort_values(['improvement', 'delta_cost'], ascending=[False, True])
    pairs = []
    seen = set()
    for _, r in df.iterrows():
        key = (r['Meal Type'], int(r['gen_idx']))
        if key in seen:
            continue
        pairs.append((r['Meal Type'], int(r['gen_idx']), int(r['real_idx']), r.to_dict()))
        seen.add(key)
        if len(pairs) >= n_pairs:
            break
    return pairs


def _present_food_codes(meal_vec: np.ndarray, feature_names: List[str]) -> List[str]:
    idxs = np.where(meal_vec > 0)[0]
    return [feature_names[i] for i in idxs]


def select_k1_k2_same_base(
    matches_csv: Union[Path, List[Path]],
    data: Dict,
    require_same_base: bool = True,
    base_spec: Optional[str] = None,
    print_top_k: int = 5,
    select_rank: int = 1,
) -> Tuple[Optional[List[Tuple[str, int, int, Dict]]], List[Dict]]:
    """Select two substitutions sharing the same base real meal: one k=1 and one k=2.

    Costs are re-priced with restaurant pricing, then candidates are selected winners with
    ΔRDI > 0 and ΔCost <= 0 on a base real meal of 4-5 foods, with different swaps for k=1
    and k=2 and no pizza added. Each (k=1, k=2) combination of the top five per k is scored
    by summed ΔRDI, minus 0.01 x summed ΔCost, plus a small nutrient-direction bonus.

    Returns (pair, candidates): pair is [(mt, gen_idx_k1, real_idx, meta_k1),
    (mt, gen_idx_k2, real_idx, meta_k2)] for the best combination, or for the select_rank-th
    one when that many exist, or None if nothing qualifies.
    """
    df = _read_matches_any(matches_csv)
    # Override costs using restaurant pricing before filtering
    try:
        df = _apply_restaurant_pricing(df, data)
    except Exception:
        pass
    if 'selected' in df.columns:
        df = df[df['selected'] == True]
    df = df[(df.get('improvement', 0) > 0) & (df.get('delta_cost', 0) <= 0)]
    if df.empty:
        return None, []
    # Normalize added/removed lists and enforce exact k
    df = df.copy()
    df['added_list'] = df['added_foods'].apply(_parse_food_list) if 'added_foods' in df.columns else [[]]*len(df)
    df['removed_list'] = df['removed_foods'].apply(_parse_food_list) if 'removed_foods' in df.columns else [[]]*len(df)
    if 'k_substitution' in df.columns:
        df['k_substitution'] = pd.to_numeric(df['k_substitution'], errors='coerce')
    else:
        df['k_substitution'] = df.apply(lambda r: max(len(r['added_list']), len(r['removed_list'])), axis=1)

    # Optional force base
    if base_spec:
        try:
            mt, ridx = base_spec.split(':', 1)
            real_idx_force = int(ridx)
            df = df[(df['Meal Type'] == mt) & (df['real_idx'] == real_idx_force)]
        except Exception:
            pass
        if df.empty:
            return None, []

    # Group by base
    grouped = df.groupby(['Meal Type', 'real_idx'], as_index=False)
    best: Optional[List[Tuple[str, int, int, Dict]]] = None
    best_score = -np.inf
    candidates: List[Dict] = []
    for (mt, real_idx), g in grouped:
        # try to find valid k=1 and k=2
        g1 = g[(g['k_substitution'] == 1) & (g['added_list'].map(len) == 1) & (g['removed_list'].map(len) == 1)]
        g2 = g[(g['k_substitution'] == 2) & (g['added_list'].map(len) == 2) & (g['removed_list'].map(len) == 2)]
        if g1.empty or g2.empty:
            continue
        # Base composition
        real_meal = data['real_by_type'][mt][int(real_idx)]
        base_codes = set(_present_food_codes(real_meal, data['feature_names']))
        # Enforce base meal realism: 4-5 foods exactly
        if not (4 <= len(base_codes) <= 5):
            continue

        def _is_valid_comp(row: pd.Series) -> bool:
            if not require_same_base:
                return True
            gi = int(row['gen_idx'])
            gen_meal = data['gen_by_type'][mt][gi]
            gen_codes = set(_present_food_codes(gen_meal, data['feature_names']))
            removed = set(row['removed_list'])
            added = set(row['added_list'])
            # unchanged should equal base minus removed
            unchanged_expected = base_codes - removed
            unchanged_actual = base_codes & gen_codes
            return (unchanged_actual == unchanged_expected) and len(added) == len(removed)

        g1v = g1[g1.apply(_is_valid_comp, axis=1)]
        g2v = g2[g2.apply(_is_valid_comp, axis=1)]
        if g1v.empty or g2v.empty:
            continue
        # enforce minimum foods (>=4) for both generated meals
        def _has_min_foods(row: pd.Series, min_foods: int = 4) -> bool:
            gi = int(row['gen_idx'])
            gen_meal = data['gen_by_type'][mt][gi]
            return int(np.sum(gen_meal > 0)) >= min_foods
        g1 = g1[g1.apply(_has_min_foods, axis=1)]
        g2 = g2[g2.apply(_has_min_foods, axis=1)]
        if g1.empty or g2.empty:
            continue
        # choose best by improvement then cost then portion shift (if columns exist)
        def _sorted(_g: pd.DataFrame) -> pd.DataFrame:
            by_cols = ['improvement']
            asc = [False]
            if 'delta_cost' in _g.columns:
                by_cols.append('delta_cost'); asc.append(True)
            if 'portion_shift_pct' in _g.columns:
                by_cols.append('portion_shift_pct'); asc.append(True)
            return _g.sort_values(by=by_cols, ascending=asc)
        g1s = _sorted(g1v)
        g2s = _sorted(g2v)
        # Try top few combinations and enforce different foods across k
        top1 = list(g1s.head(5).itertuples(index=False))
        top2 = list(g2s.head(5).itertuples(index=False))
        for r1t in top1:
            r1 = pd.Series(r1t._asdict()) if hasattr(r1t, '_asdict') else pd.Series(r1t._asdict())
            set_add1 = set(r1.get('added_list', []))
            set_rem1 = set(r1.get('removed_list', []))
            for r2t in top2:
                r2 = pd.Series(r2t._asdict()) if hasattr(r2t, '_asdict') else pd.Series(r2t._asdict())
                set_add2 = set(r2.get('added_list', []))
                set_rem2 = set(r2.get('removed_list', []))
                # Require different foods (added or removed sets differ)
                if (set_add1 == set_add2) and (set_rem1 == set_rem2):
                    continue
                # Exclude candidates where pizza is being added (pizza allowed in original, not as a substitute)
                def _has_pizza(added_codes: set) -> bool:
                    c2n = data.get('code2name', {})
                    for code in added_codes:
                        name = str(c2n.get(code, code)).lower()
                        if 'pizza' in name:
                            return True
                    return False
                if _has_pizza(set_add1) or _has_pizza(set_add2):
                    continue
                # Health heuristic: net increase in healthy nutrients and net decrease in bad nutrients
                try:
                    def _score_health(row: pd.Series) -> float:
                        mt_local = mt
                        gi = int(row['gen_idx']); ri = int(real_idx)
                        gen = data['gen_by_type'][mt_local][gi]
                        real = data['real_by_type'][mt_local][ri]
                        gN = (gen / 100.0) @ data['W']
                        rN = (real / 100.0) @ data['W']
                        delta = gN - rN
                        # map nutrient deltas
                        dmap = {data['nutrient_names'][i]: delta[i] for i in range(len(data['nutrient_names']))}
                        inc_healthy = float(np.nansum([max(0.0, dmap.get(k, 0.0)) for k in HEALTHY_NUTRIENTS]))
                        dec_bad = float(np.nansum([max(0.0, -dmap.get(k, 0.0)) for k in BAD_NUTRIENTS]))
                        return inc_healthy + dec_bad
                    hscore = _score_health(r1) + _score_health(r2)
                except Exception:
                    hscore = 0.0
                score = float(r1.get('improvement', 0.0)) + float(r2.get('improvement', 0.0))
                score -= 0.01 * (float(r1.get('delta_cost', 0.0)) + float(r2.get('delta_cost', 0.0)))
                score += 0.001 * hscore
                cand = {
                    'Meal Type': mt,
                    'real_idx': int(real_idx),
                    'score': score,
                    'r1': r1.to_dict(),
                    'r2': r2.to_dict(),
                }
                candidates.append(cand)
                if score > best_score:
                    best_score = score
                    best = [
                        (mt, int(r1['gen_idx']), int(real_idx), r1.to_dict()),
                        (mt, int(r2['gen_idx']), int(real_idx), r2.to_dict()),
                    ]
    # Print top-k candidates if requested and not forcing base
    if candidates and (not base_spec) and int(print_top_k) > 0:
        try:
            code2name = data.get('code2name', {})
            def _nm(code: str) -> str:
                import re as _re
                nm = code2name.get(code, code)
                m = _re.findall(r"\(([^)]+)\)", nm)
                return (m[-1] if m else nm)
            cand_sorted = sorted(candidates, key=lambda c: c['score'], reverse=True)
            print("\nTop candidate bases (k=1 & k=2):")
            for i, c in enumerate(cand_sorted[:int(print_top_k)]):
                mt = c['Meal Type']; ridx = c['real_idx']
                a1 = '; '.join([_nm(x) for x in c['r1'].get('added_list', [])])
                a2 = '; '.join([_nm(x) for x in c['r2'].get('added_list', [])])
                imp = float(c['r1'].get('improvement', 0.0) + c['r2'].get('improvement', 0.0))
                dcost = float(c['r1'].get('delta_cost', 0.0) + c['r2'].get('delta_cost', 0.0))
                print(f"{i+1}. {mt} base real_idx={ridx} | ΔRDIpp={imp*100:+.1f} | Δ$={dcost:+.2f} | add k1=[{a1}] k2=[{a2}]")
        except Exception:
            pass
    # Select a specific rank if requested
    if candidates and int(select_rank) > 1:
        cand_sorted = sorted(candidates, key=lambda c: c['score'], reverse=True)
        idx = int(select_rank) - 1
        if 0 <= idx < len(cand_sorted):
            c = cand_sorted[idx]
            mt = c['Meal Type']; ridx = int(c['real_idx'])
            r1 = c['r1']; r2 = c['r2']
            best = [
                (mt, int(r1['gen_idx']), ridx, r1),
                (mt, int(r2['gen_idx']), ridx, r2),
            ]
    return best, candidates


def _clean_name_display(code: str, code2name: Dict[str, str]) -> str:
    if rp_clean_name_display is not None:
        return rp_clean_name_display(code, code2name)
    import re as _re
    nm = code2name.get(code, code)
    m = _re.findall(r"\(([^)]+)\)", nm)
    return (m[-1] if m else nm)


def _meal_to_str_list(mt: str, meal_vec: np.ndarray, data: Dict) -> List[str]:
    items = []
    for j, code in enumerate(data['feature_names']):
        grams = float(meal_vec[j])
        if grams > 0:
            nm = _clean_name_display(code, data['code2name'])
            items.append(f"{nm} ({grams:.0f}g)")
    return items


def _meal_to_items(meal_vec: np.ndarray, data: Dict) -> List[Tuple[str, float, bool]]:
    items = []
    for j, code in enumerate(data['feature_names']):
        grams = float(meal_vec[j])
        if grams > 0:
            nm = _clean_name_display(code, data['code2name'])
            items.append((nm, grams, False))
    return items


def compute_pair_meta(data: Dict, mt: str, gen_idx: int, real_idx: int) -> Dict:
    """Compute k_substitution and added/removed foods for a (gen, real) pair.
    Uses simple presence (>0 g) without portion thresholds.
    """
    feature_names = data['feature_names']
    gen = data['gen_by_type'][mt][gen_idx]
    real = data['real_by_type'][mt][real_idx]
    gen_present = gen > 0
    real_present = real > 0
    added_codes = [feature_names[i] for i in np.where(gen_present & (~real_present))[0]]
    removed_codes = [feature_names[i] for i in np.where((~gen_present) & real_present)[0]]
    k_sub = int(max(len(added_codes), len(removed_codes)))
    return {
        'k_substitution': k_sub,
        'added_foods': ';'.join(added_codes),
        'removed_foods': ';'.join(removed_codes),
    }


def _friendly_portion(name: str, grams: float) -> str:
    n = name.lower().strip()
    # Base display name simplifications (keep meaning)
    def base_label() -> str:
        if 'potato' in n and 'mashed' in n:
            return 'Mashed potatoes'
        if 'salisbury steak' in n:
            return 'Salisbury steak with gravy'
        if ('meat loaf' in n) or ('meatloaf' in n):
            return 'Beef/pork meatloaf with tomato sauce'
        if 'chicken breast' in n:
            return 'Baked skinless chicken breast'
        if 'peas and carrots' in n:
            return 'Steamed peas & carrots'
        if 'cabbage' in n and 'red' in n:
            return 'Shredded red cabbage'
        if 'hot dog' in n and 'bun' in n:
            return 'Beef hot dog on bun'
        if 'ketchup' in n:
            return 'Ketchup'
        if 'mayonnaise' in n or ('salad dressing' in n and 'mayonnaise' in n):
            return 'Mayonnaise'
        if 'bread' in n and 'whole wheat' in n:
            return 'Whole wheat bread'
        if 'fruit juice' in n:
            return 'Fruit juice'
        if ('yogurt' in n and 'liquid' in n) or ('drinkable' in n and 'yogurt' in n):
            return 'Drinkable yogurt'
        if ('yogurt' in n) and ('lowfat' in n or 'plain' in n):
            return 'Low-fat yogurt'
        if 'rice' in n and 'cooked' in n:
            return 'Steamed white rice'
        if ('bean' in n) and ('dried' in n or 'no added fat' in n or 'cooked' in n):
            return 'Cooked beans'
        if 'cherries' in n:
            return 'Cherries'
        return name.strip()

    label = base_label()
    g = float(grams) if grams is not None else 0.0

    # Heuristic grams-to-portion conversions per food type
    def cups_from(gm: float, grams_per_cup: float) -> str:
        if grams_per_cup <= 0 or gm <= 0:
            return ''
        cups = gm / grams_per_cup
        # Round to nearest 1/4 cup for readability
        cups_q = max(0.25, round(cups * 4) / 4)
        if abs(cups_q - 1.0) < 1e-6:
            return '~1 cup'
        return f'~{cups_q:g} cups'

    def tbsp_from(gm: float, grams_per_tbsp: float) -> str:
        if grams_per_tbsp <= 0 or gm <= 0:
            return ''
        spoons = gm / grams_per_tbsp
        spoons_r = max(1.0, round(spoons))
        unit = 'tbsp' if spoons_r == 1 else 'tbsp'
        return f'~{int(spoons_r)} {unit}'

    def slices_from(gm: float, grams_per_slice: float = 28.0) -> str:
        if gm <= 0:
            return ''
        cnt = max(1, int(round(gm / grams_per_slice)))
        unit = 'slice' if cnt == 1 else 'slices'
        return f'{cnt} {unit}'

    def ounces_from(gm: float) -> str:
        if gm <= 0:
            return ''
        oz = gm / 28.3495
        oz_r = round(oz, 1)
        return f'~{oz_r:g} oz'

    portion = ''
    if 'mashed potatoes' in label.lower():
        portion = cups_from(g, 210.0)  # ~210 g per cup mashed potatoes
    elif 'salisbury steak' in label.lower() or 'chicken breast' in label.lower():
        portion = ounces_from(g)
    elif 'meatloaf' in label.lower():
        portion = ounces_from(g)
    elif 'steamed white rice' in label.lower():
        portion = cups_from(g, 158.0)  # cooked rice ~158 g/cup
    elif 'steamed peas & carrots' in label.lower():
        portion = cups_from(g, 160.0)
    elif 'cooked beans' in label.lower():
        portion = cups_from(g, 170.0)  # cooked beans ~170 g/cup
    elif 'shredded red cabbage' in label.lower():
        portion = cups_from(g, 70.0)
    elif 'cherries' in label.lower():
        portion = cups_from(g, 140.0)
    elif 'beef hot dog on bun' in label.lower():
        if g >= 90:
            cnt = max(1, int(round(g / 120.0)))
            unit = 'sandwich' if cnt == 1 else 'sandwiches'
            portion = f'~{cnt} {unit}'
        else:
            portion = ounces_from(g)
    elif 'ketchup' in label.lower():
        portion = tbsp_from(g, 17.0)
    elif 'mayonnaise' in label.lower():
        portion = tbsp_from(g, 14.0)
    elif 'whole wheat bread' in label.lower():
        portion = slices_from(g, 28.0)
    elif 'fruit juice' in label.lower():
        portion = cups_from(g, 240.0)  # 240 g ~ 8 fl oz
    elif 'drinkable yogurt' in label.lower():
        portion = cups_from(g, 120.0)  # ~120 g per 1/2 cup
    elif 'low-fat yogurt' in label.lower():
        portion = cups_from(g, 227.0)  # ~227 g per cup; 3/4 cup ~ 170 g

    if portion:
        return f"{label} ({portion})"
    return label


def _beautify_items(items: List[Tuple[str, float, bool]]) -> List[Tuple[str, float, bool]]:
    out = []
    for name, grams, is_added in items:
        txt = _friendly_portion(name, grams)
        txt = _replace_fractions(txt)
        # Preserve actual grams so the rendered bullets show both portion and grams
        out.append((txt, float(grams), is_added))
    return out
def _portion_spec_for_name(name: str) -> Tuple[float, float]:
    if rp_portion_spec_for_label is not None:
        return rp_portion_spec_for_label(name)
    # Fallback to local defaults
    n = name.lower()
    if 'salisbury steak' in n:
        return (170.0, 13.00)
    if 'meatloaf' in n:
        return (170.0, 12.50)
    if 'chicken breast' in n:
        return (170.0, 12.00)
    if 'beef hot dog on bun' in n or ('hot dog' in n and 'bun' in n):
        return (120.0, 4.00)
    if 'whole wheat bread' in n or ('bread' in n and 'wheat' in n):
        return (56.0, 1.00)
    if 'steamed white rice' in n:
        return (158.0, 3.00)
    if 'steamed peas & carrots' in n or 'peas and carrots' in n:
        return (160.0, 3.50)
    if 'cooked beans' in n:
        return (170.0, 4.00)
    if 'shredded red cabbage' in n:
        return (70.0, 3.00)
    if 'cherries' in n:
        return (140.0, 4.00)
    if 'low-fat yogurt' in n:
        return (170.0, 3.00)
    if 'drinkable yogurt' in n:
        return (120.0, 2.50)
    if 'fruit juice' in n or 'juice' in n:
        return (240.0, 2.50)
    if 'ketchup' in n:
        return (17.0, 0.10)
    if 'mayonnaise' in n:
        return (14.0, 0.20)
    if 'iced coffee' in n or ('coffee' in n and 'iced' in n):
        return (240.0, 3.00)
    if 'water' in n:
        return (240.0, 0.00)
    return (150.0, 3.00)


def _max_portions_for_name(name: str) -> float:
    if rp_max_portions_for_label is not None:
        return rp_max_portions_for_label(name)
    n = name.lower()
    if 'water' in n:
        return 0.0
    if 'fruit juice' in n or 'juice' in n:
        return 1.0
    if 'drinkable yogurt' in n or 'low-fat yogurt' in n:
        return 1.0
    if 'iced coffee' in n or 'coffee' in n:
        return 1.0
    if 'ketchup' in n or 'mayonnaise' in n:
        return 2.0
    if any(k in n for k in ['pizza', 'burrito', 'sandwich', 'submarine', 'burger', 'casserole', 'steak', 'meatloaf']):
        return 1.5
    if any(k in n for k in ['rice', 'beans', 'peas & carrots', 'peas and carrots', 'cabbage']):
        return 2.0
    return 3.0


def _estimate_cost_from_vec(meal_vec: np.ndarray, data: Dict) -> float:
    if rp_estimate_restaurant_cost is not None:
        return float(rp_estimate_restaurant_cost(meal_vec, data['feature_names'], data['code2name'], overhead_usd=2.00))
    # Fallback to local estimation
    total = 0.0
    for j, code in enumerate(data['feature_names']):
        grams = float(meal_vec[j])
        if grams <= 0:
            continue
        nm = _clean_name_display(code, data['code2name'])
        friendly = _friendly_portion(nm, grams)
        base = friendly.split(' (')[0].strip()
        g_per, price_per = _portion_spec_for_name(base)
        portions = grams / max(g_per, 1e-6)
        cap = _max_portions_for_name(base)
        portions = min(portions, cap)
        total += portions * price_per
    return round(total + 2.00, 2)


def _apply_restaurant_pricing(df: pd.DataFrame, data: Dict) -> pd.DataFrame:
    """Compute restaurant-style costs for each (Meal Type, gen_idx, real_idx) row
    and overwrite delta_cost/real_cost/gen_cost so downstream filters use them by default.
    """
    try:
        mt_col = 'Meal Type'
        if (mt_col not in df.columns) or ('gen_idx' not in df.columns) or ('real_idx' not in df.columns):
            return df
        real_costs = []
        gen_costs = []
        deltas = []
        # Iterate over columns directly: itertuples() renames 'Meal Type' (not an identifier).
        for mt, gi, ri in zip(df[mt_col], df['gen_idx'], df['real_idx']):
            try:
                real_vec = data['real_by_type'][mt][int(ri)]
                gen_vec = data['gen_by_type'][mt][int(gi)]
                rc = _estimate_cost_from_vec(real_vec, data)
                gc = _estimate_cost_from_vec(gen_vec, data)
                real_costs.append(rc)
                gen_costs.append(gc)
                deltas.append(gc - rc)
            except Exception:
                real_costs.append(float('nan'))
                gen_costs.append(float('nan'))
                deltas.append(float('nan'))
        df = df.copy()
        df['real_cost'] = real_costs
        df['gen_cost'] = gen_costs
        df['delta_cost'] = deltas
        with np.errstate(divide='ignore', invalid='ignore'):
            df['dCost_pct'] = 100.0 * (pd.to_numeric(df['delta_cost'], errors='coerce') / pd.to_numeric(df['real_cost'], errors='coerce'))
    except Exception:
        pass
    return df


def _replace_fractions(s: str) -> str:
    # Replace common ASCII fractions with Unicode vulgar fractions for stacked look
    repl = {
        '1/2': '½', '1/4': '¼', '3/4': '¾', '1/3': '⅓', '2/3': '⅔',
        '1 1/2': '1½', '1 1/4': '1¼', '1 3/4': '1¾', '1 1/3': '1⅓', '1 2/3': '1⅔'
    }
    # Replace longer patterns first to avoid partial replacements
    for k in sorted(repl.keys(), key=len, reverse=True):
        s = s.replace(k, repl[k])
    return s


def export_top_candidates(candidates: List[Dict], data: Dict, outdir: Path, top_k: int = 50) -> None:
    if not candidates:
        return
    cand_sorted = sorted(candidates, key=lambda c: c['score'], reverse=True)[:int(top_k)]
    rows = []
    for rank, c in enumerate(cand_sorted, start=1):
        mt = c['Meal Type']; ridx = int(c['real_idx'])
        r1 = c['r1']; r2 = c['r2']
        base_meal = data['real_by_type'][mt][ridx]
        base_items = '; '.join(_meal_to_str_list(mt, base_meal, data))
        # k1
        g1 = data['gen_by_type'][mt][int(r1['gen_idx'])]
        k1_meal = '; '.join(_meal_to_str_list(mt, g1, data))
        k1_added_codes = r1.get('added_list', [])
        k1_removed_codes = r1.get('removed_list', [])
        k1_added = []
        for code in k1_added_codes:
            if code in data['feature_names']:
                j = data['feature_names'].index(code)
                grams = float(g1[j])
            else:
                grams = 0.0
            k1_added.append(f"{_clean_name_display(code, data['code2name'])} ({grams:.0f}g)")
        k1_removed = []
        for code in k1_removed_codes:
            if code in data['feature_names']:
                j = data['feature_names'].index(code)
                grams = float(base_meal[j])
            else:
                grams = 0.0
            k1_removed.append(f"{_clean_name_display(code, data['code2name'])} ({grams:.0f}g)")
        # k2
        g2 = data['gen_by_type'][mt][int(r2['gen_idx'])]
        k2_meal = '; '.join(_meal_to_str_list(mt, g2, data))
        k2_added_codes = r2.get('added_list', [])
        k2_removed_codes = r2.get('removed_list', [])
        k2_added = []
        for code in k2_added_codes:
            if code in data['feature_names']:
                j = data['feature_names'].index(code)
                grams = float(g2[j])
            else:
                grams = 0.0
            k2_added.append(f"{_clean_name_display(code, data['code2name'])} ({grams:.0f}g)")
        k2_removed = []
        for code in k2_removed_codes:
            if code in data['feature_names']:
                j = data['feature_names'].index(code)
                grams = float(base_meal[j])
            else:
                grams = 0.0
            k2_removed.append(f"{_clean_name_display(code, data['code2name'])} ({grams:.0f}g)")
        rows.append({
            'rank': rank,
            'meal_type': mt,
            'real_idx': ridx,
            'score': float(c.get('score', 0.0)),
            'k1_gen_idx': int(r1.get('gen_idx', -1)),
            'k1_improvement': float(r1.get('improvement', 0.0)),
            'k1_delta_cost': float(r1.get('delta_cost', 0.0)),
            'k1_added': '; '.join(k1_added),
            'k1_removed': '; '.join(k1_removed),
            'k1_meal': k1_meal,
            'k2_gen_idx': int(r2.get('gen_idx', -1)),
            'k2_improvement': float(r2.get('improvement', 0.0)),
            'k2_delta_cost': float(r2.get('delta_cost', 0.0)),
            'k2_added': '; '.join(k2_added),
            'k2_removed': '; '.join(k2_removed),
            'k2_meal': k2_meal,
            'original_meal': base_items,
        })
    try:
        import pandas as _pd
        df = _pd.DataFrame(rows)
        df.to_csv(outdir / 'candidate_meals_top50.csv', index=False)
        # Also export JSON for nested readability
        import json as _json
        (outdir / 'candidate_meals_top50.json').write_text(_json.dumps(rows, indent=2))
        print(f"Exported top-{min(len(rows), int(top_k))} candidates to: {outdir / 'candidate_meals_top50.csv'}")
    except Exception as e:
        print(f"[warn] failed to export candidates: {e}")


def export_vitamins_up_fats_down(matches_csv: Union[Path, List[Path]], data: Dict, outdir: Path,
                                 min_foods: int = 4,
                                 sort_by: str = 'vitamin_increase') -> None:
    """Export swaps where vitamins increase and MUFA/SFA decrease.

    We scan the matches CSV (filtered to winners) and compute nutrient deltas
    for each (generated, real) pair. Rows are emitted when:
      sum(delta over vitamin keys) > 0 AND delta_mufa < 0 AND delta_sfa < 0.
    """
    try:
        df = _read_matches_any(matches_csv)
    except Exception as e:
        print(f"[warn] failed to read matches for vitamin/fat analysis: {e}")
        return
    if 'selected' in df.columns:
        df = df[df['selected'] == True]
    df = df[(df.get('improvement', 0) > 0) & (df.get('delta_cost', 0) <= 0)]
    if df.empty:
        return

    vitamin_keys = _groups.get('Vitamins', [])
    # Build index map once
    name_to_idx = {n: i for i, n in enumerate(data['nutrient_names'])}
    vit_idx = [name_to_idx[k] for k in vitamin_keys if k in name_to_idx]
    mufa_idx = name_to_idx.get('mono_fat_g', None)
    sfa_idx = name_to_idx.get('sfa_g', None)
    rows = []
    for r in df.itertuples(index=False):
        try:
            mt = getattr(r, 'Meal Type')
            gi = int(getattr(r, 'gen_idx'))
            ri = int(getattr(r, 'real_idx'))
            gen = data['gen_by_type'][mt][gi]
            real = data['real_by_type'][mt][ri]
            # Minimum foods constraint for realism
            if int(np.sum(gen > 0)) < int(min_foods):
                continue
            gN = (gen / 100.0) @ data['W']
            rN = (real / 100.0) @ data['W']
            dN = gN - rN
            vit_delta = float(np.nansum(dN[vit_idx])) if vit_idx else float('nan')
            d_mufa = float(dN[mufa_idx]) if mufa_idx is not None else float('nan')
            d_sfa = float(dN[sfa_idx]) if sfa_idx is not None else float('nan')
            cond = (np.isfinite(vit_delta) and vit_delta > 0.0 and
                    (not np.isnan(d_mufa)) and (d_mufa < 0.0) and
                    (not np.isnan(d_sfa)) and (d_sfa < 0.0))
            if not cond:
                continue
            rows.append({
                'meal_type': mt,
                'gen_idx': gi,
                'real_idx': ri,
                'vitamin_increase': vit_delta,
                'delta_mufa': d_mufa,
                'delta_sfa': d_sfa,
                'improvement': float(getattr(r, 'improvement', np.nan)),
                'delta_cost': float(getattr(r, 'delta_cost', np.nan)),
                'added_foods': getattr(r, 'added_foods', ''),
                'removed_foods': getattr(r, 'removed_foods', ''),
            })
        except Exception:
            continue
    if not rows:
        print("[info] No swaps found with vitamins up and MUFA/SFA down.")
        return
    try:
        import pandas as _pd
        df_out = _pd.DataFrame(rows)
        if sort_by in df_out.columns:
            df_out = df_out.sort_values(sort_by, ascending=False)
        fp = outdir / 'candidate_swaps_vitamins_up_fats_down.csv'
        df_out.to_csv(fp, index=False)
        print(f"Exported vitamin-up fat-down swaps to: {fp}")
    except Exception as e:
        print(f"[warn] failed to export vitamin/fat analysis: {e}")

def compute_nutrients(gen_meal: np.ndarray, real_meal: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_gen = (gen_meal / 100.0) @ W
    n_real = (real_meal / 100.0) @ W
    return n_gen, n_real


def log10_fold_change(n_gen: np.ndarray, n_real: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.log10((n_gen + eps) / (n_real + eps))


def panel_title(meta: Dict, idx: int, code2name: Dict, feature_names: List[str],
                gen_meal: np.ndarray, real_meal: np.ndarray,
                W: Optional[np.ndarray] = None, RDI: Optional[np.ndarray] = None) -> str:
    mt = meta.get('Meal Type', '')
    k = int(meta.get('k_substitution', meta.get('total_food_diff', 0)))
    # Report improvement as percent change relative to the real meal's deviation
    imp = float(meta.get('improvement', np.nan))
    real_dev = float(meta.get('real_deviation', np.nan))
    dpp = np.nan
    # Prefer explicit percent if both values exist
    if np.isfinite(imp) and np.isfinite(real_dev) and real_dev != 0:
        dpp = (imp / real_dev) * 100.0
    # Otherwise, recompute from RDI and W to ensure a percentage
    if (not np.isfinite(dpp)) and (W is not None) and (RDI is not None):
        try:
            rN = (real_meal / 100.0) @ W
            gN = (gen_meal / 100.0) @ W
            with np.errstate(divide='ignore', invalid='ignore'):
                r_ratio = np.where(RDI > 0, rN / RDI, np.nan)
                g_ratio = np.where(RDI > 0, gN / RDI, np.nan)
            r_dev2 = np.nanmean(np.abs(r_ratio - 1.0))
            g_dev2 = np.nanmean(np.abs(g_ratio - 1.0))
            if np.isfinite(r_dev2) and r_dev2 > 0:
                dpp = (r_dev2 - g_dev2) / r_dev2 * 100.0
        except Exception:
            pass
    if not np.isfinite(dpp):
        dpp = float('nan')
    dcost = float(meta.get('delta_cost', np.nan))
    # Summary-only title; colored bullet foods drawn separately
    t = f"{chr(ord('a')+idx)}  {mt}; k-sub={k}; ΔRDI={dpp:+.1f}%; Δ$={dcost:+.2f}"
    return t


def _clean_food_name(code: str, code2name: Dict[str, str]) -> str:
    import re as _re
    nm = code2name.get(code, code)
    m = _re.findall(r"\(([^)]+)\)", nm)
    if m:
        return m[-1].strip()
    return nm.strip()


def foods_to_show_with_highlights(mt: str, gen_idx: int, real_idx: int, meta: Dict, data: Dict) -> List[Tuple[str, float, bool]]:
    feature_names = data['feature_names']
    code2name = data['code2name']
    gen_meal = data['gen_by_type'][mt][gen_idx]
    real_meal = data['real_by_type'][mt][real_idx]
    added_list = _parse_food_list(meta.get('added_foods', ''))
    removed_list = _parse_food_list(meta.get('removed_foods', ''))
    removed_set = set(removed_list)
    # Base foods not removed (use gen grams if available; else real grams)
    items: List[Tuple[str, float, bool]] = []
    for j, code in enumerate(feature_names):
        if real_meal[j] > 0 and (code not in removed_set):
            grams = float(gen_meal[j]) if gen_meal[j] > 0 else float(real_meal[j])
            items.append((_clean_food_name(code, code2name), grams, False))
    # Then added foods (highlight)
    for code in added_list:
        if code in feature_names:
            j = feature_names.index(code)
            grams = float(gen_meal[j]) if gen_meal[j] > 0 else float(real_meal[j])
        else:
            grams = 0.0
        items.append((_clean_food_name(code, code2name), grams, True))
    return items


def draw_food_bullets(
    ax,
    items: List[Tuple[str, float, bool]],
    start_y: float = 1.03,
    dy: float = 0.050,
    wrap_cols: int = 42,
    *,
    show_grams: bool = True,
    red_names: Optional[set] = None,
) -> None:
    import textwrap as _tw
    y = start_y
    for name, grams, is_added in items:
        # Normalize fractions for readability (e.g., 1/3 -> ⅓)
        name = _replace_fractions(str(name))
        # Color rules: added → green, optional red for names included in red_names
        # Normalize both sides: strip any trailing portions in parentheses and lowercase
        if red_names:
            try:
                import re as _re
                base_nm = _re.sub(r"\s*\(.*\)$", "", str(name)).strip().lower()
                red_norm = {str(x).strip().lower() for x in red_names}
                is_red = base_nm in red_norm
            except Exception:
                is_red = False
        else:
            is_red = False
        if is_red:
            color = '#c0392b'
        else:
            color = '#27ae60' if is_added else '#000000'
        # Show grams if provided (>0)
        if show_grams and grams and grams > 0:
            s = f"• {name} — {grams:.0f} g"
        else:
            s = f"• {name}"
        # Wrap long names to avoid clipping; keep within axis width
        lines = _tw.wrap(s, width=int(wrap_cols)) or [s]
        text = "\n".join(lines)
        ax.text(
            0.0, y, text,
            transform=ax.transAxes,
            ha='left', va='top', fontsize=8, color=color,
            clip_on=False, zorder=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.0)
        )
        # Advance y by number of lines
        y -= dy * max(1, len(lines))


def plot_pair(ax, nutrient_names: List[str], lfc: np.ndarray, highlight_thresh: float = 2.0, title: str = '') -> None:
    order = np.argsort(-np.abs(lfc))
    lfc_sorted = lfc[order]
    names_sorted = [nutrient_names[i] for i in order]
    y = np.arange(len(names_sorted))
    colors = np.where(np.abs(lfc_sorted) >= highlight_thresh, '#c0392b', '#27ae60')
    ax.scatter(lfc_sorted, y, s=30, c=colors, edgecolor='none', alpha=0.9)
    ax.axvline(0.0, color='gray', lw=1.0)
    ax.axvline(+highlight_thresh, color='gray', lw=0.8, ls='--')
    ax.axvline(-highlight_thresh, color='gray', lw=0.8, ls='--')
    ax.set_yticks(y)
    ax.set_yticklabels(names_sorted, fontsize=7)
    ax.set_xlim(-3.1, 3.1)
    ax.set_xlabel('log10 fold-change (generated / real)')
    ax.set_title(title, fontsize=10)


def plot_butterfly_two(
    ax_left, ax_mid, ax_right,
    names: List[str],
    lfc_left: np.ndarray,
    lfc_right: np.ndarray,
    highlight_thresh: float,
    title_left: str,
    title_right: str,
    rdi_targets: Optional[np.ndarray] = None,
    meal_fraction: float = 0.33,
    real_amounts: Optional[np.ndarray] = None,
    gen_left_amounts: Optional[np.ndarray] = None,
    gen_right_amounts: Optional[np.ndarray] = None,
    # Optional extras for right-panel nutrient contribution inset
    real_meal_vec: Optional[np.ndarray] = None,
    gen_right_meal_vec: Optional[np.ndarray] = None,
    W_matrix: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    code2name: Optional[Dict[str, str]] = None,
    inset_nutrient_key: Optional[str] = 'vitamin_d_iu',
    nutrient_names_full: Optional[List[str]] = None,
    # Substitution-specific: restrict insets to changed foods only
    added_codes_right: Optional[set] = None,
    removed_codes_right: Optional[set] = None,
) -> None:
    """Two ratio panels (generated / real nutrient amounts, log scale 1e-2..1e2) around the
    nutrient names drawn in ax_mid: left = k=1 pair, right = k=2 pair. A point is green when
    the generated amount is closer to the meal target (rdi_targets * meal_fraction) than the
    real one, red otherwise; '|' marks target / real. Insets beside the right panel show the
    swapped foods' contributions to inset_nutrient_key and to cholesterol.
    title_left and title_right are not drawn.
    """
    # Order by group, then by magnitude within group
    mag = np.maximum(np.abs(lfc_left), np.abs(lfc_right))
    group_idx = np.array([NUTRIENT_GROUP_INDEX.get(names[i], 99) for i in range(len(names))])
    order = np.lexsort(( -mag, group_idx ))  # primary: group_idx; secondary: -magnitude
    names_sorted = [names[i] for i in order]
    l_left = lfc_left[order]
    l_right = lfc_right[order]
    y = np.arange(len(names_sorted))

    # Middle axis: draw centered nutrient names as text (not ytick labels)
    ax_mid.set_xlim(0.0, 1.0)
    ax_mid.set_xticks([])
    ax_mid.set_ylim(-0.5, len(y)-0.5)
    ax_mid.set_yticks([])
    for spine in ax_mid.spines.values():
        spine.set_visible(False)
    # Faint connector lines across left and right axes only (not drawn through labels)
    for yi in y:
        ax_left.axhline(yi, color='0.7', lw=0.8, alpha=0.45, zorder=0)
        ax_right.axhline(yi, color='0.7', lw=0.8, alpha=0.45, zorder=0)
    # Place labels with blended transform (x in axes coords, y in data coords)
    trans = blended_transform_factory(ax_mid.transAxes, ax_mid.transData)
    texts = []
    for yi, key in enumerate(names_sorted):
        label = NUTRIENT_LABELS.get(key, key)
        t = ax_mid.text(0.5, yi, label, ha='center', va='center', fontsize=9.5, transform=trans, zorder=2)
        texts.append((t, yi))
    # After placing text, compute extents to draw connector segments up to text edges (flush)
    fig = ax_mid.figure
    try:
        fig.canvas.draw()
        inv_axes = ax_mid.transAxes.inverted()
        for t, yi in texts:
            bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
            x0_axes = inv_axes.transform((bb.x0, 0))[0]
            x1_axes = inv_axes.transform((bb.x1, 0))[0]
            gap = 0.010  # small padding around text
            # left segment: from left boundary to just before text
            ax_mid.plot([0.0, max(x0_axes - gap, 0.0)], [yi, yi], color='0.7', lw=0.8, alpha=0.45, transform=trans, zorder=1)
            # right segment: from just after text to right boundary
            ax_mid.plot([min(x1_axes + gap, 1.0), 1.0], [yi, yi], color='0.7', lw=0.8, alpha=0.45, transform=trans, zorder=1)
    except Exception:
        pass

    # Plot left panel on ratio scale (1e-2 .. 1e2) around 1.0
    axis_min, axis_max = 1e-2, 1e2
    r_left = np.power(10.0, l_left)
    # Clip extreme ratios to axis bounds so points remain visible
    r_left = np.clip(r_left, axis_min, axis_max)
    # Color by movement toward RDI target when available; else direction
    if (rdi_targets is not None) and (real_amounts is not None) and (gen_left_amounts is not None):
        try:
            t = (np.array([rdi_targets[names.index(k)] for k in names_sorted], dtype=float) * float(meal_fraction))
            r = np.array([real_amounts[names.index(k)] for k in names_sorted], dtype=float)
            g = np.array([gen_left_amounts[names.index(k)] for k in names_sorted], dtype=float)
            closer = np.abs(g - t) < np.abs(r - t)
            colors_l = np.where(closer, '#27ae60', '#c0392b')
        except Exception:
            colors_l = np.where(r_left >= 1.0, '#27ae60', '#c0392b')
    else:
        colors_l = np.where(r_left >= 1.0, '#27ae60', '#c0392b')
    ax_left.set_xscale('log')
    ax_left.set_xlim(axis_min, axis_max)
    ax_left.scatter(r_left, y, s=60, c=colors_l, edgecolor='none', alpha=0.9, zorder=3)
    # Draw origin reference line at ratio=1
    ax_left.axvline(1.0, color='0.7', lw=0.8, ls='--', alpha=0.6, zorder=1)
    # RDI indicator for each nutrient: target/real ratio
    if (rdi_targets is not None) and (real_amounts is not None):
        try:
            # align order
            t = (np.array([rdi_targets[names.index(k)] for k in names_sorted], dtype=float) * float(meal_fraction))
            r = np.array([real_amounts[names.index(k)] for k in names_sorted], dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                tr_raw = np.divide(t, r, out=np.full_like(t, np.inf), where=r > 0)
            # Clip to visible axis range
            tr = np.clip(tr_raw, axis_min, axis_max)
            # Larger, higher-contrast RDI marker
            ax_left.scatter(tr, y, s=36, marker='|', color='#4d4d4d', alpha=0.9, zorder=4)
            # Add edge annotation for infinite targets (real=0): draw a small arrow at axis edge
            inf_mask = ~np.isfinite(tr_raw) | (tr_raw > axis_max)
            if np.any(inf_mask):
                for yi, flag in enumerate(inf_mask):
                    if flag:
                        ax_left.annotate('target', xy=(axis_max, yi), xytext=(0, 0), textcoords='offset points',
                                         ha='left', va='center', fontsize=7, color='#4d4d4d',
                                         arrowprops=dict(arrowstyle='-|>', color='#4d4d4d', lw=0.8),
                                         zorder=5)
        except Exception:
            pass
    ax_left.set_ylim(-0.5, len(y)-0.5)
    ax_left.set_yticks(y)
    ax_left.set_yticklabels([])
    ax_left.tick_params(axis='y', which='both', labelleft=False, left=False, length=0)
    ax_left.set_xlabel('Generated nutrients / Real nutrients')
    ax_left.xaxis.set_major_locator(LogLocator(base=10.0))
    ax_left.xaxis.set_major_formatter(LogFormatterMathtext())
    ax_left.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1)))
    # No panel titles
    ax_left.set_title('')

    # RDI target highlight on left panel for Folate (mcg)
    try:
        fol_key = 'folate_mcg'
        if (rdi_targets is not None) and (fol_key in names):
            fol_sorted_idx = names_sorted.index(fol_key)
            x_fol = float(tr[fol_sorted_idx]) if 'tr' in locals() else None
            y_fol = float(y[fol_sorted_idx])
            if x_fol is not None and np.isfinite(x_fol):
                # Arrow from the right, with extra padding, and star at arrow tip
                tip_x = min(axis_max, x_fol * 1.40)
                tip_y = y_fol
                # Lift the tail to align vertically with the star text baseline
                ax_left.annotate(
                    '',
                    xy=(x_fol * 1.02, y_fol + 0.50), xycoords=ax_left.transData,
                    xytext=(tip_x, tip_y), textcoords=ax_left.transData,
                    arrowprops=dict(arrowstyle='-|>', color='#4d4d4d', lw=1.2, connectionstyle='arc3,rad=0.25')
                )
                # Larger star; keep label bold; align star and text at same y
                ax_left.text(tip_x + 0.02, tip_y + 0.50, '★ RDI target', ha='left', va='bottom', fontsize=10.0, fontweight='bold', color='#4d4d4d', transform=ax_left.transData)
    except Exception:
        pass

    # Right panel: ratio scale (1e-2 .. 1e2)
    r_right = np.power(10.0, l_right)
    # Clip extreme ratios to axis bounds so points remain visible
    r_right = np.clip(r_right, axis_min, axis_max)
    if (rdi_targets is not None) and (real_amounts is not None) and (gen_right_amounts is not None):
        try:
            t = (np.array([rdi_targets[names.index(k)] for k in names_sorted], dtype=float) * float(meal_fraction))
            r = np.array([real_amounts[names.index(k)] for k in names_sorted], dtype=float)
            g = np.array([gen_right_amounts[names.index(k)] for k in names_sorted], dtype=float)
            closer = np.abs(g - t) < np.abs(r - t)
            colors_r = np.where(closer, '#27ae60', '#c0392b')
        except Exception:
            colors_r = np.where(r_right >= 1.0, '#27ae60', '#c0392b')
    else:
        colors_r = np.where(r_right >= 1.0, '#27ae60', '#c0392b')
    ax_right.set_xscale('log')
    ax_right.set_xlim(axis_min, axis_max)
    ax_right.scatter(r_right, y, s=60, c=colors_r, edgecolor='none', alpha=0.9, zorder=3)
    # Draw origin reference line at ratio=1
    ax_right.axvline(1.0, color='0.7', lw=0.8, ls='--', alpha=0.6, zorder=1)
    if (rdi_targets is not None) and (real_amounts is not None):
        try:
            t = (np.array([rdi_targets[names.index(k)] for k in names_sorted], dtype=float) * float(meal_fraction))
            r = np.array([real_amounts[names.index(k)] for k in names_sorted], dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                tr_raw = np.divide(t, r, out=np.full_like(t, np.inf), where=r > 0)
            tr = np.clip(tr_raw, axis_min, axis_max)
            ax_right.scatter(tr, y, s=36, marker='|', color='#4d4d4d', alpha=0.9, zorder=4)
            inf_mask = ~np.isfinite(tr_raw) | (tr_raw > axis_max)
            if np.any(inf_mask):
                for yi, flag in enumerate(inf_mask):
                    if flag:
                        ax_right.annotate('target', xy=(axis_max, yi), xytext=(0, 0), textcoords='offset points',
                                          ha='left', va='center', fontsize=7, color='#4d4d4d',
                                          arrowprops=dict(arrowstyle='-|>', color='#4d4d4d', lw=0.8),
                                          zorder=5)
        except Exception:
            pass
    ax_right.set_ylim(-0.5, len(y)-0.5)
    ax_right.set_yticks(y)
    ax_right.set_yticklabels([])
    ax_right.tick_params(axis='y', which='both', labelleft=False, left=False, right=False, length=0)
    ax_right.set_xlabel('Generated nutrients / Real nutrients')
    ax_right.xaxis.set_major_locator(LogLocator(base=10.0))
    ax_right.xaxis.set_major_formatter(LogFormatterMathtext())
    ax_right.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1)))
    # No panel titles
    ax_right.set_title('')

    # Optional: inset showing per-food contributions for the chosen nutrient (right panel)
    try:
        if (real_meal_vec is not None) and (gen_right_meal_vec is not None) and (W_matrix is not None) and feature_names:
            vit_key = inset_nutrient_key
            # Allow fallback key naming
            if (vit_key not in (nutrient_names_full or names)) and ('vitamin_d_mcg' in (nutrient_names_full or names)):
                vit_key = 'vitamin_d_mcg'
            # Index in the full nutrient list for contributions
            if (nutrient_names_full is not None) and (vit_key in nutrient_names_full):
                vit_idx = nutrient_names_full.index(vit_key)
            elif vit_key in names:
                vit_idx = names.index(vit_key)
            else:
                vit_idx = None
            if vit_idx is not None:
                # Anchor to dot if present in the plotted (filtered) list
                has_dot = vit_key in names_sorted
                if has_dot:
                    vit_sorted_idx = names_sorted.index(vit_key)
                    x_dot = float(r_right[vit_sorted_idx])
                    y_dot = float(y[vit_sorted_idx])
                # Compute per-food contributions
                # Consider ONLY foods present in either meal to avoid non-meal items in inset
                present_mask = (np.array(real_meal_vec) > 0) | (np.array(gen_right_meal_vec) > 0)
                present_idx = np.where(present_mask)[0]
                if (added_codes_right is not None) or (removed_codes_right is not None):
                    changed_codes = set(added_codes_right or set()) | set(removed_codes_right or set())
                    changed_idx = np.array([i for i, c in enumerate(feature_names) if c in changed_codes], dtype=int)
                    present_idx = np.array([i for i in present_idx if i in set(changed_idx)], dtype=int)
                    if present_idx.size == 0:
                        present_idx = np.where(present_mask)[0]
                # Align W row index: contributions per 100 g of each food for this nutrient
                w_col = np.array([W_matrix[j, vit_idx] for j in present_idx], dtype=float)
                real_contrib = (np.array(real_meal_vec, dtype=float)[present_idx] / 100.0) * w_col
                gen_contrib = (np.array(gen_right_meal_vec, dtype=float)[present_idx] / 100.0) * w_col
                delta = gen_contrib - real_contrib
                # Split into removed vs added groups and rank within each group
                try:
                    added_codes = set(added_codes_right or set())
                    removed_codes = set(removed_codes_right or set())
                except Exception:
                    added_codes, removed_codes = set(), set()
                codes_all = [feature_names[i] for i in present_idx]
                vals_all = [delta[k] for k in range(len(present_idx))]
                removed_items = [
                    (c, (code2name.get(c, c) if code2name else c), float(v))
                    for c, v in zip(codes_all, vals_all) if c in removed_codes
                ]
                added_items = [
                    (c, (code2name.get(c, c) if code2name else c), float(v))
                    for c, v in zip(codes_all, vals_all) if c in added_codes
                ]
                removed_items.sort(key=lambda x: -abs(x[2]))
                added_items.sort(key=lambda x: -abs(x[2]))
                max_total = 6
                target = 3
                rem_sel = removed_items[:target]
                add_sel = added_items[:target]
                remaining = max_total - (len(rem_sel) + len(add_sel))
                if remaining > 0:
                    extra = (removed_items[target:] + added_items[target:])
                    extra.sort(key=lambda x: -abs(x[2]))
                    for (c_e, l_e, v_e) in extra[:remaining]:
                        if c_e in removed_codes:
                            rem_sel.append((c_e, l_e, v_e))
                        else:
                            add_sel.append((c_e, l_e, v_e))
                seq = rem_sel + add_sel  # removed first, then added
                codes_idx = [c for (c, _, _) in seq]
                labels = [lbl for (_, lbl, _) in seq]
                vals = np.array([v for (_, _, v) in seq], dtype=float)
                colors = np.where(vals >= 0, '#27ae60', '#c0392b')
                # Place inset OUTSIDE the right panel, with dotted connectors
                fig = ax_right.figure
                # Compute figure coords of the dot
                x_fig, y_fig = fig.transFigure.inverted().transform(ax_right.transData.transform((x_dot, y_dot)))
                # Base rectangle of ax_right in figure coords
                bbox = ax_right.get_position()
                # Inset size as fraction of figure (narrower)
                w, h = 0.15, 0.17
                # Pin inset near the right canvas edge to avoid overlap
                x0 = 0.98 - w
                y0 = max(0.05, 0.90 - h)
                ax_in = fig.add_axes([x0, y0, w, h])
                # Lollipop plot: line from 0 to value, circle at value
                yy = np.arange(len(vals))
                for i, v in enumerate(vals):
                    x_start = 0.0 if v >= 0 else v
                    x_end = v if v >= 0 else 0.0
                    ax_in.hlines(y=yy[i], xmin=x_start, xmax=x_end, colors=colors[i], lw=1.6)
                    ax_in.plot([v], [yy[i]], marker='o', markersize=4.5, color=colors[i])
                ax_in.set_yticks(np.arange(len(vals)))
                # Shorten labels to avoid overflow
                def _short(s: str) -> str:
                    return s if len(s) <= 26 else (s[:23] + '…')
                ax_in.set_yticklabels([_short(s) for s in labels], fontsize=7)
                # Divider between removed and added groups
                try:
                    n_removed = len(rem_sel)
                    n_total = len(seq)
                    if n_removed > 0 and n_removed < n_total:
                        ax_in.set_ylim(-0.5, n_total - 0.5)
                        ax_in.axhline(n_removed - 0.5, color='0.7', lw=0.8)
                except Exception:
                    pass
                # Bold substituted items (added or removed)
                try:
                    added_codes = set([feature_names[j] for j in np.where((np.array(real_meal_vec)<=0) & (np.array(gen_right_meal_vec)>0))[0]])
                    removed_codes = set([feature_names[j] for j in np.where((np.array(real_meal_vec)>0) & (np.array(gen_right_meal_vec)<=0))[0]])
                    for tick, code in zip(ax_in.get_yticklabels(), codes_idx):
                        if (code in added_codes) or (code in removed_codes):
                            tick.set_fontweight('bold')
                except Exception:
                    pass
                # Move labels to the right side
                ax_in.yaxis.tick_right()
                ax_in.tick_params(axis='y', labelright=True, labelleft=False)
                ax_in.axvline(0.0, color='0.6', lw=0.8, ls='--')
                ax_in.tick_params(axis='x', labelsize=7)
                # Label the x-axis with units when known
                # Dynamic title based on chosen nutrient key
                try:
                    _nut_label = NUTRIENT_LABELS.get(vit_key, vit_key)
                except Exception:
                    _nut_label = vit_key
                # Simplify title: nutrient name without units + "contributions"
                try:
                    base_label = _nut_label.split(' (')[0]
                except Exception:
                    base_label = _nut_label
                # Reinstate inset title (contributors)
                ax_in.set_title(f'{base_label} contributors', fontsize=8, pad=2)
                # Determine units from label text for axis label
                try:
                    unit_txt_axis = _nut_label[_nut_label.rfind('(')+1:_nut_label.rfind(')')] if '(' in _nut_label and ')' in _nut_label else ''
                except Exception:
                    unit_txt_axis = ''
                if unit_txt_axis:
                    ax_in.set_xlabel(f'Δ contribution ({unit_txt_axis})', fontsize=7)
                else:
                    ax_in.set_xlabel('Δ contribution', fontsize=7)
                # Connectors only if the dot exists on the plot (no square marker)
                if has_dot:
                    from matplotlib.patches import ConnectionPatch
                    for y_off in [0.15, 0.85]:
                        conn = ConnectionPatch(
                            xyA=(x_dot, y_dot), coordsA=ax_right.transData,
                            xyB=(x0, y0 + y_off*h), coordsB=fig.transFigure,
                            linestyle=(0, (2, 2)), color='0.4', lw=0.8, zorder=6, clip_on=False
                        )
                        fig.add_artist(conn)
                # Group headers at top-left inside each section
                try:
                    n_removed = len(rem_sel)
                    n_added = len(add_sel)
                    n_total = len(seq)
                    ax_in.set_ylim(-0.5, max(n_total - 0.5, 0.5))
                    if n_total > 0:
                        # Convert data-space y to axes fraction so labels are guaranteed inside
                        if n_removed > 0:
                            y_removed_top = max(-0.40, min(n_removed - 0.60, n_total - 0.60))
                            yfrac_removed = (y_removed_top + 0.5) / max(n_total, 1)
                            ax_in.text(0.01, yfrac_removed, 'Removed', transform=ax_in.transAxes, ha='left', va='bottom', fontsize=7, fontweight='normal', color='0.35')
                        if n_added > 0:
                            y_added_top = n_total - 0.60
                            yfrac_added = (y_added_top + 0.5) / max(n_total, 1)
                            ax_in.text(0.01, yfrac_added, 'Added', transform=ax_in.transAxes, ha='left', va='bottom', fontsize=7, fontweight='normal', color='0.35')
                except Exception:
                    pass
                # Net delta absolute and percent vs real total
                try:
                    real_total_vit = float(((np.array(real_meal_vec, dtype=float) / 100.0) @ W_matrix[:, vit_idx]))
                except Exception:
                    real_total_vit = float('nan')
                net_val = float(np.nansum(vals))
                unit_txt = _nut_label[_nut_label.rfind('(')+1:_nut_label.rfind(')')] if '(' in _nut_label and ')' in _nut_label else ''
                if np.isfinite(real_total_vit) and real_total_vit != 0:
                    pct = 100.0 * net_val / real_total_vit
                    ax_in.text(0.98, 0.02, f"Net Δ: {net_val:+.2f} {unit_txt} ({pct:+.1f}%)", transform=ax_in.transAxes, ha='right', va='bottom', fontsize=7)
                else:
                    ax_in.text(0.98, 0.02, f"Net Δ: {net_val:+.2f} {unit_txt}", transform=ax_in.transAxes, ha='right', va='bottom', fontsize=7)
    except Exception:
        pass

    # Second inset: per-food contributions for Cholesterol – bottom-right
    try:
        if (real_meal_vec is not None) and (gen_right_meal_vec is not None) and (W_matrix is not None) and feature_names:
            mufa_key = 'cholesterol_mg'
            # Resolve index using full nutrient list when available (so inset renders even if filtered)
            if (nutrient_names_full is not None) and (mufa_key in nutrient_names_full):
                mufa_idx = nutrient_names_full.index(mufa_key)
            elif mufa_key in names:
                mufa_idx = names.index(mufa_key)
            else:
                mufa_idx = None
            if mufa_idx is not None:
                # Anchor to dot only if Cholesterol is present in the plotted list
                has_dot2 = mufa_key in names_sorted
                if has_dot2:
                    mufa_sorted_idx = names_sorted.index(mufa_key)
                    x_dot2 = float(r_right[mufa_sorted_idx])
                    y_dot2 = float(y[mufa_sorted_idx])
                present_mask2 = (np.array(real_meal_vec) > 0) | (np.array(gen_right_meal_vec) > 0)
                present_idx2 = np.where(present_mask2)[0]
                if (added_codes_right is not None) or (removed_codes_right is not None):
                    changed_codes2 = set(added_codes_right or set()) | set(removed_codes_right or set())
                    changed_idx2 = np.array([i for i, c in enumerate(feature_names) if c in changed_codes2], dtype=int)
                    present_idx2 = np.array([i for i in present_idx2 if i in set(changed_idx2)], dtype=int)
                    if present_idx2.size == 0:
                        present_idx2 = np.where(present_mask2)[0]
                w_col2 = np.array([W_matrix[j, mufa_idx] for j in present_idx2], dtype=float)
                real_contrib2 = (np.array(real_meal_vec, dtype=float)[present_idx2] / 100.0) * w_col2
                gen_contrib2 = (np.array(gen_right_meal_vec, dtype=float)[present_idx2] / 100.0) * w_col2
                delta2 = gen_contrib2 - real_contrib2
                # Split into removed vs added groups and rank within each group
                try:
                    added_codes2_set = set(added_codes_right or set())
                    removed_codes2_set = set(removed_codes_right or set())
                except Exception:
                    added_codes2_set, removed_codes2_set = set(), set()
                codes_all2 = [feature_names[i] for i in present_idx2]
                vals_all2 = [delta2[k] for k in range(len(present_idx2))]
                removed_items2 = [
                    (c, (code2name.get(c, c) if code2name else c), float(v))
                    for c, v in zip(codes_all2, vals_all2) if c in removed_codes2_set
                ]
                added_items2 = [
                    (c, (code2name.get(c, c) if code2name else c), float(v))
                    for c, v in zip(codes_all2, vals_all2) if c in added_codes2_set
                ]
                removed_items2.sort(key=lambda x: -abs(x[2]))
                added_items2.sort(key=lambda x: -abs(x[2]))
                max_total2 = 6
                target2 = 3
                rem_sel2 = removed_items2[:target2]
                add_sel2 = added_items2[:target2]
                remaining2 = max_total2 - (len(rem_sel2) + len(add_sel2))
                if remaining2 > 0:
                    extra2 = (removed_items2[target2:] + added_items2[target2:])
                    extra2.sort(key=lambda x: -abs(x[2]))
                    for (c_e, l_e, v_e) in extra2[:remaining2]:
                        if c_e in removed_codes2_set:
                            rem_sel2.append((c_e, l_e, v_e))
                        else:
                            add_sel2.append((c_e, l_e, v_e))
                seq2 = rem_sel2 + add_sel2
                codes_idx2 = [c for (c, _, _) in seq2]
                labels2 = [lbl for (_, lbl, _) in seq2]
                vals2 = np.array([v for (_, _, v) in seq2], dtype=float)
                colors2 = np.where(vals2 >= 0, '#c0392b', '#27ae60')  # increase = red; decrease = green
                fig2 = ax_right.figure
                # Place bottom-right; make slightly narrower
                w2, h2 = 0.15, 0.17
                x0b = 0.98 - w2
                y0b = 0.06
                ax_in2 = fig2.add_axes([x0b, y0b, w2, h2])
                # Lollipop plot
                yy2 = np.arange(len(vals2))
                for i, v in enumerate(vals2):
                    x_start = 0.0 if v >= 0 else v
                    x_end = v if v >= 0 else 0.0
                    ax_in2.hlines(y=yy2[i], xmin=x_start, xmax=x_end, colors=colors2[i], lw=1.6)
                    ax_in2.plot([v], [yy2[i]], marker='o', markersize=4.5, color=colors2[i])
                ax_in2.set_yticks(np.arange(len(vals2)))
                def _short2(s: str) -> str:
                    return s if len(s) <= 26 else (s[:23] + '…')
                ax_in2.set_yticklabels([_short2(s) for s in labels2], fontsize=7)
                # Divider between removed and added groups
                try:
                    n_removed2 = len(rem_sel2)
                    n_total2 = len(seq2)
                    if n_removed2 > 0 and n_removed2 < n_total2:
                        ax_in2.set_ylim(-0.5, n_total2 - 0.5)
                        ax_in2.axhline(n_removed2 - 0.5, color='0.7', lw=0.8)
                except Exception:
                    pass
                # Bold substituted items
                try:
                    added_codes2 = set([feature_names[j] for j in np.where((np.array(real_meal_vec)<=0) & (np.array(gen_right_meal_vec)>0))[0]])
                    removed_codes2 = set([feature_names[j] for j in np.where((np.array(real_meal_vec)>0) & (np.array(gen_right_meal_vec)<=0))[0]])
                    for tick, code in zip(ax_in2.get_yticklabels(), codes_idx2):
                        if (code in added_codes2) or (code in removed_codes2):
                            tick.set_fontweight('bold')
                except Exception:
                    pass
                ax_in2.yaxis.tick_right()
                ax_in2.tick_params(axis='y', labelright=True, labelleft=False)
                ax_in2.axvline(0.0, color='0.6', lw=0.8, ls='--')
                ax_in2.tick_params(axis='x', labelsize=7)
                ax_in2.set_xlabel('Δ contribution (mg)', fontsize=7)
                try:
                    _nut_label2 = NUTRIENT_LABELS.get(mufa_key, 'Cholesterol')
                except Exception:
                    _nut_label2 = 'MUFA'
                # Simplify title: remove units from label
                try:
                    base_label2 = _nut_label2.split(' (')[0]
                except Exception:
                    base_label2 = _nut_label2
                # Reinstate inset title (contributors)
                ax_in2.set_title(f'{base_label2} contributors', fontsize=8, pad=2)
                # Box the Cholesterol dot and connect with dotted lines only if it exists on the plot
                if has_dot2:
                    from matplotlib.patches import ConnectionPatch as _CP
                    for y_off in [0.15, 0.85]:
                        conn2 = _CP(
                            xyA=(x_dot2, y_dot2), coordsA=ax_right.transData,
                            xyB=(x0b, y0b + y_off*h2), coordsB=fig2.transFigure,
                            linestyle=(0, (2, 2)), color='0.4', lw=0.8, zorder=6, clip_on=False
                        )
                        fig2.add_artist(conn2)
                # Group headers and net summary for Cholesterol (% vs real; units=mg)
                try:
                    n_removed2 = len(rem_sel2)
                    n_added2 = len(add_sel2)
                    n_total2 = len(seq2)
                    ax_in2.set_ylim(-0.5, max(n_total2 - 0.5, 0.5))
                    if n_total2 > 0:
                        # Convert to axes fraction to keep headers inside inset bounds
                        if n_removed2 > 0:
                            y_removed_top2 = max(-0.40, min(n_removed2 - 0.60, n_total2 - 0.60))
                            yfrac_removed2 = (y_removed_top2 + 0.5) / max(n_total2, 1)
                            ax_in2.text(0.01, yfrac_removed2, 'Removed', transform=ax_in2.transAxes, ha='left', va='bottom', fontsize=7, fontweight='normal', color='0.35')
                        if n_added2 > 0:
                            y_added_top2 = n_total2 - 0.60
                            yfrac_added2 = (y_added_top2 + 0.5) / max(n_total2, 1)
                            ax_in2.text(0.01, yfrac_added2, 'Added', transform=ax_in2.transAxes, ha='left', va='bottom', fontsize=7, fontweight='normal', color='0.35')
                except Exception:
                    pass
                net_mufa = float(np.nansum(vals2))
                try:
                    real_total_mufa = float(((np.array(real_meal_vec, dtype=float) / 100.0) @ W_matrix[:, mufa_idx]))
                except Exception:
                    real_total_mufa = float('nan')
                if np.isfinite(real_total_mufa) and real_total_mufa != 0:
                    pct2 = 100.0 * net_mufa / real_total_mufa
                    ax_in2.text(0.98, 0.02, f"Net Δ: {net_mufa:+.2f} mg ({pct2:+.1f}%)", transform=ax_in2.transAxes, ha='right', va='bottom', fontsize=7)
                else:
                    ax_in2.text(0.98, 0.02, f"Net Δ: {net_mufa:+.2f} mg", transform=ax_in2.transAxes, ha='right', va='bottom', fontsize=7)
    except Exception:
        pass


def _knee_matches_csvs(sweep_dir: Path) -> List[str]:
    """substitution_matches.csv for k=1 and k=2, each at its knee theta (written by `make theta_sweep`)."""
    theta_by_k = json.loads((sweep_dir / 'figures_theta_sweep' / 'optimal_theta_by_k.json').read_text())
    out = []
    for k in (1, 2):
        tag = f"{float(theta_by_k[str(k)]):g}".replace('.', 'p')
        out.append(str(sweep_dir / f'tradeoff_theta_{tag}_k{k}' / 'substitution_matches.csv'))
    return out


def main():
    p = argparse.ArgumentParser(description='Plot per-nutrient log10 fold-changes for selected substitutions')
    p.add_argument('--matches_csv', type=str, action='append', default=None, help='One or more substitution_matches.csv files (repeatable)')
    p.add_argument('--sweep_dir', type=str, default=None,
                   help="Theta-sweep folder (e.g. outputs/runs/eval_run/substitutions_k3); used instead of "
                        "--matches_csv, it takes the k=1 and k=2 matches at each k's knee theta")
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--outdir', type=str, default='outputs/figure4a')
    p.add_argument('--n_pairs', type=int, default=2)
    p.add_argument('--pair', action='append', default=None, help='Format MealType:gen_idx,real_idx (repeatable)')
    p.add_argument('--base', type=str, default=None, help='Force base MealType:real_idx and pick k=1 & k=2 for it')
    p.add_argument('--require_same_base', action='store_true', default=False, help='Require k=1 and k=2 share exact base composition minus replacements')
    p.add_argument('--top_n', type=int, default=None)
    p.add_argument('--eps', type=float, default=1e-8)
    p.add_argument('--extras_file', type=str, default=None, help='Optional text file with extra items to append in meal compositions (Original/k=1/k=2)')
    args = p.parse_args()
    if not args.matches_csv:
        if not args.sweep_dir:
            p.error('give --matches_csv or --sweep_dir')
        args.matches_csv = _knee_matches_csvs(Path(args.sweep_dir))
        print('Using matches:', ', '.join(args.matches_csv))

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    data = load_data(Path(args.data_dir))
    pairs: List[Tuple[str, int, int, Dict]] = []
    matches_list = [Path(x) for x in args.matches_csv]
    if args.pair:
        for spec in args.pair:
            try:
                mt, rest = spec.split(':', 1)
                gi, ri = rest.split(',', 1)
                pairs.append((mt, int(gi), int(ri), {}))
            except Exception:
                continue
    else:
        # A k=1 and a k=2 substitution of the same real meal (with the default n_pairs=2);
        # select_rank=2 draws the second-highest-scoring candidate when there are several
        same_base = None
        if args.n_pairs >= 2:
            same_base, all_candidates = select_k1_k2_same_base(matches_list, data, require_same_base=bool(args.require_same_base), base_spec=args.base, print_top_k=5, select_rank=2)
            # Export top-50 candidate meals with portions
            try:
                export_top_candidates(all_candidates, data, outdir, top_k=50)
            except Exception:
                pass
        if same_base is not None:
            pairs = same_base
        else:
            pairs = select_pairs(matches_list, n_pairs=args.n_pairs)

    # Pairs given with --pair have no metadata: take ΔRDI, ΔCost and the swapped foods from the matches
    try:
        # Use the union of matches files to enrich metadata
        import pandas as _pd
        df_meta = _pd.concat([_pd.read_csv(p) for p in matches_list], ignore_index=True)
        # Normalize keys for reliable lookup
        if 'k_substitution' in df_meta.columns:
            df_meta['k_substitution'] = pd.to_numeric(df_meta['k_substitution'], errors='coerce')
        if 'added_foods' in df_meta.columns:
            df_meta['added_foods'] = df_meta['added_foods'].fillna('')
        if 'removed_foods' in df_meta.columns:
            df_meta['removed_foods'] = df_meta['removed_foods'].fillna('')
        key_cols = ['Meal Type', 'gen_idx', 'real_idx']
        have_cols = [c for c in key_cols if c in df_meta.columns]
        if len(have_cols) == 3 and len(pairs) > 0:
            new_pairs = []
            for (mt, gi, ri, meta) in pairs:
                if not meta:
                    m = df_meta[(df_meta['Meal Type'] == mt) & (df_meta['gen_idx'] == gi) & (df_meta['real_idx'] == ri)]
                    if not m.empty:
                        row = m.iloc[0].to_dict()
                        meta = {
                            'Meal Type': row.get('Meal Type', mt),
                            'gen_idx': int(row.get('gen_idx', gi)),
                            'real_idx': int(row.get('real_idx', ri)),
                            'improvement': float(row.get('improvement', np.nan)),
                            'delta_cost': float(row.get('delta_cost', np.nan)),
                            'real_deviation': float(row.get('real_deviation', np.nan)) if 'real_deviation' in row else np.nan,
                            'k_substitution': int(row.get('k_substitution', np.nan)) if not pd.isna(row.get('k_substitution', np.nan)) else None,
                            'added_foods': row.get('added_foods', ''),
                            'removed_foods': row.get('removed_foods', ''),
                        }
                        # If k_substitution is still missing, compute from actual vectors
                        if meta.get('k_substitution') is None:
                            meta = {**meta, **compute_pair_meta(data, mt, gi, ri)}
                # Ensure we at least have k_substitution/added/removed
                if 'k_substitution' not in meta or meta.get('k_substitution') in [None, 0]:
                    meta = {**meta, **compute_pair_meta(data, mt, gi, ri)}
                new_pairs.append((mt, gi, ri, meta))
            pairs = new_pairs
    except Exception:
        pass

    # Also run secondary candidate analysis: vitamins up, MUFA/SFA down
    try:
        export_vitamins_up_fats_down(matches_list, data, outdir)
    except Exception:
        pass

    # Build figure
    if len(pairs) == 0:
        print('No pairs to plot.')
        return
    if len(pairs) >= 2:
        # Use butterfly layout with a single nutrients list in the middle
        pairs2 = pairs[:2]
        # Compute both
        lfc_values = []
        titles = []
        for i, (mt, gi, ri, meta) in enumerate(pairs2):
            gen_meal = data['gen_by_type'][mt][gi]
            real_meal = data['real_by_type'][mt][ri]
            n_gen, n_real = compute_nutrients(gen_meal, real_meal, data['W'])
            lfc = log10_fold_change(n_gen, n_real, eps=float(args.eps))
            lfc_values.append(lfc)
            titles.append(panel_title(meta, i, data.get('code2name', {}), data['feature_names'], gen_meal, real_meal, W=data['W'], RDI=data.get('RDI', None)))
        # RDI vector for filtering must be available before applying mask
        rdi = _align_rdi_units_with_names(data.get('RDI', None), data['nutrient_names'])
        names_all = list(data['nutrient_names'])
        # Filter out nutrients without an RDI target (e.g., sugars, cholesterol)
        rdi_filtered = rdi
        rdi_mask = None
        try:
            if rdi is not None:
                rdi_arr = np.asarray(rdi)
                rdi_mask = np.isfinite(rdi_arr) & (rdi_arr > 0)
                if rdi_mask.any():
                    lfc_values = [v[rdi_mask] for v in lfc_values]
                    names_all = [names_all[i] for i, keep in enumerate(rdi_mask) if keep]
                    rdi_filtered = rdi_arr[rdi_mask]
        except Exception:
            rdi_filtered = rdi
            rdi_mask = None
        # Optional top_n by max magnitude across both (after filtering)
        if args.top_n is not None and args.top_n > 0:
            mag = np.maximum(np.abs(lfc_values[0]), np.abs(lfc_values[1]))
            order = np.argsort(-mag)[:args.top_n]
            lfc_values = [v[order] for v in lfc_values]
            names = [names_all[j] for j in order]
        else:
            names = names_all
        # Layout 1x3: k=1 ratios | nutrient names | k=2 ratios
        from matplotlib.gridspec import GridSpec
        # Make figure wider to accommodate an external inset without overlap
        fig = plt.figure(figsize=(17, 9))
        # Single row layout (no bullet overlays in this figure)
        gs = GridSpec(1, 3, width_ratios=[2.4, 0.6, 2.4], wspace=0.0005)
        ax_left = fig.add_subplot(gs[0, 0])
        # Do not share y with middle axis, to ensure its labels render as centered text
        ax_mid = fig.add_subplot(gs[0, 1])
        # Share y only between left and right for alignment
        ax_right = fig.add_subplot(gs[0, 2], sharey=ax_left)
        # Real amounts for indicator computation (use real meal of base for each side)
        # Enforce left=k1 and right=k2 ordering when two pairs are provided (e.g., via --pair)
        try:
            mt_l, gi_l, ri_l, meta_l = pairs2[0]
            mt_r, gi_r, ri_r, meta_r = pairs2[1]
        except Exception:
            mt_l, gi_l, ri_l, meta_l = pairs2[0][0], pairs2[0][1], pairs2[0][2], (pairs2[0][3] if len(pairs2[0])>3 else {})
            mt_r, gi_r, ri_r, meta_r = pairs2[1][0], pairs2[1][1], pairs2[1][2], (pairs2[1][3] if len(pairs2[1])>3 else {})
        # Compute metadata if missing
        try:
            meta_l = {**(meta_l or {}), **compute_pair_meta(data, mt_l, gi_l, ri_l)}
        except Exception:
            meta_l = meta_l or {}
        try:
            meta_r = {**(meta_r or {}), **compute_pair_meta(data, mt_r, gi_r, ri_r)}
        except Exception:
            meta_r = meta_r or {}
        try:
            k_l = int(meta_l.get('k_substitution', 0))
            k_r = int(meta_r.get('k_substitution', 0))
            if not (k_l == 1 and k_r == 2) and (k_r == 1 and k_l == 2):
                # swap to make left=k1, right=k2
                (mt_l, gi_l, ri_l, meta_l), (mt_r, gi_r, ri_r, meta_r) = (mt_r, gi_r, ri_r, meta_r), (mt_l, gi_l, ri_l, meta_l)
        except Exception:
            pass

        real_meal = data['real_by_type'][mt_l][ri_l]
        real_meal_right = data['real_by_type'][mt_r][ri_r]
        real_nutrients = (real_meal / 100.0) @ data['W']
        # Generated nutrients for both panels (for toward-target coloring)
        gen_left = data['gen_by_type'][mt_l][gi_l]
        gen_right = data['gen_by_type'][mt_r][gi_r]
        gen_left_n = (gen_left / 100.0) @ data['W']
        gen_right_n = (gen_right / 100.0) @ data['W']
        # Apply the same RDI mask to real/gen nutrient vectors for plotting alignment
        try:
            if (rdi_mask is not None) and (rdi_mask.any()):
                real_nutrients = real_nutrients[rdi_mask]
                gen_left_n = gen_left_n[rdi_mask]
                gen_right_n = gen_right_n[rdi_mask]
        except Exception:
            pass
        # Compute changes strictly from right-panel meal vectors (k=2); no fallback
        real_codes = {data['feature_names'][i] for i, g in enumerate(real_meal_right) if float(g) > 0}
        gen_codes = {data['feature_names'][i] for i, g in enumerate(gen_right) if float(g) > 0}
        added_codes_right = set(gen_codes - real_codes)
        removed_codes_right = set(real_codes - gen_codes)
        # Meal targets use the dinner energy share (0.40) whatever the meal type
        plot_butterfly_two(
            ax_left, ax_mid, ax_right,
            names, lfc_values[0], lfc_values[1], 2.0, titles[0], titles[1],
            rdi_targets=rdi_filtered, meal_fraction=0.40, real_amounts=real_nutrients,
            gen_left_amounts=gen_left_n, gen_right_amounts=gen_right_n,
            real_meal_vec=real_meal_right, gen_right_meal_vec=gen_right,
            W_matrix=data['W'], feature_names=data['feature_names'],
            code2name=data.get('code2name', {}), inset_nutrient_key='vitamin_b12_mcg', nutrient_names_full=data['nutrient_names'],
            added_codes_right=added_codes_right, removed_codes_right=removed_codes_right
        )
        # Remove vertical grids and vertical spines; keep a clean look
        for ax in (ax_left, ax_right):
            ax.grid(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # ticks on top and bottom; face inward towards the dots
            ax.tick_params(axis='x', which='both', direction='in', top=True, bottom=True, labeltop=False, length=4)
        # Middle axis: no grid, keep labels readable
        ax_mid.set_xticks([])
        ax_mid.grid(False)
        fig_fp_svg = outdir / 'figure_substitution_foldchange_two_panels.svg'
        fig_fp_png = outdir / 'figure_substitution_foldchange_two_panels.png'
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
        fig.savefig(fig_fp_svg, bbox_inches='tight')
        fig.savefig(fig_fp_png, dpi=300, bbox_inches='tight')
        print(f'Saved figure to: {fig_fp_svg} and {fig_fp_png}')
    else:
        # Single pair fallback (classic single column)
        (mt, gi, ri, meta) = pairs[0]
        gen_meal = data['gen_by_type'][mt][gi]
        real_meal = data['real_by_type'][mt][ri]
        n_gen, n_real = compute_nutrients(gen_meal, real_meal, data['W'])
        lfc = log10_fold_change(n_gen, n_real, eps=float(args.eps))
        if args.top_n is not None and args.top_n > 0:
            order = np.argsort(-np.abs(lfc))[:args.top_n]
            lfc = lfc[order]
            names = [data['nutrient_names'][j] for j in order]
        else:
            names = list(data['nutrient_names'])
        fig, ax = plt.subplots(1, 1, figsize=(6, 10))
        title = panel_title(meta, 0, data.get('code2name', {}), data['feature_names'], gen_meal, real_meal)
        plot_pair(ax, names, lfc, highlight_thresh=2.0, title=title)
        fig_fp_svg = outdir / 'figure_substitution_foldchange_single.svg'
        fig_fp_png = outdir / 'figure_substitution_foldchange_single.png'
        fig.tight_layout()
        fig.savefig(fig_fp_svg, bbox_inches='tight')
        fig.savefig(fig_fp_png, dpi=300, bbox_inches='tight')
        print(f'Saved figure to: {fig_fp_svg} and {fig_fp_png}')

    # Secondary figure: meals summary (original, k=1, k=2) with substitutions highlighted
    try:
        from matplotlib.gridspec import GridSpec
        fig2 = plt.figure(figsize=(12, 4))
        gs2 = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.05)
        axm1 = fig2.add_subplot(gs2[0, 0]); axm1.axis('off')
        axm2 = fig2.add_subplot(gs2[0, 1]); axm2.axis('off')
        axm3 = fig2.add_subplot(gs2[0, 2]); axm3.axis('off')
        # Support both auto-selected and --pair flows by using the current pairs list
        pairs_for_fig = pairs[:2]
        mt = pairs_for_fig[0][0]
        ridx = pairs_for_fig[0][2]
        gidx1 = pairs_for_fig[0][1]
        gidx2 = pairs_for_fig[1][1]
        # Use the actual real meal (original) for the base column
        base_meal_vec = data['real_by_type'][mt][ridx]
        base_items = _meal_to_items(base_meal_vec, data)
        # Ensure k_substitution is 1 and 2 for the two panels; recompute if metadata missing or inconsistent
        meta1 = pairs_for_fig[0][3] or {}
        meta2 = pairs_for_fig[1][3] or {}
        if (meta1.get('k_substitution') not in [1, 2]) or (meta2.get('k_substitution') not in [1, 2]):
            meta1 = {**meta1, **compute_pair_meta(data, mt, gidx1, ridx)}
            meta2 = {**meta2, **compute_pair_meta(data, mt, gidx2, ridx)}
        # Swap order if needed to enforce left=k1, right=k2
        if int(meta1.get('k_substitution', 0)) != 1 or int(meta2.get('k_substitution', 0)) != 2:
            # Try swapping if that fixes it
            meta1_sw = {**meta2}
            meta2_sw = {**meta1}
            if int(meta1_sw.get('k_substitution', 0)) == 1 and int(meta2_sw.get('k_substitution', 0)) == 2:
                gidx1, gidx2 = gidx2, gidx1
                meta1, meta2 = meta1_sw, meta2_sw
        # Original: only base foods (no highlighting)
        draw_food_bullets(axm1, base_items, start_y=0.95, dy=0.06)
        axm1.set_title('Original', fontsize=11, pad=6)
        # k=1: highlight its added foods
        items_k1 = foods_to_show_with_highlights(mt, gidx1, ridx, meta1, data)
        draw_food_bullets(axm2, items_k1, start_y=0.95, dy=0.06)
        axm2.set_title('k=1', fontsize=11, pad=6)
        # k=2: highlight its added foods
        items_k2 = foods_to_show_with_highlights(mt, gidx2, ridx, meta2, data)
        draw_food_bullets(axm3, items_k2, start_y=0.95, dy=0.06)
        axm3.set_title('k=2', fontsize=11, pad=6)
        # Optional: append extra display-only items from a text file (three sections Original/k1/k2)
        try:
            if args.extras_file:
                from pathlib import Path as _P
                fp = _P(args.extras_file)
                if fp.exists():
                    txt = fp.read_text().splitlines()
                    # Expect 3 blocks separated by blank lines
                    blocks = []
                    block = []
                    for line in txt:
                        if line.strip() == '':
                            if block:
                                blocks.append(block); block = []
                        else:
                            block.append(line.strip())
                    if block:
                        blocks.append(block)
                    while len(blocks) < 3:
                        blocks.append([])
                    def _as_items(lines):
                        out = []
                        for s in lines:
                            # Allow lines like "Side salad (1 cup)" or "Water (12 fl oz)"
                            out.append((s, 0.0, False))
                        return out
                    draw_food_bullets(axm1, _as_items(blocks[0]), start_y=0.27, dy=0.06)
                    draw_food_bullets(axm2, _as_items(blocks[1]), start_y=0.27, dy=0.06)
                    draw_food_bullets(axm3, _as_items(blocks[2]), start_y=0.27, dy=0.06)
        except Exception:
            pass
        fig2.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
        fig2_fp_svg = outdir / 'figure_meal_compositions.svg'
        fig2_fp_png = outdir / 'figure_meal_compositions.png'
        fig2.savefig(fig2_fp_svg, bbox_inches='tight')
        fig2.savefig(fig2_fp_png, dpi=300, bbox_inches='tight')
        print(f'Saved meal compositions to: {fig2_fp_svg} and {fig2_fp_png}')

        # Also emit a fixed hardcoded version with display-only extras baked in
        try:
            fig3 = plt.figure(figsize=(12, 4))
            gs3 = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.05)
            fx1 = fig3.add_subplot(gs3[0, 0]); fx1.axis('off')
            fx2 = fig3.add_subplot(gs3[0, 1]); fx2.axis('off')
            fx3 = fig3.add_subplot(gs3[0, 2]); fx3.axis('off')
            # Re-draw the base and k items
            # Beautify names and omit grams for readability
            # Determine removed foods from the original relative to k=2 (for red highlighting)
            removed_codes = set(_parse_food_list(meta2.get('removed_foods', ''))) if isinstance(meta2, dict) else set()
            # Map codes to the same friendly labels used in bullets to ensure exact matches
            removed_for_red = set()
            for code in removed_codes:
                nm = _clean_name_display(code, data['code2name'])
                friendly = _friendly_portion(nm, 0.0)
                # strip any portion suffix
                friendly_base = friendly.split(' (')[0].strip()
                removed_for_red.add(friendly_base)
            # Every food of the three meals, with portions computed from its grams
            base_disp = _beautify_items(base_items)
            k1_disp = _beautify_items(items_k1)
            k2_disp = _beautify_items(items_k2)

            draw_food_bullets(fx1, base_disp, start_y=0.95, dy=0.06, show_grams=False, red_names=removed_for_red)
            draw_food_bullets(fx2, k1_disp, start_y=0.95, dy=0.06, show_grams=False)
            draw_food_bullets(fx3, k2_disp, start_y=0.95, dy=0.06, show_grams=False)
            fx1.set_title('Original Meal', fontsize=11, pad=6)
            fx2.set_title('1-hop substitution', fontsize=11, pad=6)
            fx3.set_title('2-hop substitution', fontsize=11, pad=6)
            # Illustrative side items listed in every column (display only: they are not part
            # of the computed meals, their costs or ΔRDI)
            def _mk_fixed(lines):
                return [(s, 0.0, False) for s in lines]
            extras_common = _mk_fixed(['Side salad (1 cup)', 'Fruit cup (1 cup)', 'Iced tea (12 fl oz)'])
            # Place extras closer to the real items
            draw_food_bullets(fx1, extras_common, start_y=0.55, dy=0.055, show_grams=False)
            draw_food_bullets(fx2, extras_common, start_y=0.55, dy=0.055, show_grams=False)
            draw_food_bullets(fx3, extras_common, start_y=0.55, dy=0.055, show_grams=False)
            # Add estimated costs and ΔRDI% text at the bottom of each column
            # ΔRDI% from the meal vectors, used when the matches row has no real_deviation
            def _delta_rdi_percent(mt_local: str, gi_local: int, ri_local: int) -> float:
                real_v = data['real_by_type'][mt_local][ri_local]
                gen_v = data['gen_by_type'][mt_local][gi_local]
                return compute_delta_rdi_percent_from_vectors(real_v, gen_v, data['W'], data.get('RDI', None))

            # Always use restaurant-style portion pricing for display costs
            cost_orig = _estimate_cost_from_vec(base_meal_vec, data)
            cost_k1 = _estimate_cost_from_vec(data['gen_by_type'][mt][gidx1], data)
            cost_k2 = _estimate_cost_from_vec(data['gen_by_type'][mt][gidx2], data)
            # If all missing, fallback to heuristic buckets
            if not np.isfinite(cost_orig) or not np.isfinite(cost_k1) or not np.isfinite(cost_k2):
                def _heuristic_cost(items: List[Tuple[str, float, bool]]) -> float:
                    cost = 0.0
                    for name, _, _ in items:
                        n = name.lower()
                        if 'chicken' in n or 'steak' in n or 'beef hot dog' in n:
                            cost += 6.0
                        elif 'rice' in n:
                            cost += 2.5
                        elif 'yogurt' in n:
                            cost += 2.0
                        elif 'cherries' in n:
                            cost += 3.0
                        elif 'peas & carrots' in n or 'peas and carrots' in n:
                            cost += 2.5
                        elif 'beans' in n:
                            cost += 3.0
                        elif 'cabbage' in n:
                            cost += 2.0
                        elif 'fruit' in n or 'juice' in n:
                            cost += 1.5
                        elif 'mayonnaise' in n or 'ketchup' in n:
                            cost += 0.5
                        else:
                            cost += 1.0
                    return cost
                if not np.isfinite(cost_orig):
                    cost_orig = _heuristic_cost(_beautify_items(base_items) + extras_common)
                if not np.isfinite(cost_k1):
                    cost_k1 = _heuristic_cost(_beautify_items(items_k1) + extras_common)
                if not np.isfinite(cost_k2):
                    cost_k2 = _heuristic_cost(_beautify_items(items_k2) + extras_common)
            def _meta_to_percent(meta_local: Dict, gi_local: int) -> float:
                imp = float(meta_local.get('improvement', np.nan))
                rdev = float(meta_local.get('real_deviation', np.nan))
                if np.isfinite(imp) and np.isfinite(rdev) and rdev != 0:
                    return (imp / rdev) * 100.0
                return _delta_rdi_percent(mt, gi_local, ridx)
            dRDI_k1 = _meta_to_percent(meta1, gidx1)
            dRDI_k2 = _meta_to_percent(meta2, gidx2)

            def _footer(ax, cost_val: float, dRDI_val: float, cost_change_text: str = ''):
                ax.text(
                    0.0, 0.03,
                    f"Cost: ${cost_val:.2f}{cost_change_text}\nΔRDI: {dRDI_val:+.1f}%",
                    transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=9,
                    clip_on=False,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.0)
                )

            # Original column shows baseline; ΔRDI = 0.0%
            _footer(fx1, cost_orig, 0.0, '')
            # Cost change relative to original
            def _chg_txt(c_new: float, c_base: float) -> str:
                if c_base and c_base > 0:
                    pct = (c_new - c_base) / c_base * 100.0
                    return f" ({pct:+.1f}%)"
                return ''
            _footer(fx2, cost_k1, dRDI_k1, _chg_txt(cost_k1, cost_orig))
            _footer(fx3, cost_k2, dRDI_k2, _chg_txt(cost_k2, cost_orig))
            fig3.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
            f3_svg = outdir / 'fixed_meal_composition.svg'
            f3_png = outdir / 'fixed_meal_composition.png'
            fig3.savefig(f3_svg, bbox_inches='tight')
            fig3.savefig(f3_png, dpi=300, bbox_inches='tight')
            print(f'Saved fixed meal composition to: {f3_svg} and {f3_png}')
        except Exception:
            pass
    except Exception:
        pass


if __name__ == '__main__':
    main()


