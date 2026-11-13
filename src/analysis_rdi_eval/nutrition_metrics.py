"""
Nutrition metrics computation for meal evaluation.

This module provides functions to compute various nutrition-related metrics
including NAR/MAR, MER, AMDR compliance, and dietary diversity scores. The Figure 3
spider plots (src/pipeline_eval/visualization.py) are built from these functions.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import warnings


def _canon_map(names: List[str]) -> Dict[str, int]:
    """
    Map canonical nutrient names to indices.
    
    Since all data is already harmonized to canonical names via nutrition_harmonizer.py,
    we can use direct mapping without synonyms.
    """
    # Direct mapping from canonical names to indices
    return {name: i for i, name in enumerate(names)}


def grams_to_nutrients(Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Convert grams to nutrients using the nutrient matrix.
    
    Args:
        Y: (N, D) grams array
        W: (D, M) nutrient-per-100g matrix
        
    Returns:
        (N, M) nutrients array
    """
    return (Y / 100.0) @ W


def kcal_from_nutrients(Nmat: np.ndarray, nutrient_names: List[str],
                       kcal_spec: Dict[str, float] = None) -> np.ndarray:
    """
    Compute energy from nutrients with a per-row fallback:
    use direct 'energy_kcal' if present and >0 on that row; otherwise compute from macros.
    """
    if kcal_spec is None:
        kcal_spec = {"protein_g": 4, "carb_g": 4, "fat_g": 9}

    cmap = _canon_map(nutrient_names)
    N = Nmat.shape[0]
    kcal = np.zeros(N, dtype=float)

    have_direct = "energy_kcal" in cmap
    direct = Nmat[:, cmap["energy_kcal"]] if have_direct else None

    # rows that can use direct kcal
    if have_direct:
        use_direct = np.isfinite(direct) & (direct > 0)
        kcal[use_direct] = direct[use_direct]
    else:
        use_direct = np.zeros(N, dtype=bool)

    # fallback rows → compute from macros
    fallback = ~use_direct
    if np.any(fallback):
        for k, factor in kcal_spec.items():
            if k in cmap:
                kcal[fallback] += Nmat[fallback, cmap[k]] * factor

    if not np.any(kcal > 0):
        warnings.warn("kcal_from_nutrients: could not compute kcal; returning zeros.")

    return kcal


def macro_shares_percent(Nmat: np.ndarray, nutrient_names: List[str], 
                        kcal: np.ndarray) -> np.ndarray:
    """
    Compute macro nutrient percentages of total energy.
    
    Args:
        Nmat: (N, M) nutrients array
        nutrient_names: List of nutrient names
        kcal: (N,) total kcal array
        
    Returns:
        (N, 3) array with [Protein%, Fat%, Carb%]
    """
    cmap = _canon_map(nutrient_names)
    triples = [("protein_g", 4), ("fat_g", 9), ("carb_g", 4)]
    out = np.zeros((Nmat.shape[0], 3), dtype=float)
    
    for i, (name, factor) in enumerate(triples):
        if name in cmap:
            out[:, i] = 100.0 * (Nmat[:, cmap[name]] * factor) / np.maximum(kcal, 1e-6)
    
    return out


def amdr_within(macro_pct: np.ndarray, 
               ranges: Dict[str, List[float]] = None) -> np.ndarray:
    """
    Check if macro percentages are within AMDR ranges.
    
    Args:
        macro_pct: (N, 3) array with [Protein%, Fat%, Carb%]
        ranges: Dict with macro ranges, default DRI AMDR
        
    Returns:
        (N,) boolean array
    """
    if ranges is None:
        ranges = {"Protein": [10, 35], "Fat": [20, 35], "Carb": [45, 65]}
    
    within = np.ones(macro_pct.shape[0], dtype=bool)
    
    for i, (macro, (lo, hi)) in enumerate(ranges.items()):
        if i < macro_pct.shape[1]:
            pct = macro_pct[:, i]
            within &= (pct >= lo) & (pct <= hi)
    
    return within


def compute_nar_mar(Nmat: np.ndarray, RDI: np.ndarray,
                   meal_fraction: Union[float, np.ndarray],
                   adequacy_nutrients: List[str] = None,
                   nutrient_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Nutrient Adequacy Ratio (NAR) and Mean Adequacy Ratio (MAR).
    
    Args:
        Nmat: (N, M) nutrients array
        RDI: (M,) reference daily intakes
        meal_fraction: scalar or (M,) meal fraction of daily RDI
        adequacy_nutrients: List of nutrient names to include in MAR
        
    Returns:
        NAR: (N, M) nutrient adequacy ratios
        MAR: (N,) mean adequacy ratios
    """
    # Convert meal_fraction to array if scalar
    if np.isscalar(meal_fraction):
        meal_fraction = np.full(len(RDI), float(meal_fraction))
    
    # Target intakes for this meal
    targets = meal_fraction * RDI
    
    # Compute NAR for all nutrients
    NAR = np.minimum(1.0, Nmat / np.maximum(targets, 1e-6))
    
    # Compute MAR for selected nutrients
    if adequacy_nutrients and nutrient_names is not None:
        cmap = _canon_map(nutrient_names)
        sel = [cmap[n.lower()] for n in adequacy_nutrients if n and (n.lower() in cmap)]
        if not sel:
            warnings.warn("No adequacy nutrients matched; MAR over all nutrients.")
            MAR = np.mean(NAR, axis=1)
        else:
            MAR = np.mean(NAR[:, sel], axis=1)
    else:
        MAR = np.mean(NAR, axis=1)
    
    return NAR, MAR


def compute_mer(Nmat: np.ndarray, limits: Dict[str, float],
               nutrient_names: List[str], kcal: np.ndarray,
               meal_fraction: Union[float, np.ndarray] = 1/3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Mean Excess Ratio (MER) for moderation nutrients.
    
    Args:
        Nmat: (N, M) nutrients array
        limits: Dict mapping nutrient names to limits
        nutrient_names: List of nutrient names
        kcal: (N,) total kcal array
        meal_fraction: Meal fraction of daily RDI
        
    Returns:
        ER: (N, len(limits)) excess ratios for each moderation nutrient
        MER: (N,) mean excess ratios
    """
    if not limits:
        return np.zeros((Nmat.shape[0], 0)), np.zeros(Nmat.shape[0])
    
    cmap = _canon_map(nutrient_names)
    key2g = {
        "sfa_pct_kcal": ("sfa_g", 9.0),
        "added_sugars_pct_kcal": ("added_sugars_g", 4.0),
        "sugars_total_pct_kcal": ("sugars_total_g", 4.0),
    }
    N = Nmat.shape[0]
    ER = np.full((N, len(limits)), np.nan, dtype=float)

    if np.isscalar(meal_fraction):
        frac_vec = np.full(N, float(meal_fraction))
    else:
        frac_vec = np.asarray(meal_fraction, float).reshape(-1)
        if frac_vec.size != N:
            frac_vec = np.full(N, float(np.mean(frac_vec)))

    for i, (key, limit) in enumerate(limits.items()):
        k = key.lower()
        if k.endswith("_pct_kcal"):
            grams_name, kpg = key2g.get(k, (k.replace("_pct_kcal", "_g"), 4.0))
            if grams_name in cmap:
                pct = 100.0 * (Nmat[:, cmap[grams_name]] * kpg) / np.maximum(kcal, 1e-6)
                ER[:, i] = np.maximum(1.0, pct / max(limit, 1e-6))
            else:
                warnings.warn(f"MER: grams column for {key} not found; setting ER=NaN.")
                ER[:, i] = np.nan
        else:
            if k in cmap:
                intake = Nmat[:, cmap[k]]
                ER[:, i] = np.maximum(1.0, intake / np.maximum(limit * frac_vec, 1e-6))
            else:
                warnings.warn(f"MER: nutrient {key} not found; setting ER=NaN.")
                ER[:, i] = np.nan

    return ER, np.nanmean(ER, axis=1)


def compute_nutrition_metrics_comprehensive(
    Y: np.ndarray, W: np.ndarray, nutrient_names: List[str],
    RDI: np.ndarray, meal_fraction: Union[float, np.ndarray],
    limits: Dict[str, float] = None, kcal_spec: Dict[str, float] = None,
    food_to_group: Dict[str, str] = None, adequacy_nutrients: List[str] = None,
    feature_names: Optional[List[str]] = None, target_groups: Optional[int] = None
) -> Dict:
    """
    Compute comprehensive nutrition metrics for a set of meals.
    
    Args:
        Y: (N, D) grams array
        W: (D, M) nutrient matrix
        nutrient_names: List of nutrient names
        RDI: (M,) reference daily intakes
        meal_fraction: Meal fraction of daily RDI
        limits: Moderation limits
        kcal_spec: Energy conversion factors
        food_to_group: Food group mapping
        adequacy_nutrients: Nutrients for MAR calculation
        
    Returns:
        Dict with comprehensive metrics
    """
    if limits is None:
        limits = {}
    if kcal_spec is None:
        kcal_spec = {"protein_g": 4, "carb_g": 4, "fat_g": 9}
    
    # Convert to nutrients
    Nmat = grams_to_nutrients(Y, W)
    
    # Energy
    kcal = kcal_from_nutrients(Nmat, nutrient_names, kcal_spec)
    
    # Macro shares
    macro_pct = macro_shares_percent(Nmat, nutrient_names, kcal)
    
    # AMDR compliance
    amdr_within_bool = amdr_within(macro_pct)
    
    # Suggest a canonical adequacy list in lowercase
    if adequacy_nutrients is None:
        adequacy_nutrients = [
            "calcium_mg","iron_mg","zinc_mg","vitamin_a_mcg","vitamin_c_mg",
            "folate_mcg","vitamin_b12_mcg","thiamin_mg","riboflavin_mg",
            "niacin_mg","vitamin_b6_mg"
        ]
    
    # NAR and MAR
    NAR, MAR = compute_nar_mar(Nmat, RDI, meal_fraction, adequacy_nutrients, nutrient_names)
    
    # MER
    ER, MER = compute_mer(Nmat, limits, nutrient_names, kcal, meal_fraction=meal_fraction)
    
    # Diversity (Group-based Hill diversity if food groups available, otherwise Shannon)
    if food_to_group is not None and feature_names is not None and target_groups is not None:
        diversity = diversity_group_hill_score(Y, W, nutrient_names, feature_names, 
                                             food_to_group, target_groups)
    else:
        diversity = diversity_score_shannon(Y)
    
    # Summary statistics
    summary = {
        "n_meals": len(Y),
        "energy_kcal": {
            "mean": float(np.mean(kcal)),
            "std": float(np.std(kcal)),
            "median": float(np.median(kcal))
        },
        "macro_shares": {
            "protein_pct": {
                "mean": float(np.mean(macro_pct[:, 0])),
                "std": float(np.std(macro_pct[:, 0]))
            },
            "fat_pct": {
                "mean": float(np.mean(macro_pct[:, 1])),
                "std": float(np.std(macro_pct[:, 1]))
            },
            "carb_pct": {
                "mean": float(np.mean(macro_pct[:, 2])),
                "std": float(np.std(macro_pct[:, 2]))
            }
        },
        "amdr_within_pct": float(np.mean(amdr_within_bool)),
        "mar": {
            "mean": float(np.mean(MAR)),
            "std": float(np.std(MAR)),
            "median": float(np.median(MAR))
        },
        "mer": {
            "mean": float(np.nanmean(MER)),
            "std": float(np.nanstd(MER)),
            "median": float(np.nanmedian(MER))
        },
        "diversity": {
            "mean": float(np.mean(diversity)),
            "std": float(np.std(diversity)),
            "median": float(np.median(diversity))
        }
    }
    
    return {
        "nutrients": Nmat,
        "energy_kcal": kcal,
        "macro_shares": macro_pct,
        "amdr_within": amdr_within_bool,
        "NAR": NAR,
        "MAR": MAR,
        "ER": ER,
        "MER": MER,
        "diversity": diversity,
        "summary": summary
    }


def per_meal_coverage(Nmat, nutrient_names, RDI, meal_fraction, key):
    """Compute per-meal coverage for a specific nutrient"""
    canon_map = _canon_map(nutrient_names)
    if key not in canon_map:
        return np.zeros(Nmat.shape[0])
    
    j = canon_map[key]
    tgt = np.maximum(meal_fraction * RDI[j], 1e-8)
    return Nmat[:, j] / tgt


def per_meal_macro_percentages(Nmat, nutrient_names, meal_fraction=None):
    """Compute per-meal macro percentages of calories (meal_fraction ignored)"""
    canon_map = _canon_map(nutrient_names)
    
    # Get nutrient indices
    prot_idx = canon_map.get('protein_g', -1)
    fat_idx = canon_map.get('fat_g', -1)
    carb_idx = canon_map.get('carb_g', -1)
    
    if prot_idx == -1 or fat_idx == -1 or carb_idx == -1:
        return np.zeros((Nmat.shape[0], 3))
    
    # Compute calories per meal
    kcal = kcal_from_nutrients(Nmat, nutrient_names)
    
    # Compute macro percentages
    prot_pct = 100 * (Nmat[:, prot_idx] * 4) / np.maximum(kcal, 1e-6)
    fat_pct = 100 * (Nmat[:, fat_idx] * 9) / np.maximum(kcal, 1e-6)
    carb_pct = 100 * (Nmat[:, carb_idx] * 4) / np.maximum(kcal, 1e-6)
    
    return np.column_stack([prot_pct, fat_pct, carb_pct])


def _group_kcal_shares(X_grams, W, nutrient_names, feature_names, food_to_group,
                       min_kcal=10.0):
    """Calculate kcal shares by food group for each meal."""
    # Calculate kcal per individual food item
    eidx = nutrient_names.index('energy_kcal')
    N, D = X_grams.shape
    # Get energy density for each food (kcal per 100g)
    energy_density = W[:, eidx]  # (D,) - kcal per 100g for each food
    # Calculate kcal for each food item in each meal
    kcal_per_food = X_grams * energy_density / 100.0  # (N, D) - kcal for each food item

    groups = sorted({food_to_group.get(code, 'Other') for code in feature_names})
    g2i = {g:i for i,g in enumerate(groups)}
    G = len(groups)
    grp_kcal = np.zeros((N, G), float)

    for j, code in enumerate(feature_names):
        gi = g2i[food_to_group.get(code, 'Other')]
        grp_kcal[:, gi] += kcal_per_food[:, j]

    # threshold out tiny groups
    grp_kcal[grp_kcal < min_kcal] = 0.0
    totals = grp_kcal.sum(axis=1, keepdims=True)
    p = np.divide(grp_kcal, np.maximum(totals, 1e-8))
    return p, groups

def diversity_group_hill_score(X_grams, W, nutrient_names, feature_names, food_to_group,
                               target_groups, min_kcal=10.0):
    """Calculate group-based Hill diversity score."""
    p, _ = _group_kcal_shares(
        X_grams, W, nutrient_names, feature_names, food_to_group, min_kcal=min_kcal
    )
    H = -np.nansum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)
    N1 = np.exp(H)
    return np.clip(N1 / max(1, target_groups), 0.0, 1.0)

def diversity_score_shannon(Y: np.ndarray, min_grams: float = 5.0) -> np.ndarray:
    """
    Per-meal diversity on [0,1] using Shannon entropy normalized by log(K).

    Steps for each meal i:
      - keep items with grams >= min_grams
      - shares p = grams / sum(grams)
      - Shannon entropy H = -sum(p * log(p))
      - Normalized diversity = H / log(K) where K is number of items
    """
    N, _ = Y.shape
    out = np.zeros(N, dtype=float)
    for i in range(N):
        x = Y[i]
        x = x[np.asarray(x) >= min_grams]
        K = x.size
        if K <= 1:
            out[i] = 0.0
            continue
        p = x / np.sum(x)
        # Avoid log(0) by adding small epsilon
        p = np.maximum(p, 1e-10)
        H = -np.sum(p * np.log(p))
        # Normalize by log(K) to get [0,1] range
        out[i] = H / np.log(K)
    return out
