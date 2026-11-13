"""
Bootstrap statistics and effect size calculations for meal evaluation.

This module provides functions for bootstrap resampling, statistical testing,
and effect size calculations used in the evaluation pipeline. `make eval` calls
`bootstrap_cluster_boxplot_data` and `create_bootstrap_summary_table` through
src/pipeline_eval/evaluation.py; the results are printed to the log, not saved.

Conventions:
- nutrient_names are CANONICAL (e.g., "energy_kcal", "protein_g", "carb_g", "sugars_total_g", "sodium_mg", "sfa_g", ...)
- W is PER-100 g. Convert meals via: N = (X_grams / 100.0) @ W
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats


# -------------------------
# Effect size / Bootstraps
# -------------------------

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cliff's delta effect size between two samples.

    Args:
        x: First sample
        y: Second sample

    Returns:
        Cliff's delta value (-1 to 1)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    combined = np.concatenate([x, y])
    ranks = stats.rankdata(combined, method="average")
    rx = ranks[:n_x]
    # Mann-Whitney U for x vs y
    U = rx.sum() - n_x * (n_x + 1) / 2.0
    delta = (2.0 * U) / (n_x * n_y) - 1.0
    return float(delta)


def bootstrap_median(
    x: np.ndarray,
    n_boot: int = 200,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Bootstrap resampling to estimate the distribution of the median.
    Returns an array of length n_boot (empty if x is empty).
    """
    x = np.asarray(x)
    if x.size == 0:
        return np.array([], dtype=float)
    if sample_size is None:
        sample_size = len(x)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(n_boot, sample_size))
    return np.median(x[indices], axis=1)


# -------------------------
# Multiple testing (BH-FDR)
# -------------------------

def benjamini_hochberg(pvals: np.ndarray, q: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Benjamini-Hochberg FDR correction.

    Args:
        pvals: Array of p-values
        q: Desired FDR level

    Returns:
        rejected: Boolean array
        p_adj: Adjusted p-values (monotone)
    """
    pvals = np.asarray(pvals, float)
    if pvals.size == 0:
        return np.array([], dtype=bool), np.array([])
    m = pvals.size
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    ranks = np.arange(1, m + 1, dtype=float)
    p_adj_sorted = sorted_p * m / ranks
    # enforce monotonicity
    p_adj_sorted = np.minimum.accumulate(p_adj_sorted[::-1])[::-1]
    p_adj_sorted = np.clip(p_adj_sorted, 0.0, 1.0)
    p_adj = np.empty_like(p_adj_sorted)
    p_adj[order] = p_adj_sorted
    return p_adj <= q, p_adj


# -------------------------
# Main bootstrap function
# -------------------------

def bootstrap_cluster_boxplot_data(
    X_real_k: np.ndarray, X_gen_k: np.ndarray, W: np.ndarray,
    nutrient_names: List[str], limits: Dict[str, float],
    kcal_spec: Dict[str, float], n_boot: int = 200,
    sample_size: int = 300, seed: int = 42,
    X_gen_adj_k: Optional[np.ndarray] = None
) -> Dict:
    """
    Generate bootstrap data for cluster-wise comparisons.
    - Non-energy nutrients are standardized per 1000 kcal.
    - Energy is expressed as a ratio to the real cluster median (unitless).
    - Extreme kcal meals are filtered out for stability.
    - "p_value" is the share of bootstrap (gen - real) median differences <= 0 (one-sided).
    `limits` is not used; `X_gen_adj_k` is an optional third set (None in the pipeline).
    Returns {} (with a warning) if anything fails.
    """
    try:
        # 1) Convert grams -> nutrients
        N_real = (X_real_k / 100.0) @ W
        N_gen  = (X_gen_k  / 100.0) @ W
        N_adj  = (X_gen_adj_k / 100.0) @ W if X_gen_adj_k is not None else None

        # 2) kcal for each set
        def _kcal(N):
            if N is None: return None
            if "energy_kcal" in nutrient_names:
                return N[:, nutrient_names.index("energy_kcal")]
            # synthesize from macros if needed
            out = np.zeros(N.shape[0], dtype=float)
            for k, f in kcal_spec.items():
                if k in nutrient_names:
                    out += N[:, nutrient_names.index(k)] * f
            return out

        kcal_real = _kcal(N_real)
        kcal_gen  = _kcal(N_gen)
        kcal_adj  = _kcal(N_adj) if N_adj is not None else None

        # 3) Filter out silly kcal outliers (robust cluster-wise)
        # keep meals within [q1-3*IQR, q3+3*IQR] and also a broad absolute window
        def _valid_kcal_mask(kc):
            if kc is None or len(kc) == 0: return None
            q1, q3 = np.percentile(kc, [25, 75])
            iqr = max(q3 - q1, 1.0)
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
            # also clamp to a broad realistic window (per-meal)
            lo = max(lo, 50.0)
            hi = min(hi, 2000.0)
            return (kc >= lo) & (kc <= hi)

        m_real = _valid_kcal_mask(kcal_real)
        m_gen  = _valid_kcal_mask(kcal_gen)
        m_adj  = _valid_kcal_mask(kcal_adj) if kcal_adj is not None else None

        if m_real is not None: 
            N_real = N_real[m_real]; kcal_real = kcal_real[m_real]
        if m_gen is not None: 
            N_gen  = N_gen[m_gen];   kcal_gen  = kcal_gen[m_gen]
        if N_adj is not None and m_adj is not None:
            N_adj  = N_adj[m_adj];   kcal_adj  = kcal_adj[m_adj]

        # 4) Which nutrients to analyze (canonical)
        all_canon = [
            "energy_kcal", "protein_g", "fat_g", "sfa_g", "carb_g",
            "fiber_g", "sodium_mg", "sugars_total_g"
        ]
        analysis_nutrients = [n for n in all_canon if n in nutrient_names]

        results = {}
        # energy reference (real cluster median)
        energy_ref = float(np.median(kcal_real)) if kcal_real is not None and len(kcal_real) else None

        for nutrient in analysis_nutrients:
            idx = nutrient_names.index(nutrient)

            # raw values
            r = N_real[:, idx]
            g = N_gen[:,  idx]
            a = N_adj[:,  idx] if N_adj is not None else None

            # 5) Standardize
            if nutrient == "energy_kcal":
                # unitless ratio to real cluster median
                if energy_ref is None or energy_ref <= 0:
                    # fallback: just use kcal and hope for the best
                    real_values = r
                    gen_values  = g
                    adj_values  = a if a is not None else None
                else:
                    real_values = kcal_real / energy_ref
                    gen_values  = kcal_gen  / energy_ref
                    adj_values  = (kcal_adj / energy_ref) if kcal_adj is not None else None
            else:
                # per 1000 kcal
                real_values = r / np.maximum(kcal_real, 1e-6) * 1000.0
                gen_values  = g / np.maximum(kcal_gen,  1e-6) * 1000.0
                adj_values  = (a / np.maximum(kcal_adj, 1e-6) * 1000.0) if a is not None else None

            # 6) Bootstrap medians (with replacement, fixed sample size)
            ss_r = min(sample_size, len(real_values)) if len(real_values) else 0
            ss_g = min(sample_size, len(gen_values))  if len(gen_values)  else 0

            if ss_r == 0 or ss_g == 0:
                # not enough data to compare
                continue

            real_medians = bootstrap_median(real_values, n_boot=n_boot, sample_size=ss_r, seed=seed)
            gen_medians  = bootstrap_median(gen_values,  n_boot=n_boot, sample_size=ss_g, seed=seed)
            if adj_values is not None and len(adj_values) > 0:
                ss_a = min(sample_size, len(adj_values))
                gen_adj_medians = bootstrap_median(adj_values, n_boot=n_boot, sample_size=ss_a, seed=seed)
            else:
                gen_adj_medians = np.array([])

            delta_medians = gen_medians - real_medians
            p_value = np.mean(delta_medians <= 0) if np.std(delta_medians) > 0 else 0.5
            effect_size = cliffs_delta(real_values, gen_values)

            results[nutrient] = {
                "real_medians": real_medians,
                "gen_medians": gen_medians,
                "gen_adj_medians": gen_adj_medians,
                "delta_medians": delta_medians,
                "p_value": p_value,
                "effect_size": effect_size,
                "real_values": real_values,
                "gen_values": gen_values,
            }

        return results
    except Exception as e:
        warnings.warn(f"bootstrap_cluster_boxplot_data failed: {e}")
        return {}


# -------------------------
# Summaries / Tables
# -------------------------

def create_bootstrap_summary_table(
    bootstrap_results: Dict[str, Dict],
    meal_types: List[str],
    clusters: List[int],
) -> pd.DataFrame:
    """
    Create a summary table of bootstrap test results.
    """
    rows = []
    for meal_type in meal_types:
        for cluster in clusters:
            key = f"{meal_type}_{cluster}"
            if key not in bootstrap_results:
                continue
            by_nutr = bootstrap_results[key]
            for nutrient, res in by_nutr.items():
                rows.append({
                    "meal_type": meal_type,
                    "cluster": cluster,
                    "nutrient": nutrient,
                    "p_value": res.get("p_value", np.nan),
                    "effect_size": res.get("effect_size", np.nan),
                    "real_median": np.nanmedian(res.get("real_medians", np.array([np.nan]))),
                    "gen_median":  np.nanmedian(res.get("gen_medians",  np.array([np.nan]))),
                    "delta_median": np.nanmedian(res.get("delta_medians", np.array([np.nan]))),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    rejected, p_adj = benjamini_hochberg(df["p_value"].values)
    df["p_adj"] = p_adj
    df["rejected"] = rejected

    def stars(p):
        return "***" if p <= 0.001 else ("**" if p <= 0.01 else ("*" if p <= 0.05 else ""))

    df["significance"] = [stars(p) for p in p_adj]
    return df
