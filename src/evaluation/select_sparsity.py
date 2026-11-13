"""
Choose the food-prevalence cut-off ("sparsity percentile") for a meal x food matrix.

`choose_sparsity_threshold` scores a grid of percentile cut-offs (foods kept, meals kept, median
foods per meal, Gini utility retained) and returns the chosen percentile, the per-percentile
metrics and the reduced matrix. Called by src/preprocessing/data_preprocessing.py
(`make preprocess`), which writes `<meal>_sparsity_filtered.pkl` and the sparsity diagnostics.
"""
import numpy as np
import pandas as pd

def _presence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # strictly presence/absence for stability of metrics
    return (df > 0).astype(np.uint8)

def _prevalence(pres: pd.DataFrame) -> np.ndarray:
    # column-wise presence rate
    return pres.mean(axis=0).values  # shape (D,)

def _gini_utility(p: np.ndarray) -> float:
    # sum over foods of the Gini impurity 2*p*(1-p), each term in [0, 0.5]
    return float(np.sum(2.0 * p * (1.0 - p)))

def _apply_percentile_threshold(
    df: pd.DataFrame,
    pcts: float,
    *,
    p_min_abs: float | None = None,
    min_occurrences: int | None = None,
    max_foods: int | None = None,
    foods_ratio: float | None = None,
) -> pd.DataFrame:
    """Keep foods whose prevalence is at or above the `pcts` percentile and the optional
    absolute floors, cap them at the top K by utility 2*p*(1-p), then drop meals left empty."""
    pres = _presence_matrix(df)
    p = _prevalence(pres)                           # prevalence per column
    occ = pres.sum(axis=0).values                   # occurrence count per column
    # percentile filter
    thresh = np.percentile(p, pcts)
    keep_mask = p >= thresh
    # absolute support filters (optional)
    if p_min_abs is not None:
        keep_mask &= (p >= float(p_min_abs))
    if min_occurrences is not None:
        keep_mask &= (occ >= int(min_occurrences))
    kept_cols = pres.columns[keep_mask]
    if len(kept_cols) == 0:
        return df.iloc[0:0, :]                      # empty DF with same columns
    # cap by top-K utility / prevalence if requested
    reduced = df.loc[:, kept_cols]
    if (max_foods is not None) or (foods_ratio is not None):
        K = max_foods if max_foods is not None else int(np.ceil(foods_ratio * df.shape[1]))
        if K < reduced.shape[1]:
            # rank kept columns by Bernoulli utility u = 2*p(1-p) (prefer informative)
            p_keep = p[keep_mask]
            u = 2.0 * p_keep * (1.0 - p_keep)
            order = np.argsort(-u)                  # descending utility
            kept_cols = kept_cols[order[:K]]
            reduced = df.loc[:, kept_cols]
    # drop meals that become all-zero
    reduced = reduced.loc[(reduced != 0).any(axis=1)]
    return reduced

def _median_nonzeros_per_row(df: pd.DataFrame) -> float:
    return float((df != 0).sum(axis=1).median()) if len(df) else 0.0

def _knee_index(y: np.ndarray) -> int:
    """
    Simple knee ('kneedle') on a monotone curve y(t): the index of the point with the
    maximum distance to the chord between the endpoints.
    """
    x = np.arange(len(y))
    # line from first to last
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    # distances
    num = np.abs((y1 - y0) * x - (x1 - x0) * y + (x1*y0 - y1*x0))
    den = np.sqrt((y1 - y0)**2 + (x1 - x0)**2)
    d = num / (den + 1e-12)
    return int(np.argmax(d))

def choose_sparsity_threshold(
    df: pd.DataFrame,
    percent_grid = tuple(range(0, 100, 5)),
    min_meal_retain: float = 0.90,   # keep ≥90% meals by default
    min_median_nonzeros: int = 3,    # require at least 3 items per meal on median
    min_utility_retain: float = 0.92, # guard on retained information
    *,
    # tighter-keep knobs:
    strong_mode: bool = False,         # pick MOST aggressive pct that passes guards
    p_min_abs: float | None = None,    # absolute prevalence floor (e.g., 0.02)
    min_occurrences: int | None = None,# absolute min row count support
    min_occurrences_frac: float | None = None, # or fraction of rows (e.g., 0.02)
    max_foods: int | None = None,      # hard cap (top-K after filters)
    foods_ratio: float | None = None,  # or ratio cap (e.g., 0.5 = ≤50% columns)
) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    """
    Returns (best_percentile, metrics_df, reduced_df).
    metrics_df columns:
      ['percentile','foods_retained','meals_retained','median_nonzeros','utility','utility_retain']

    A percentile passes the guards if it keeps >= min_meal_retain of the meals, a median of
    >= min_median_nonzeros foods per meal and >= min_utility_retain of the utility. The choice
    is the knee of the utility-loss curve clamped into the passing range (strong_mode: the
    largest passing percentile); if none passes, the smallest percentile in the grid.
    """
    assert df.shape[0] > 0 and df.shape[1] > 0, "Empty input dataframe."
    pres0 = _presence_matrix(df)
    p0 = _prevalence(pres0)
    U0 = _gini_utility(p0)
    M0 = df.shape[0]

    rows = []
    reduced_cache = {}

    if (min_occurrences is None) and (min_occurrences_frac is not None):
        min_occurrences = int(np.ceil(min_occurrences_frac * M0))

    for pct in percent_grid:
        reduced = _apply_percentile_threshold(
            df, pct,
            p_min_abs=p_min_abs,
            min_occurrences=min_occurrences,
            max_foods=max_foods,
            foods_ratio=foods_ratio,
        )
        reduced_cache[pct] = reduced

        pres = _presence_matrix(reduced)
        if reduced.shape[1] == 0:
            rows.append([pct, 0, 0, 0.0, 0.0, 0.0])
            continue

        p = _prevalence(pres) if pres.shape[1] else np.array([])
        U = _gini_utility(p) if p.size else 0.0

        foods = reduced.shape[1]
        meals = reduced.shape[0]
        med_nz = _median_nonzeros_per_row(reduced)
        rows.append([pct, foods, meals, med_nz, U, U / (U0 + 1e-12)])

    metrics = pd.DataFrame(rows, columns=[
        'percentile','foods_retained','meals_retained','median_nonzeros','utility','utility_retain'
    ])

    # Convert to loss curve for knee detection (1 - utility_retain)
    util_retain = metrics['utility_retain'].values
    loss = 1.0 - np.clip(util_retain, 0.0, 1.0)

    # candidate by knee
    knee_idx = _knee_index(loss)
    knee_pct = metrics.loc[knee_idx, 'percentile']

    # candidate set that passes guards
    viable = metrics.loc[
            (metrics['meals_retained'] >= min_meal_retain * M0) &
            (metrics['median_nonzeros'] >= min_median_nonzeros) &
            (metrics['utility_retain'] >= min_utility_retain)
    ]
    if len(viable):
        if strong_mode:
            # MOST aggressive that still passes (keeps fewest foods)
            chosen = int(viable['percentile'].max())
        else:
            # knee by default, but clamp into viable range
            chosen = int(np.clip(knee_pct, viable['percentile'].min(), viable['percentile'].max()))
    else:
        # nothing passes; pick the least aggressive to preserve data
        chosen = int(metrics['percentile'].min())

    return chosen, metrics, reduced_cache[chosen]
