#!/usr/bin/env python3
"""
Write `outputs/final_results_log.txt`: the manuscript's headline numbers, computed
from one pipeline run (default `outputs/runs/eval_run`).

PLOS Digital Health, https://doi.org/10.1371/journal.pdig.0001367

Sections:
  A) Evaluation counts and cluster totals
  B) RDI deviation reductions (cluster-level medians) and improvement rate
  C) Substitution analysis knee-point results by k (health gain, cost savings)
  D) Energy density (kcal/g) median shift (generated vs real)
  E) MAR median shift (generated vs real)
  F) Vitamin C coverage median shift (generated vs real)

Each section names the files it was computed from. Run `make eval`, `make
substitutions` and `make theta_sweep` first.
"""
from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path
from statistics import median
from typing import Dict, Tuple, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = PROJECT_ROOT / 'outputs' / 'runs' / 'eval_run'
DATA_EXPORTS_DIR = RUN_DIR / 'data_exports'
SWEEP_DIR = RUN_DIR / 'substitutions_k3' / 'figures_theta_sweep'
OUT_LOG = PROJECT_ROOT / 'outputs' / 'final_results_log.txt'


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_paths(
    *,
    run_name: str | None = None,
    out_dir: str | Path | None = None,
    data_exports_dir: str | Path | None = None,
    sweep_dir: str | Path | None = None,
    out_log: str | Path | None = None,
) -> None:
    """Point the script at one run's outputs."""
    global RUN_DIR, DATA_EXPORTS_DIR, SWEEP_DIR, OUT_LOG

    if out_dir is not None:
        RUN_DIR = _resolve_path(out_dir)
    elif run_name is not None:
        RUN_DIR = PROJECT_ROOT / 'outputs' / 'runs' / run_name

    DATA_EXPORTS_DIR = (
        _resolve_path(data_exports_dir)
        if data_exports_dir is not None
        else RUN_DIR / 'data_exports'
    )
    SWEEP_DIR = (
        _resolve_path(sweep_dir)
        if sweep_dir is not None
        else RUN_DIR / 'substitutions_k3' / 'figures_theta_sweep'
    )

    if out_log is not None:
        OUT_LOG = _resolve_path(out_log)


def load_cluster_counts() -> Tuple[Dict[str, int], List[Dict[str, str]]]:
    totals_path = RUN_DIR / 'tables' / 'cluster_counts_totals.csv'
    per_path = RUN_DIR / 'tables' / 'cluster_counts.csv'

    totals = {}
    with totals_path.open(newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            meal = row['meal_type']
            totals[meal] = int(row['assessed_count'])

    rows: List[Dict[str, str]] = []
    with per_path.open(newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    return totals, rows


def load_rdi_boxplot_stats():
    stats_path = RUN_DIR / 'figures' / 'boxplots_rdi_deviation_real_vs_generated_stats.csv'
    rows = []
    with stats_path.open(newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            # Only need: meal, cluster_id, type (Real/Generated), median
            rows.append({
                'meal': row['meal'],
                'cluster_id': int(row['cluster_id']),
                'cluster_name': row.get('cluster_name', ''),
                'type': row['type'],
                'median': float(row['median']) if row['median'] else float('nan'),
            })
    return rows


def compute_rdi_reduction_summaries() -> Dict[str, object]:
    rows = load_rdi_boxplot_stats()
    # Aggregate by (meal, cluster)
    by_mc: Dict[Tuple[str, int], Dict[str, float]] = {}
    names: Dict[Tuple[str, int], str] = {}
    for row in rows:
        key = (row['meal'], row['cluster_id'])
        names[key] = row.get('cluster_name', '')
        if key not in by_mc:
            by_mc[key] = {}
        by_mc[key][row['type']] = row['median']

    improved = 0
    total = 0
    per_meal_reductions: Dict[str, List[float]] = {'Breakfast': [], 'Lunch': [], 'Dinner': []}

    for (meal, _cid), vals in by_mc.items():
        if 'Real' in vals and 'Generated' in vals:
            total += 1
            ar = abs(vals['Real'])
            ag = abs(vals['Generated'])
            if ag < ar:
                improved += 1
            if ar > 0:
                per_meal_reductions[meal].append((ar - ag) / ar * 100.0)

    results = {
        'clusters_improved_pct': (improved / total * 100.0) if total else float('nan'),
        'clusters_total': total,
        'clusters_improved': improved,
        'median_reduction_by_meal': {m: (median(v) if v else float('nan')) for m, v in per_meal_reductions.items()},
        'overall_median_reduction': median([x for mv in per_meal_reductions.values() for x in mv]) if any(per_meal_reductions.values()) else float('nan'),
        'source_path': str(RUN_DIR / 'figures' / 'boxplots_rdi_deviation_real_vs_generated_stats.csv'),
    }
    return results


def weighted_median(values: List[float], weights: List[int]) -> float:
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total_w = sum(weights)
    if total_w <= 0:
        return float('nan')
    cum = 0
    for v, w in pairs:
        cum += w
        if cum >= 0.5 * total_w:
            return v
    return pairs[-1][0]


def compute_weighted_medians_by_meal() -> Dict[str, float]:
    stats_rows = load_rdi_boxplot_stats()
    _totals, counts_rows = load_cluster_counts()
    counts = {(r['meal_type'], int(r['cluster_id'])): int(r['assessed_count']) for r in counts_rows}

    # Build per cluster medians
    med_by = {}
    for r in stats_rows:
        key = (r['meal'], r['cluster_id'])
        med_by.setdefault(key, {})[r['type']] = float(r['median'])

    out = {}
    for meal in ['Breakfast', 'Lunch', 'Dinner']:
        v_real: List[float] = []
        v_gen: List[float] = []
        wts: List[int] = []
        for (m, cid), d in med_by.items():
            if m != meal:
                continue
            if 'Real' in d and 'Generated' in d:
                w = counts.get((meal, cid))
                if w is None:
                    continue
                v_real.append(abs(d['Real']))
                v_gen.append(abs(d['Generated']))
                wts.append(w)
        if v_real and wts:
            wm_r = weighted_median(v_real, wts)
            wm_g = weighted_median(v_gen, wts)
            drop = ((wm_r - wm_g) / wm_r * 100.0) if wm_r > 0 else float('nan')
            out[meal] = drop
        else:
            out[meal] = float('nan')

    # Overall across all meals
    v_real_all: List[float] = []
    v_gen_all: List[float] = []
    w_all: List[int] = []
    for (m, cid), d in med_by.items():
        if 'Real' in d and 'Generated' in d:
            w = counts.get((m, cid))
            if w is None:
                continue
            v_real_all.append(abs(d['Real']))
            v_gen_all.append(abs(d['Generated']))
            w_all.append(w)
    if v_real_all and w_all:
        wm_r = weighted_median(v_real_all, w_all)
        wm_g = weighted_median(v_gen_all, w_all)
        out['ALL'] = ((wm_r - wm_g) / wm_r * 100.0) if wm_r > 0 else float('nan')
    else:
        out['ALL'] = float('nan')
    return out


def compute_substitution_knee_points() -> Dict[str, object]:
    # The knee file and the per-theta debug CSVs must come from the same sweep folder,
    # so read only the configured run's sweep (written by `make theta_sweep`).
    base = SWEEP_DIR
    theta_txt = base / 'optimal_theta_by_k.txt'
    if not theta_txt.exists():
        return {'error': f'missing {theta_txt} (run `make theta_sweep`)'}
    try:
        # The file contains a Python dict literal like: {1: 2.0, 2: 1.0, 3: 2.0}
        theta_map = json.loads(theta_txt.read_text().replace("'", '"'))
    except Exception:
        # fallback: eval-like parsing without using eval
        txt = theta_txt.read_text().strip().strip('{}')
        theta_map = {}
        if txt:
            for pair in txt.split(','):
                k, v = pair.split(':')
                theta_map[int(k.strip())] = float(v.strip().replace('p', '.')) if 'p' in v else float(v)

    def read_debug(theta: float, k: int) -> Tuple[float, float]:
        tag = str(theta).replace('.', 'p')
        p = base / f'debug_points_by_theta_and_k_theta_{tag}.csv'
        with p.open(newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                if int(row['k']) == k:
                    return float(row['H_med']), float(row['C_med'])
        raise FileNotFoundError(f'k={k} not found in {p}')

    # C_med is the median cost change (negative = cheaper); savings = -C_med.
    out = {'source_dir': str(base), 'theta_by_k': theta_map, 'knee_points': {}, 'theta1_points': {}}
    for k, th in theta_map.items():
        h_med, c_med = read_debug(float(th), int(k))
        out['knee_points'][int(k)] = {'theta': float(th), 'health_gain_pct': h_med, 'cost_change_pct': c_med}
        # The Results text reports every k at theta = 1.
        if (base / 'debug_points_by_theta_and_k_theta_1p0.csv').exists():
            h1, c1 = read_debug(1.0, int(k))
            out['theta1_points'][int(k)] = {'health_gain_pct': h1, 'cost_change_pct': c1}
    return out


def compute_energy_density_shifts() -> Dict[str, object]:
    """Compute median energy density (kcal/g) for real vs generated, aligned by
    configured run cluster counts per meal type, and report percent change.

    Uses existing exports only:
      - <run>/data_exports/real_meals_by_type.pkl
      - <run>/data_exports/generated_meals_by_type.pkl
      - <run>/data_exports/clusters_by_meal.pkl
      - <run>/data_exports/gen_clusters_by_meal.pkl
      - <run>/data_exports/nutrient_matrix.npy
      - <run>/data_exports/nutrient_names.json
      - <run>/tables/cluster_counts.csv
    """
    import pickle
    import numpy as np

    data_dir = DATA_EXPORTS_DIR
    path_real = data_dir / 'real_meals_by_type.pkl'
    path_gen = data_dir / 'generated_meals_by_type.pkl'
    path_real_cl = data_dir / 'clusters_by_meal.pkl'
    path_gen_cl = data_dir / 'gen_clusters_by_meal.pkl'
    path_W = data_dir / 'nutrient_matrix.npy'
    path_names = data_dir / 'nutrient_names.json'
    path_counts = RUN_DIR / 'tables' / 'cluster_counts.csv'

    with open(path_real, 'rb') as f:
        real_by_meal = pickle.load(f)
    with open(path_gen, 'rb') as f:
        gen_by_meal = pickle.load(f)
    with open(path_real_cl, 'rb') as f:
        real_clusters = pickle.load(f)
    with open(path_gen_cl, 'rb') as f:
        gen_clusters = pickle.load(f)
    W = np.load(path_W)
    nutrient_names = json.loads(path_names.read_text())
    eidx = nutrient_names.index('energy_kcal')

    # Read desired assessed counts per (meal, cluster)
    desired = {}
    with path_counts.open(newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            desired[(row['meal_type'], int(row['cluster_id']))] = int(row['assessed_count'])

    def energy_density(grams: np.ndarray) -> np.ndarray:
        # grams: N x D; W[:, eidx] is per-100g energy
        kcal = (grams / 100.0) @ W[:, eidx]
        gtot = grams.sum(axis=1)
        # avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ed = np.where(gtot > 0, kcal / gtot, np.nan)
        return ed

    by_meal = {}
    for meal in ['Breakfast', 'Lunch', 'Dinner']:
        Xr = real_by_meal.get(meal)
        Xg = gen_by_meal.get(meal)
        Cr = np.asarray(real_clusters.get(meal))
        Cg = np.asarray(gen_clusters.get(meal))
        if Xr is None or Xg is None or Cr is None or Cg is None:
            raise RuntimeError(f'missing arrays for {meal}')

        # Build aligned subsets by cluster id according to assessed_count
        rows_r = []
        rows_g = []
        for cid in np.unique(Cr):
            key = (meal, int(cid))
            n = desired.get(key)
            if n is None or n <= 0:
                continue
            m_r = np.where(Cr == cid)[0]
            m_g = np.where(Cg == cid)[0]
            if m_r.size < n or m_g.size < n:
                # skip clusters without enough rows in either set
                continue
            rows_r.append(Xr[m_r[:n], :])
            rows_g.append(Xg[m_g[:n], :])
        if not rows_r or not rows_g:
            raise RuntimeError(f'no aligned rows for {meal}')
        Ar = np.vstack(rows_r)
        Ag = np.vstack(rows_g)

        ed_r = energy_density(Ar)
        ed_g = energy_density(Ag)
        med_r = float(np.nanmedian(ed_r))
        med_g = float(np.nanmedian(ed_g))
        pct = ((med_g - med_r) / med_r * 100.0) if med_r != 0 else float('nan')
        by_meal[meal] = {'median_real': med_r, 'median_gen': med_g, 'percent_change': pct}

    return {
        'by_meal': by_meal,
        'sources': [
            str(path_real), str(path_gen), str(path_real_cl), str(path_gen_cl), str(path_W), str(path_names), str(path_counts)
        ],
    }


def _aligned_real_gen_by_meal():
    """Helper: returns aligned (Ar, Ag) grams matrices per meal type using configured run counts.
    """
    import pickle
    import numpy as np
    data_dir = DATA_EXPORTS_DIR
    with open(data_dir / 'real_meals_by_type.pkl', 'rb') as f:
        real_by_meal = pickle.load(f)
    with open(data_dir / 'generated_meals_by_type.pkl', 'rb') as f:
        gen_by_meal = pickle.load(f)
    with open(data_dir / 'clusters_by_meal.pkl', 'rb') as f:
        real_clusters = pickle.load(f)
    with open(data_dir / 'gen_clusters_by_meal.pkl', 'rb') as f:
        gen_clusters = pickle.load(f)
    desired = {}
    with (RUN_DIR / 'tables' / 'cluster_counts.csv').open(newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            desired[(row['meal_type'], int(row['cluster_id']))] = int(row['assessed_count'])
    out = {}
    for meal in ['Breakfast', 'Lunch', 'Dinner']:
        Xr = real_by_meal.get(meal)
        Xg = gen_by_meal.get(meal)
        Cr = np.asarray(real_clusters.get(meal))
        Cg = np.asarray(gen_clusters.get(meal))
        rows_r = []
        rows_g = []
        for cid in np.unique(Cr):
            key = (meal, int(cid))
            n = desired.get(key)
            if n is None or n <= 0:
                continue
            m_r = np.where(Cr == cid)[0]
            m_g = np.where(Cg == cid)[0]
            if m_r.size < n or m_g.size < n:
                continue
            rows_r.append(Xr[m_r[:n], :])
            rows_g.append(Xg[m_g[:n], :])
        if not rows_r or not rows_g:
            raise RuntimeError(f'no aligned rows for {meal}')
        out[meal] = (np.vstack(rows_r), np.vstack(rows_g))
    return out


def compute_mar_shifts() -> Dict[str, object]:
    """Compute MAR (Mean Adequacy Ratio) median percent changes by meal type.
    MAR is computed as mean over nutrients of min(1, coverage), where
    coverage = intake / (RDI_daily * meal_fraction[meal]). Excludes energy_kcal.
    """
    import numpy as np
    aligned = _aligned_real_gen_by_meal()
    W = np.load(DATA_EXPORTS_DIR / 'nutrient_matrix.npy')
    nutrient_names = json.loads((DATA_EXPORTS_DIR / 'nutrient_names.json').read_text())
    RDI_daily = np.load(DATA_EXPORTS_DIR / 'RDI.npy')
    meal_fraction = {'Breakfast': 0.25, 'Lunch': 0.35, 'Dinner': 0.40}

    include = np.array([(n != 'energy_kcal') and np.isfinite(RDI_daily[i]) and (RDI_daily[i] > 0)
                        for i, n in enumerate(nutrient_names)])

    def nutrients(grams):
        return (grams / 100.0) @ W

    by_meal = {}
    for meal in ['Breakfast', 'Lunch', 'Dinner']:
        Ar, Ag = aligned[meal]
        Nr = nutrients(Ar)[:, include]
        Ng = nutrients(Ag)[:, include]
        target = (RDI_daily[include] * meal_fraction[meal])[None, :]
        cov_r = np.clip(Nr / np.maximum(target, 1e-8), 0, 1)
        cov_g = np.clip(Ng / np.maximum(target, 1e-8), 0, 1)
        mar_r = np.mean(cov_r, axis=1)
        mar_g = np.mean(cov_g, axis=1)
        med_r = float(np.nanmedian(mar_r))
        med_g = float(np.nanmedian(mar_g))
        pct = ((med_g - med_r) / med_r * 100.0) if med_r != 0 else float('nan')
        by_meal[meal] = {'median_real': med_r, 'median_gen': med_g, 'percent_change': pct}

    return {
        'by_meal': by_meal,
        'sources': [
            str(DATA_EXPORTS_DIR / 'real_meals_by_type.pkl'),
            str(DATA_EXPORTS_DIR / 'generated_meals_by_type.pkl'),
            str(DATA_EXPORTS_DIR / 'clusters_by_meal.pkl'),
            str(DATA_EXPORTS_DIR / 'gen_clusters_by_meal.pkl'),
            str(DATA_EXPORTS_DIR / 'nutrient_matrix.npy'),
            str(DATA_EXPORTS_DIR / 'nutrient_names.json'),
            str(DATA_EXPORTS_DIR / 'RDI.npy'),
            str(RUN_DIR / 'tables' / 'cluster_counts.csv'),
        ],
    }


def compute_vitamin_c_shifts() -> Dict[str, object]:
    """Compute vitamin C coverage median percent changes by meal type.
    coverage = intake / (RDI_daily * meal_fraction[meal]).
    """
    import numpy as np
    aligned = _aligned_real_gen_by_meal()
    W = np.load(DATA_EXPORTS_DIR / 'nutrient_matrix.npy')
    nutrient_names = json.loads((DATA_EXPORTS_DIR / 'nutrient_names.json').read_text())
    RDI_daily = np.load(DATA_EXPORTS_DIR / 'RDI.npy')
    meal_fraction = {'Breakfast': 0.25, 'Lunch': 0.35, 'Dinner': 0.40}

    # find vitamin C index by strict name pattern
    vit_idx = None
    for i, n in enumerate(nutrient_names):
        nn = n.lower().replace(' ', '_')
        if 'vitamin_c' in nn or 'vitc' in nn:
            vit_idx = i
            break
    if vit_idx is None:
        raise RuntimeError('vitamin C not found in nutrient_names')

    def nutrients(grams):
        return (grams / 100.0) @ W

    by_meal = {}
    for meal in ['Breakfast', 'Lunch', 'Dinner']:
        Ar, Ag = aligned[meal]
        Nr = nutrients(Ar)[:, vit_idx]
        Ng = nutrients(Ag)[:, vit_idx]
        target = float(RDI_daily[vit_idx]) * meal_fraction[meal]
        cov_r = Nr / max(target, 1e-8)
        cov_g = Ng / max(target, 1e-8)
        med_r = float(np.nanmedian(cov_r))
        med_g = float(np.nanmedian(cov_g))
        pct = ((med_g - med_r) / med_r * 100.0) if med_r != 0 else float('nan')
        by_meal[meal] = {'median_real': med_r, 'median_gen': med_g, 'percent_change': pct}

    return {
        'by_meal': by_meal,
        'vitamin_c_name': nutrient_names[vit_idx],
        'sources': [
            str(DATA_EXPORTS_DIR / 'real_meals_by_type.pkl'),
            str(DATA_EXPORTS_DIR / 'generated_meals_by_type.pkl'),
            str(DATA_EXPORTS_DIR / 'clusters_by_meal.pkl'),
            str(DATA_EXPORTS_DIR / 'gen_clusters_by_meal.pkl'),
            str(DATA_EXPORTS_DIR / 'nutrient_matrix.npy'),
            str(DATA_EXPORTS_DIR / 'nutrient_names.json'),
            str(DATA_EXPORTS_DIR / 'RDI.npy'),
            str(RUN_DIR / 'tables' / 'cluster_counts.csv'),
        ],
    }


def write_log():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LOG.open('w') as f:
        # A) Counts
        totals, _per = load_cluster_counts()
        f.write('=== Section A | Evaluation Counts and Clusters ===\n')
        f.write(f"Source (totals): {RUN_DIR / 'tables' / 'cluster_counts_totals.csv'}\n")
        f.write(f"Source (per-cluster): {RUN_DIR / 'tables' / 'cluster_counts.csv'}\n")
        for meal in ['Breakfast', 'Lunch', 'Dinner', 'ALL']:
            if meal in totals:
                f.write(f"{meal}: assessed_count={totals[meal]}\n")
        f.write('\n')

        # B) RDI deviation reductions and improvement
        rdi = compute_rdi_reduction_summaries()
        weighted = compute_weighted_medians_by_meal()
        f.write('=== Section B | RDI Deviation Reductions (Generated vs Real) ===\n')
        f.write(f"Source: {rdi['source_path']}\n")
        f.write(f"Clusters improved: {rdi['clusters_improved']}/{rdi['clusters_total']} ({rdi['clusters_improved_pct']:.1f}%)\n")
        f.write('Median % reduction by meal (cluster medians):\n')
        for meal in ['Breakfast', 'Lunch', 'Dinner']:
            v = rdi['median_reduction_by_meal'][meal]
            f.write(f"  {meal}: {v:.1f}%\n")
        f.write(f"Overall median % reduction (across cluster medians): {rdi['overall_median_reduction']:.1f}%\n")
        f.write('Weighted-median of cluster medians (by assessed_count):\n')
        for k in ['Breakfast', 'Lunch', 'Dinner', 'ALL']:
            if k in weighted:
                f.write(f"  {k}: {weighted[k]:.1f}%\n")
        f.write('\n')

        # C) Substitution knee points
        subs = compute_substitution_knee_points()
        f.write('=== Section C | Substitution Knee Points (per k) ===\n')
        if 'error' in subs:
            f.write(f"Error: {subs['error']}\n")
        else:
            f.write(f"Source dir: {subs['source_dir']}\n")
            f.write(f"Knee theta by k (Figure 4B): {subs['theta_by_k']}\n")
            for k in sorted(subs['knee_points'].keys()):
                kp = subs['knee_points'][k]
                f.write(f"  k={k}: theta={kp['theta']}, health_gain_med={kp['health_gain_pct']:.2f}%, "
                        f"cost_change_med={kp['cost_change_pct']:.2f}% (savings {-kp['cost_change_pct']:.1f}%)\n")
            if subs['theta1_points']:
                f.write("At theta = 1 for every k (Results text, S1 Table 18):\n")
                for k in sorted(subs['theta1_points'].keys()):
                    p = subs['theta1_points'][k]
                    f.write(f"  k={k}: health_gain_med={p['health_gain_pct']:.2f}%, "
                            f"cost_change_med={p['cost_change_pct']:.2f}% (savings {-p['cost_change_pct']:.1f}%)\n")
        f.write('\n')

        # D) Energy density shifts
        try:
            ed = compute_energy_density_shifts()
            f.write('=== Section D | Energy Density (kcal/g) Median Shift (Gen vs Real) ===\n')
            f.write('Sources:\n')
            for src in ed['sources']:
                f.write(f"  {src}\n")
            for meal in ['Breakfast', 'Lunch', 'Dinner']:
                vals = ed['by_meal'][meal]
                f.write(f"{meal}: median_real={vals['median_real']:.4f}, median_gen={vals['median_gen']:.4f}, percent_change={vals['percent_change']:.2f}%\n")
            f.write('\n')
        except Exception as e:
            f.write('=== Section D | Energy Density (kcal/g) Median Shift (Gen vs Real) ===\n')
            f.write(f"Error computing energy density shifts: {e}\n\n")

        # E) MAR shifts
        try:
            mar = compute_mar_shifts()
            f.write('=== Section E | MAR Median Shift (Gen vs Real) ===\n')
            f.write('Sources:\n')
            for src in mar['sources']:
                f.write(f"  {src}\n")
            for meal in ['Breakfast', 'Lunch', 'Dinner']:
                vals = mar['by_meal'][meal]
                f.write(f"{meal}: median_real={vals['median_real']:.4f}, median_gen={vals['median_gen']:.4f}, percent_change={vals['percent_change']:.2f}%\n")
            f.write('\n')
        except Exception as e:
            f.write('=== Section E | MAR Median Shift (Gen vs Real) ===\n')
            f.write(f"Error computing MAR shifts: {e}\n\n")

        # F) Vitamin C coverage shifts
        try:
            vc = compute_vitamin_c_shifts()
            f.write('=== Section F | Vitamin C Coverage Median Shift (Gen vs Real) ===\n')
            f.write(f"Nutrient: {vc['vitamin_c_name']}\n")
            f.write('Sources:\n')
            for src in vc['sources']:
                f.write(f"  {src}\n")
            for meal in ['Breakfast', 'Lunch', 'Dinner']:
                vals = vc['by_meal'][meal]
                f.write(f"{meal}: median_real={vals['median_real']:.4f}, median_gen={vals['median_gen']:.4f}, percent_change={vals['percent_change']:.2f}%\n")
            f.write('\n')
        except Exception as e:
            f.write('=== Section F | Vitamin C Coverage Median Shift (Gen vs Real) ===\n')
            f.write(f"Error computing Vitamin C shifts: {e}\n\n")

    return str(OUT_LOG)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate manuscript headline numbers from regenerated pipeline outputs.'
    )
    parser.add_argument(
        '--run-name',
        default=None,
        help='Run name under outputs/runs/ to read (default: eval_run unless --out-dir is provided).',
    )
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Explicit run output directory containing tables/, figures/, and data_exports/.',
    )
    parser.add_argument(
        '--data-exports-dir',
        default=None,
        help='Explicit data_exports directory. Defaults to <out-dir>/data_exports.',
    )
    parser.add_argument(
        '--sweep-dir',
        default=None,
        help='Theta-sweep folder. Defaults to <out-dir>/substitutions_k3/figures_theta_sweep.',
    )
    parser.add_argument(
        '--out-log',
        default=None,
        help='Output log path. Defaults to outputs/final_results_log.txt.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    configure_paths(
        run_name=args.run_name,
        out_dir=args.out_dir,
        data_exports_dir=args.data_exports_dir,
        sweep_dir=args.sweep_dir,
        out_log=args.out_log,
    )
    path = write_log()
    print(f"Results log written: {path}")


