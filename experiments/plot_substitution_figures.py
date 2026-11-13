"""
Figures from the food-substitution runs (Figure 4B-D).

--mode k_from_input (`make substitutions`, one run folder per k):
    with --with_category_heatmap, writes figure_category_heatmap and
    figure_category_add_vs_remove_scatter (.png/.svg) to --out_dir; when the sibling
    tradeoff_theta_<θ>_k{1,2,3} runs exist it also writes
    figure_category_scatter_plus_heatmap_k123 to figures_theta<θ>_k123/.
--mode optimal_theta (`make theta_sweep`, all θ runs under --theta_runs_base):
    picks the knee θ per k (optimal_theta_by_k.json/.txt) and draws Figure 4B
    (figure_cost_benefit_points_by_theta_and_k) and Figure 4C-D
    (figure_category_scatter_plus_heatmap_k123, each k at its knee θ). The per-k matches
    of each θ are combined under <out_dir>/_tmp_theta_grouped/. --dump_debug also writes
    debug_points_by_theta_and_k_*.csv, read by scripts/generate_manuscript_numbers_log.py.
--with_ternary adds optional health/savings/simplicity ternary plots (not used by the Makefile).

Usage (as in the Makefile):
    python -m experiments.plot_substitution_figures \
        --input_dir outputs/runs/eval_run/substitutions_k3/tradeoff_theta_0p5_k1 \
        --out_dir outputs/runs/eval_run/substitutions_k3/figures_theta0p5_k1 \
        --mode k_from_input --with_category_heatmap
    python -m experiments.plot_substitution_figures \
        --theta_runs_base outputs/runs/eval_run/substitutions_k3 \
        --input_dir outputs/runs/eval_run/substitutions_k3 \
        --out_dir outputs/runs/eval_run/substitutions_k3/figures_theta_sweep \
        --mode optimal_theta --dump_debug
"""
import logging
import sys
from pathlib import Path
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

import argparse
import json
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import shutil
## SVG/Path defaults to reduce filesize while keeping quality
mpl.rcParams['svg.fonttype'] = 'none'  # keep text selectable
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 0.6
mpl.rcParams['agg.path.chunksize'] = 20000
mpl.rcParams['image.composite_image'] = True


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# Global debug dump switch (set in main from --dump_debug)
_DUMP_DEBUG: bool = False


def _dump_df(path: Path, df: pd.DataFrame) -> None:
    try:
        _ensure_dir(path.parent)
        df.to_csv(path, index=False)
    except Exception as _e:
        print(f"[debug dump] Failed writing {path}: {_e}")


def _ensure_k_substitution(df: pd.DataFrame) -> pd.DataFrame:
    """Cast df['k_substitution'] to Int64 and copy it to df['k'], the column some plots read."""
    if 'k_substitution' not in df.columns:
        raise ValueError("Missing 'k_substitution' in substitution_matches.csv.")
    out = df.copy()
    out['k_substitution'] = pd.to_numeric(out['k_substitution'], errors='coerce').astype('Int64')
    out['k'] = out['k_substitution']
    return out


def _read_matches(input_dir: Path, winners_only: bool = False) -> pd.DataFrame:
    fp = input_dir / "substitution_matches.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing substitution_matches.csv in {input_dir}")
    df = pd.read_csv(fp)
    # Optional winners filter (default: all rows)
    if winners_only and "selected" in df.columns:
        df = df[df["selected"] == True]
    # Normalize column names
    rename = {
        "total_food_diff": "k",
        "improvement": "dRDI",
        "delta_cost": "dCost",
        "delta_mar": "dMAR",
        "delta_amdr": "dAMDR",
        "delta_mer": "dMER",
        "delta_ed": "dED",
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    # Normalize/ensure k_substitution and mirror k for labels
    df = _ensure_k_substitution(df)
    # types
    if "k" in df.columns:
        df["k"] = pd.to_numeric(df["k"], errors="coerce").astype("Int64")
    if "dCost" in df.columns:
        df["dCost"] = pd.to_numeric(df["dCost"], errors="coerce")
    if "dRDI" in df.columns:
        df["dRDI"] = pd.to_numeric(df["dRDI"], errors="coerce")
    return df


def _replace_inf_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ±inf replaced by NaN."""
    return df.replace([np.inf, -np.inf], np.nan)


def _ensure_percent_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dRDI_pct and dCost_pct exist with the same logic used throughout the file."""
    out = df.copy()
    # ΔRDI in percent (prefer explicit column; else derive from dRDI or improvement)
    if 'dRDI_pct' in out.columns:
        out['dRDI_pct'] = pd.to_numeric(out['dRDI_pct'], errors='coerce')
    else:
        out['dRDI_pct'] = pd.to_numeric(out.get('dRDI_pct', 100.0 * out.get('dRDI', out.get('improvement', np.nan))), errors='coerce')
    # ΔCost in percent relative to real cost (create only if not already present)
    if 'dCost_pct' in out.columns:
        out['dCost_pct'] = pd.to_numeric(out['dCost_pct'], errors='coerce')
    elif {'real_cost', 'dCost'}.issubset(out.columns):
        with np.errstate(divide='ignore', invalid='ignore'):
            out['dCost_pct'] = 100.0 * (pd.to_numeric(out['dCost'], errors='coerce') / pd.to_numeric(out['real_cost'], errors='coerce'))
    return out


def _norm01(x: np.ndarray | float, vmax: float, gamma: float = 0.65):
    """Normalize to [0,1] with optional gamma to spread low values (gamma<1 brightens lows)."""
    import numpy as _np
    vmax = float(vmax) if _np.isfinite(vmax) else 0.0
    if vmax <= 0.0:
        return 0.0 if _np.isscalar(x) else _np.zeros_like(_np.asarray(x, float))
    r = _np.clip(_np.asarray(x, float) / vmax, 0.0, 1.0)
    if isinstance(gamma, (int, float)) and float(gamma) > 0 and float(gamma) != 1.0:
        r = _np.power(r, float(gamma))
    return float(r) if _np.isscalar(x) else r


def _render_tri_heatmap_as_image(ax: plt.Axes, f, cmap, scale: int, nx: int = 360) -> None:
    """Render a ternary heatmap function as a single raster image clipped to the
    equilateral triangle. This avoids thousands of SVG path elements.

    f: callable taking a lattice tuple (c,b,h) with sum==scale and returning [0,1]
    cmap: Matplotlib colormap
    scale: ternary lattice scale used for integer conversion
    nx: image width in pixels (height derived to maintain triangle aspect)
    """
    import numpy as _np
    from matplotlib.path import Path as _Path
    from matplotlib.patches import PathPatch as _PathPatch
    # Triangle bounds in XY (see _bary_to_xy)
    x0, x1 = 0.0, 1.0
    y0, y1 = 0.0, _np.sqrt(3.0) / 2.0
    ny = int(nx * (y1 - y0) / (x1 - x0))
    xs = _np.linspace(x0, x1, nx)
    ys = _np.linspace(y0, y1, ny)
    XX, YY = _np.meshgrid(xs, ys)
    # Convert (x,y) -> (h,c,b)
    H = (2.0 / _np.sqrt(3.0)) * YY
    B = XX - 0.5 * H
    C = 1.0 - H - B
    # Triangle inequality mask
    inside = (YY >= 0.0) & (YY <= _np.sqrt(3.0)*XX + 1e-9) & (YY <= -_np.sqrt(3.0)*XX + _np.sqrt(3.0) + 1e-9)
    H = _np.clip(H, 0.0, 1.0); B = _np.clip(B, 0.0, 1.0); C = _np.clip(C, 0.0, 1.0)
    vals = _np.full_like(XX, _np.nan, dtype=float)
    if _np.any(inside):
        hs = H[inside] * scale; cs = C[inside] * scale; bs = B[inside] * scale
        h_i = _np.floor(hs).astype(int); c_i = _np.floor(cs).astype(int); b_i = _np.floor(bs).astype(int)
        s = h_i + c_i + b_i
        diff = (scale - s).astype(int)
        if diff.size:
            fr_h = hs - h_i; fr_c = cs - c_i; fr_b = bs - b_i
            winners = _np.argmax(_np.stack([fr_c, fr_b, fr_h], axis=0), axis=0)
            c_i = c_i + (diff * (winners == 0)); b_i = b_i + (diff * (winners == 1)); h_i = h_i + (diff * (winners == 2))
        flat = []
        for ci, bi, hi in zip(c_i.tolist(), b_i.tolist(), h_i.tolist()):
            try:
                flat.append(float(_np.clip(f((ci, bi, hi)), 0.0, 1.0)))
            except Exception:
                flat.append(_np.nan)
        vals[inside] = _np.asarray(flat, float)
    if not _np.isfinite(vals).any():
        return
    try:
        cmap_use = cmap.copy(); cmap_use.set_bad(alpha=0.0)
    except Exception:
        cmap_use = cmap
    im = ax.imshow(vals, extent=(x0, x1, y0, y1), origin='lower', cmap=cmap_use, vmin=0.0, vmax=1.0, interpolation='bilinear', zorder=0.5)
    try:
        im.set_rasterized(True)
    except Exception:
        pass
    tri = _Path([(0.0, 0.0), (1.0, 0.0), (0.5, y1)])
    im.set_clip_path(_PathPatch(tri, transform=ax.transData))
    try:
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    except Exception:
        pass


def _filter_selected(df: pd.DataFrame, winners_only: bool) -> pd.DataFrame:
    """If winners_only and 'selected' in columns, filter to selected == True."""
    if winners_only and 'selected' in df.columns:
        return df[df['selected'] == True]
    return df


def _matches_df(input_dir: Path, winners_only: bool = False) -> pd.DataFrame:
    """Read matches and apply common cleaning: replace inf, optional winners filter, ensure percent cols."""
    df = _read_matches(Path(input_dir))
    df = _replace_inf_nan(df)
    df = _filter_selected(df, winners_only=winners_only)
    df = _ensure_percent_cols(df)
    return df


def _smooth_lattice_grid(
    grid: dict[tuple[int,int,int], float],
    scale: int,
    passes: int = 1,
    include_self: bool = True,
) -> dict[tuple[int,int,int], float]:
    """Iterative neighbor-averaging over the triangular lattice keyed by (h,s,p).

    - passes: number of smoothing iterations (>=1)
    - include_self: include a cell's current value in its own average when present
    """
    nbrs = [(1,0,-1),(-1,0,1),(0,1,-1),(0,-1,1),(1,-1,0),(-1,1,0)]
    cur = dict(grid)
    for _ in range(max(1, int(passes))):
        out: dict[tuple[int,int,int], float] = {}
        for h in range(scale+1):
            for s in range(scale+1-h):
                p = scale - h - s
                acc = 0.0; w = 0.0
                if include_self:
                    key = (h, s, p)
                    if key in cur:
                        acc += cur[key]; w += 1.0
                for dh, ds, dp in nbrs:
                    nh, ns, np0 = h+dh, s+ds, p+dp
                    if 0 <= nh <= scale and 0 <= ns <= scale and 0 <= np0 <= scale and nh+ns+np0 == scale:
                        k2 = (nh, ns, np0)
                        if k2 in cur:
                            acc += cur[k2]; w += 1.0
                if w > 0:
                    out[(h, s, p)] = acc / w
        cur = out
    return cur


def _effort_kaware(gm: np.ndarray, rm: np.ndarray, k: int, alpha: float = 0.5) -> tuple[float, float, float, float, float, float, float]:
    """K-aware effort that scales portion and composition changes against k-based baselines.

    Returns (E, E_port, E_comp, portion_pct, comp_pct, portion_base_pct, comp_base_pct)
    where E, E_port, E_comp are fractions in [0,1] and the *_pct values are percentages.
    """
    gm = np.asarray(gm, float); rm = np.asarray(rm, float)
    r_items = (rm > 0); g_items = (gm > 0)
    n = int(np.sum(r_items)) or 1

    # Portion change (% of total real grams)
    l1 = float(np.sum(np.abs(gm - rm)))
    base = float(np.sum(rm)) or 1.0
    portion_pct = 100.0 * (l1 / base)
    # Baseline: expected grams touched by k substitutions (capped at 200%)
    portion_base_frac = min(2.0 * float(max(k, 0)) / float(n), 2.0)
    portion_base_pct = 100.0 * portion_base_frac
    E_port = min(portion_pct / max(portion_base_pct, 1e-9), 1.0)

    # Composition change via Jaccard distance
    inter = float(np.sum(r_items & g_items))
    uni = float(np.sum(r_items | g_items)) or 1.0
    jacc = 1.0 - inter / uni
    comp_pct = 100.0 * jacc
    # Baseline Jaccard for k subs on n items (capped at 1)
    comp_base = min(2.0 * float(max(k, 0)) / float(n + max(k, 0)), 1.0)
    comp_base_pct = 100.0 * comp_base
    E_comp = min(comp_pct / max(comp_base_pct, 1e-9), 1.0)

    E = float(alpha) * float(np.clip(E_port, 0.0, 1.0)) + (1.0 - float(alpha)) * float(np.clip(E_comp, 0.0, 1.0))
    return float(np.clip(E, 0.0, 1.0)), float(np.clip(E_port, 0.0, 1.0)), float(np.clip(E_comp, 0.0, 1.0)), float(portion_pct), float(comp_pct), float(portion_base_pct), float(comp_base_pct)


def _load_meals_and_features(data_dir: Path) -> dict:
    data = {}
    try:
        import json as _json
        data['gen_by_type'] = pickle.load(open(data_dir / 'generated_meals_by_type.pkl', 'rb'))
        data['real_by_type'] = pickle.load(open(data_dir / 'real_meals_by_type.pkl', 'rb'))
        data['W'] = np.load(data_dir / 'nutrient_matrix.npy')
        data['nutrient_names'] = _json.load(open(data_dir / 'nutrient_names.json','r'))
    except Exception as e:
        raise FileNotFoundError(f"Missing meal/nutrient data in {data_dir}: {e}")
    return data

def _find_data_exports(start: Path) -> Path:
    """Nearest `data_exports` folder at or above `start`; runs keep it at the run root."""
    start = Path(start)
    for anc in [start, *start.parents][:5]:
        if (anc / 'data_exports').is_dir():
            return anc / 'data_exports'
    return start.parent / 'data_exports'

def _read_food_category_map(data_dir: Path) -> dict:
    # Try explicit food_to_group.json
    try:
        import json as _json
        p = data_dir / 'food_to_group.json'
        if p.exists():
            return _json.load(open(p, 'r'))
    except Exception:
        pass
    return {}


def figure_category_heatmap(
    input_dir: Path,
    out_dir: Path,
    data_dir: Optional[Path] = None,
    theta: float = 1.0,
    *,
    use_hclust: bool = True,
    same_axis_order: bool = True,
) -> None:
    """Category substitution plots for one run folder (selected winners only).

    Writes figure_category_heatmap (P(added category | removed category), rows sum to 1)
    and figure_category_add_vs_remove_scatter to out_dir. When input_dir/out_dir follow the
    Makefile naming (tradeoff_theta_<θ>_k<k>, figures_theta<θ>_k<k>) and at least two sibling
    k runs exist, also writes figure_category_scatter_plus_heatmap_k123 to
    figures_theta<θ>_k123/. Beverages and uncategorised foods ('other') are left out.
    """
    try:
        import seaborn as sns  # type: ignore
    except Exception:
        print("seaborn not installed; skipping category heatmap.")
        return
    df = _matches_df(input_dir, winners_only=True)
    # Debug: overall sample counts per k
    try:
        kcol = 'k_substitution' if 'k_substitution' in df.columns else ('k' if 'k' in df.columns else None)
        if kcol is not None:
            vc = pd.to_numeric(df[kcol], errors='coerce').value_counts().sort_index()
            print("[debug k_from_input] samples by k:", ", ".join([f"k={int(k)}: {int(n)}" for k, n in vc.items()]))
    except Exception:
        pass
    # winners_only already applied above via _matches_df
    data_dir = data_dir or _find_data_exports(input_dir)
    cat_map = _read_food_category_map(data_dir)
    if not cat_map:
        raise RuntimeError("Missing WWEIA mapping: food_to_group.json not found or empty. Cannot build category heatmap without WWEIA Main Categories.")

    def cat_of(code: str) -> str:
        # Normalize code tokens like '11100000.0' -> '11100000'
        _s = str(code)
        try:
            if '.' in _s:
                _s = _s.split('.')[0]
            _s = ''.join(ch for ch in _s if ch.isdigit())
        except Exception:
            pass
        c = _s.zfill(8)
        cat = cat_map.get(c)
        if cat:
            return str(cat)
        # Strict WWEIA-only categorization: fallback to canonical 'other'
        return "other"

    rows = []
    omit_cats = {"other", "beverages"}
    for _, r in df.iterrows():
        rem = str(r.get('removed_foods', '') or '')
        add = str(r.get('added_foods', '') or '')
        if not rem and not add:
            continue
        r_codes = [x for x in rem.split(';') if x]
        a_codes = [x for x in add.split(';') if x]
        r_cats = list({cat_of(x) for x in r_codes})
        a_cats = list({cat_of(x) for x in a_codes})
        # Drop omitted categories from consideration
        r_cats = [c for c in r_cats if c not in omit_cats]
        a_cats = [c for c in a_cats if c not in omit_cats]
        for rc in r_cats:
            for ac in a_cats:
                rows.append({
                    'removed_cat': rc,
                    'added_cat': ac,
                    'dRDI_pp': float(100.0 * r.get('dRDI', r.get('improvement', np.nan))),
                    'dCost_pct': float(r.get('dCost', r.get('delta_cost', np.nan)) / r.get('real_cost', np.nan) * 100.0) if (pd.notna(r.get('real_cost', np.nan)) and float(r.get('real_cost', 0)) != 0 and pd.notna(r.get('dCost', r.get('delta_cost', np.nan)))) else np.nan
                })
    if not rows:
        return
    md = pd.DataFrame(rows)

    # ---- Scatterplot (pairwise-marginal): P(remove) vs P(add) per category from same pair counts as heatmap ----
    try:
        counts_pw = md.groupby(['removed_cat','added_cat']).size().reset_index(name='n')
        if not counts_pw.empty:
            total_pairs = float(counts_pw['n'].sum()) or 1.0
            row_sums = counts_pw.groupby('removed_cat')['n'].sum()
            col_sums = counts_pw.groupby('added_cat')['n'].sum()
            cats_all = sorted(set(row_sums.index.tolist()) | set(col_sums.index.tolist()))
            # Pretty display names for categories
            def _pretty_cat(c: str) -> str:
                mapping = {
                    'snacks_sweets': 'Snacks',
                    'fats_oils': 'Fats & oils',
                    'condiments_sauces': 'Condiments & sauces',
                    'mixed_dishes': 'Mixed dishes',
                    'coffee_tea': 'Coffee & tea',
                    'plain_water': 'Water',
                    'diet_beverages': 'Diet beverages',
                }
                c0 = str(c)
                if c0 in mapping:
                    return mapping[c0]
                return c0.replace('_', ' ').capitalize() if c0 else c0
            _sc_rows = []
            for c in cats_all:
                p_remove = float(row_sums.get(c, 0.0)) / total_pairs
                p_add = float(col_sums.get(c, 0.0)) / total_pairs
                _sc_rows.append({'category': c, 'category_pretty': _pretty_cat(c), 'p_remove': p_remove, 'p_add': p_add,
                                 'row_sum': int(row_sums.get(c, 0.0) or 0), 'col_sum': int(col_sums.get(c, 0.0) or 0)})
            scdf = pd.DataFrame(_sc_rows)
            # Point sizes by marginal mass (row+col sums); cap at p99 for robustness
            scdf['mass'] = scdf['row_sum'].astype(float) + scdf['col_sum'].astype(float)
            m99 = float(np.nanpercentile(scdf['mass'].to_numpy(), 99)) if len(scdf) else 1.0
            m99 = m99 or 1.0
            sizes = 30.0 + 170.0 * (np.clip(scdf['mass'].to_numpy() / m99, 0.0, 1.0))

            # Axes zoom to p99
            px = float(np.nanpercentile(scdf['p_add'].to_numpy(), 99)) if len(scdf) else 0.0
            py = float(np.nanpercentile(scdf['p_remove'].to_numpy(), 99)) if len(scdf) else 0.0
            lim = float(max(0.05, 1.05 * max(px, py)))

            plt.figure(figsize=(7, 6))
            ax_sc = sns.scatterplot(data=scdf, x='p_add', y='p_remove', hue='category_pretty', palette='tab20', s=sizes, edgecolor='white', linewidth=0.5)
            # Ensure all points fit; axes start at -0.01 (ticks still from 0.00) and 0.03 padding
            xmax = float(np.nanmax(scdf['p_add'].to_numpy())) if len(scdf) else 0.0
            ymax = float(np.nanmax(scdf['p_remove'].to_numpy())) if len(scdf) else 0.0
            lim_use = max(lim, xmax, ymax) + 0.03
            # Diagonal reference spanning full axis
            ax_sc.plot([0, lim_use], [0, lim_use], ls='--', lw=1.0, c='#999999')
            ax_sc.set_xlim(-0.01, lim_use)
            ax_sc.set_ylim(-0.01, lim_use)
            try:
                ax_sc.set_xticks(np.arange(0.0, lim_use + 0.0001, 0.10))
                ax_sc.set_yticks(np.arange(0.0, lim_use + 0.0001, 0.05))
            except Exception:
                pass
            ax_sc.set_aspect('equal', adjustable='box')
            ax_sc.set_xlabel('Probability of food added')
            ax_sc.set_ylabel('Probability of food removed')
            ax_sc.set_title('Category marginals over removed–added pairs')
            ax_sc.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
            # Prefer a clean legend-only presentation (no per-point labels)
            fig = plt.gcf()
            try:
                fig.subplots_adjust(left=0.12, right=0.78, top=0.92, bottom=0.12)
            except Exception:
                pass
            _ensure_dir(out_dir)
            plt.savefig(out_dir / 'figure_category_add_vs_remove_scatter.png', dpi=200, bbox_inches='tight', pad_inches=0.08)
            plt.savefig(out_dir / 'figure_category_add_vs_remove_scatter.svg', bbox_inches='tight', pad_inches=0.08)
            plt.close()
    except Exception as _e:
        print(f"[warn] Failed to render category add-vs-remove scatter: {_e}")

    # ---- Optional triptych: side-by-side panels for k=1,2,3 for same θ ----
    try:
        # Detect sibling run folders and a common figures directory name
        import re
        in_name = str(input_dir.name)
        m = re.match(r'^(tradeoff_theta_[0-9p\.]+)_k[123](?:run)?$', in_name)
        m_fig = re.match(r'^(figures_theta[^_]+)_k[123]$', str(out_dir.name))

        k_to_dir = {}
        if m and m_fig:
            base_prefix = m.group(1)  # e.g., tradeoff_theta_0p5
            base_runs_dir = input_dir.parent
            k_to_dir = {k: (base_runs_dir / f"{base_prefix}_k{k}") for k in (1,2,3)}

        # Ensure at least two exist to warrant a triptych
        existings = [k for k, p in k_to_dir.items() if p.exists() and (p / 'substitution_matches.csv').exists()]
        if len(existings) >= 2:
                # Build per-k scatter dataframes using pairwise-marginal counting (to match heatmaps)
                def _scatter_df_for_dir(run_dir: Path) -> pd.DataFrame:
                    dfk = _matches_df(run_dir, winners_only=True)
                    rows_h = []
                    for _, rr in dfk.iterrows():
                        rem = str(rr.get('removed_foods', '') or '')
                        add = str(rr.get('added_foods', '') or '')
                        if not rem and not add:
                            continue
                        r_codes = [x for x in rem.split(';') if x]
                        a_codes = [x for x in add.split(';') if x]
                        r_cats = list({cat_of(x) for x in r_codes})
                        a_cats = list({cat_of(x) for x in a_codes})
                        r_cats = [c for c in r_cats if c not in omit_cats]
                        a_cats = [c for c in a_cats if c not in omit_cats]
                        for rc in r_cats:
                            for ac in a_cats:
                                rows_h.append({'removed_cat': rc, 'added_cat': ac})
                    if not rows_h:
                        return pd.DataFrame(columns=['category','p_add','p_remove'])
                    mdh = pd.DataFrame(rows_h)
                    counts = mdh.groupby(['removed_cat','added_cat']).size().reset_index(name='n')
                    total_pairs = float(counts['n'].sum()) or 1.0
                    row_sums = counts.groupby('removed_cat')['n'].sum()
                    col_sums = counts.groupby('added_cat')['n'].sum()
                    cats_all_k = sorted(set(row_sums.index.tolist()) | set(col_sums.index.tolist()))
                    rows_k = []
                    for c in cats_all_k:
                        rows_k.append({
                            'category': c,
                            'p_remove': float(row_sums.get(c, 0.0)) / total_pairs,
                            'p_add': float(col_sums.get(c, 0.0)) / total_pairs,
                            'row_sum': int(row_sums.get(c, 0.0) or 0),
                            'col_sum': int(col_sums.get(c, 0.0) or 0),
                        })
                    return pd.DataFrame(rows_k)

                data_by_k = {k: _scatter_df_for_dir(p) for k, p in k_to_dir.items() if k in existings}
                if data_by_k:
                    # Union of categories for consistent coloring
                    all_cats = sorted({c for dfk in data_by_k.values() for c in dfk['category'].tolist()})
                    ncat = len(all_cats)
                    try:
                        import seaborn as sns  # type: ignore
                        pal = sns.color_palette('tab20', ncat) if ncat <= 20 else sns.color_palette('hls', ncat)
                    except Exception:
                        pal = [(0.12,0.47,0.71)] * ncat
                    cat2color = {c: pal[i % len(pal)] for i, c in enumerate(all_cats)}

                    # Build per-k heatmap pivots (row-normalized probabilities)
                    def _pivot_for_dir(run_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float, list[str]]:
                        dfk = _matches_df(run_dir, winners_only=True)
                        rows_h = []
                        for _, rr in dfk.iterrows():
                            rem = str(rr.get('removed_foods', '') or '')
                            add = str(rr.get('added_foods', '') or '')
                            if not rem and not add:
                                continue
                            r_codes = [x for x in rem.split(';') if x]
                            a_codes = [x for x in add.split(';') if x]
                            r_cats = list({cat_of(x) for x in r_codes})
                            a_cats = list({cat_of(x) for x in a_codes})
                            r_cats = [c for c in r_cats if c not in omit_cats]
                            a_cats = [c for c in a_cats if c not in omit_cats]
                            for rc in r_cats:
                                for ac in a_cats:
                                    rows_h.append({'removed_cat': rc, 'added_cat': ac})
                        if not rows_h:
                            return pd.DataFrame(), np.array([]), np.array([]), 0.0, []
                        mdh = pd.DataFrame(rows_h)
                        counts = mdh.groupby(['removed_cat','added_cat']).size().reset_index(name='n')
                        row_tot = counts.groupby('removed_cat')['n'].transform('sum')
                        counts['p_row'] = counts['n'] / row_tot.replace(0, np.nan)
                        pivot = counts.pivot(index='removed_cat', columns='added_cat', values='p_row')
                        total_pairs = float(counts['n'].sum()) or 1.0
                        rsum = counts.groupby('removed_cat')['n'].sum()
                        csum = counts.groupby('added_cat')['n'].sum()
                        cats = sorted(set(pivot.index.tolist()) | set(pivot.columns.tolist()))
                        rarr = np.array([float(rsum.get(c, 0.0)) / total_pairs for c in cats], float)
                        carr = np.array([float(csum.get(c, 0.0)) / total_pairs for c in cats], float)
                        return pivot, rarr, carr, total_pairs, cats

                    piv_and_marg_by_k = {k: _pivot_for_dir(p) for k, p in k_to_dir.items() if k in existings}
                    piv_by_k = {k: piv_and_marg_by_k[k][0] for k in piv_and_marg_by_k}
                    marg_by_k = {k: {'row': piv_and_marg_by_k[k][1], 'col': piv_and_marg_by_k[k][2], 'cats': piv_and_marg_by_k[k][4]} for k in piv_and_marg_by_k}
                    # Build common category order and global vmax for consistent scaling
                    cats_union = sorted({c for piv in piv_by_k.values() for c in (list(piv.index) + list(piv.columns))}) if piv_by_k else []

                    # Determine hierarchical clustering order from symmetrized average across k
                    order = list(cats_union)
                    if cats_union:
                        try:
                            import numpy as _np  # type: ignore
                            from scipy.cluster.hierarchy import linkage, leaves_list  # type: ignore
                            from scipy.spatial.distance import pdist  # type: ignore
                            mats = []
                            for _k, _piv in piv_by_k.items():
                                P = _piv.reindex(index=cats_union, columns=cats_union).fillna(0.0).values
                                S = 0.5 * (P + P.T)
                                mats.append(S)
                            if mats:
                                S_avg = _np.mean(_np.stack(mats, axis=0), axis=0)
                                D = pdist(S_avg, metric='euclidean')
                                if not _np.allclose(D, 0.0):
                                    Z = linkage(D, method='average')
                                    leaves = leaves_list(Z)
                                    order = [cats_union[i] for i in leaves]
                        except Exception:
                            order = list(cats_union)

                    for k in list(piv_by_k.keys()):
                        piv_by_k[k] = piv_by_k[k].reindex(index=order, columns=order).fillna(0.0) if order else piv_by_k[k].fillna(0.0)
                        # Reindex marginals into order
                        if k in marg_by_k and order:
                            rmap = {c: float(v) for c, v in zip(marg_by_k[k]['cats'], marg_by_k[k]['row'])}
                            cmap = {c: float(v) for c, v in zip(marg_by_k[k]['cats'], marg_by_k[k]['col'])}
                            marg_by_k[k]['row'] = np.array([rmap.get(c, 0.0) for c in order], float)
                            marg_by_k[k]['col'] = np.array([cmap.get(c, 0.0) for c in order], float)
                            marg_by_k[k]['cats'] = list(order)
                    vmax = 0.0
                    for piv in piv_by_k.values():
                        try:
                            vmax = max(vmax, float(np.nanmax(piv.values)))
                        except Exception:
                            pass
                    if not np.isfinite(vmax) or vmax <= 0:
                        vmax = 1.0

                    # Create 3x2 grid: left column heatmaps, right column scatters (rows=k)
                    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=False, sharey=False, constrained_layout=False)
                    # Attempt to detect theta per k for labeling
                    def _theta_for_dir(d: Path) -> str:
                        try:
                            tfile = d / 'theta.txt'
                            if tfile.exists():
                                tval = tfile.read_text().strip()
                                if tval:
                                    return tval
                            import re as _re
                            m = _re.search(r'tradeoff_theta_([0-9p\.]+)_k[123]', str(d.name))
                            if m:
                                return m.group(1).replace('p', '.')
                        except Exception:
                            pass
                        return ''
                    theta_label_by_k = {k: _theta_for_dir(p) for k, p in k_to_dir.items()}
                    labels_by_k = {1: '1-hop Substitution Patterns', 2: '2-hop Substitution Patterns', 3: '3-hop Substitution Patterns'}
                    for ridx, k in enumerate((1,2,3)):
                        # Right column: scatter
                        ax_sc = axes[ridx, 1]
                        dfk = data_by_k.get(k, pd.DataFrame(columns=['category','p_add','p_remove','row_sum','col_sum']))
                        if not dfk.empty:
                            colors = [cat2color.get(c, (0.12,0.47,0.71)) for c in dfk['category']]
                            mass = dfk['row_sum'].astype(float) + dfk['col_sum'].astype(float)
                            m99 = float(np.nanpercentile(mass.to_numpy(), 99)) if len(mass) else 1.0
                            m99 = m99 or 1.0
                            sizes = 25.0 + 160.0 * (np.clip(mass.to_numpy() / m99, 0.0, 1.0))
                            ax_sc.scatter(dfk['p_add'], dfk['p_remove'], s=sizes, c=colors, edgecolors='white', linewidths=0.5)
                            # Ensure all points fit; y_min = -0.03 and ticks at 0.05
                            xmax = float(np.nanmax(dfk['p_add'].to_numpy())) if len(dfk) else 0.0
                            ymax = float(np.nanmax(dfk['p_remove'].to_numpy())) if len(dfk) else 0.0
                            px = float(np.nanpercentile(dfk['p_add'].to_numpy(), 99)) if len(dfk) else 0.0
                            py = float(np.nanpercentile(dfk['p_remove'].to_numpy(), 99)) if len(dfk) else 0.0
                            lim = float(max(0.05, 1.05 * max(px, py)))
                            lim_use = max(lim, xmax, ymax) + 0.03
                            # y=x reference spanning full axis
                            ax_sc.plot([0, lim_use], [0, lim_use], ls='--', lw=1.0, c='#aaaaaa')
                            ax_sc.set_xlim(-0.01, lim_use)
                            ax_sc.set_ylim(-0.01, lim_use)
                            try:
                                ax_sc.set_xticks(np.arange(0.0, lim_use + 0.0001, 0.10))
                                ax_sc.set_yticks(np.arange(0.0, lim_use + 0.0001, 0.10))
                            except Exception:
                                pass
                            ax_sc.set_aspect('equal', adjustable='box')
                            # Legend-only for clarity; avoid per-point labels
                            try:
                                fig = plt.gcf(); fig.subplots_adjust(left=0.10, right=0.80, top=0.90, bottom=0.12)
                            except Exception:
                                pass
                            # Do not annotate points; rely on shared legend for professionalism
                        ax_sc.set_title(labels_by_k[k], fontsize=12)
                        ax_sc.set_ylabel('Probability of food removed', fontsize=12)
                        ax_sc.set_xlabel('Probability of food added', fontsize=12)
                        ax_sc.grid(alpha=0.15, linestyle=':')

                        # Left column: heatmap P(add|remove)
                        ax_hm = axes[ridx, 0]
                        piv = piv_by_k.get(k, pd.DataFrame())
                        if piv is not None and not piv.empty:
                            try:
                                # Dendrogram ordering and labels; draw dendrograms and colorbar in overlay axes
                                import numpy as _np
                                from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
                                from matplotlib.colors import Normalize
                                from matplotlib.cm import ScalarMappable
                                cats_here = sorted(set(piv.index.tolist()) | set(piv.columns.tolist()))
                                P_sq = piv.reindex(index=cats_here, columns=cats_here).fillna(0.0)
                                S = 0.5 * (P_sq.values + P_sq.values.T)
                                if S.size and not _np.allclose(S, 0.0):
                                    Z = linkage(S, method='average')
                                    order_idx = leaves_list(Z)
                                    order_names = [cats_here[i] for i in order_idx]
                                else:
                                    order_names = cats_here
                                piv_ord = piv.reindex(index=order_names, columns=order_names).fillna(0.0)
                                vmax_local = float(_np.nanmax(piv_ord.values)) if _np.isfinite(_np.nanmax(piv_ord.values)) and _np.nanmax(piv_ord.values) > 0 else 1.0
                                piv_disp = piv_ord.copy(); piv_disp.index = [_pretty_cat(x) for x in piv_ord.index]; piv_disp.columns = [_pretty_cat(x) for x in piv_ord.columns]
                                # Heatmap (square) in main axis
                                # Use specific 3-color gradients for each k-level heatmap (white=0, dark=high)
                                k_gradients = {
                                    1: ['#ffffff', '#b4e1d3', '#68c3a6'],  # teal gradient (inverted)
                                    2: ['#ffffff', '#fec6b1', '#fc8d62'],  # orange gradient (inverted, updated)
                                    3: ['#ffffff', '#c8d1e6', '#90a3cc']   # blue gradient (inverted)
                                }
                                from matplotlib.colors import LinearSegmentedColormap
                                colors = k_gradients[k]
                                n_bins = 100
                                custom_cmap = LinearSegmentedColormap.from_list(f'custom_{k}', colors, N=n_bins)
                                sns.heatmap(
                                    piv_disp,
                                    ax=ax_hm,
                                    cmap=custom_cmap, vmin=0.0, vmax=vmax_local,
                                    linewidths=0.3, linecolor='#eeeeee',
                                    cbar=False,
                                    square=True,
                                )
                                ax_hm.set_xticks(_np.arange(len(piv_disp.columns))+0.5)
                                ax_hm.set_yticks(_np.arange(len(piv_disp.index))+0.5)
                                ax_hm.set_xticklabels(list(piv_disp.columns), rotation=45, ha='right', fontsize=11)
                                ax_hm.set_yticklabels(list(piv_disp.index), rotation=0, fontsize=11)
                                # Overlay dendrogram axes without shrinking the heatmap
                                fig_ref = ax_hm.figure
                                pos = ax_hm.get_position()
                                top_h = 0.10 * pos.height
                                left_w = 0.10 * pos.width
                                ax_top = fig_ref.add_axes([pos.x0, pos.y1, pos.width, top_h])
                                # Place dendrogram on the right side instead of left
                                ax_left = fig_ref.add_axes([pos.x1, pos.y0, left_w, pos.height])
                                ax_top.set_axis_off(); ax_left.set_axis_off()
                                if S.size and not _np.allclose(S, 0.0):
                                    dendrogram(Z, ax=ax_top, no_labels=True, color_threshold=0, above_threshold_color='black', link_color_func=lambda k: 'black')
                                    dendrogram(Z, ax=ax_left, orientation='right', no_labels=True, color_threshold=0, above_threshold_color='black', link_color_func=lambda k: 'black')
                                # Per-panel colorbar with local scale
                                cax_w = 0.015
                                cax = fig_ref.add_axes([pos.x1 + left_w + 0.005, pos.y0, cax_w, pos.height])
                                sm = ScalarMappable(norm=Normalize(0.0, vmax_local), cmap=custom_cmap); sm.set_array([])
                                cb = fig_ref.colorbar(sm, cax=cax)
                                cb.set_label('Substitution Probability')
                                # Set tick intervals to 0.1
                                cb.set_ticks(np.arange(0.0, vmax_local + 0.05, 0.1))
                            except Exception:
                                pass
                        # Append theta label if known
                        tlabel = theta_label_by_k.get(k) or ''
                        if tlabel:
                            ax_hm.set_title(f"θ={tlabel}", fontsize=11, loc='left')
                        ax_hm.set_ylabel('Removed category', fontsize=12)
                        ax_hm.set_xlabel('Added category', fontsize=12)

                    # Shared legends on right: scatter legend only (no shared colorbar)
                    try:
                        fig.subplots_adjust(left=0.07, right=0.80, top=0.95, bottom=0.08, wspace=0.35, hspace=0.45)
                        # Scatter legend
                        from matplotlib.lines import Line2D
                        leg_ax = fig.add_axes([0.82, 0.10, 0.16, 0.40]); leg_ax.axis('off')
                        pretty_names = [_pretty_cat(c) for c in all_cats]
                        handles = [Line2D([0],[0], marker='o', linestyle='', markersize=6,
                                           markerfacecolor=cat2color[c], markeredgecolor='white', markeredgewidth=0.5)
                                   for c in all_cats]
                        leg = leg_ax.legend(handles, pretty_names, title='Food Categories', loc='center left', frameon=True, borderaxespad=0.0, handletextpad=0.4, labelspacing=0.4)
                        try:
                            leg.get_frame().set_edgecolor('#444444'); leg.get_frame().set_linewidth(0.8)
                        except Exception:
                            pass
                    except Exception:
                        plt.tight_layout()
                    # Where to save the combined triptych
                    if m_fig:
                        triptych_dir = out_dir.parent / f"{m_fig.group(1)}_k123"
                    else:
                        triptych_dir = out_dir
                    _ensure_dir(triptych_dir)
                    fig.savefig(triptych_dir / 'figure_category_scatter_plus_heatmap_k123.png', dpi=200, bbox_inches='tight')
                    fig.savefig(triptych_dir / 'figure_category_scatter_plus_heatmap_k123.svg', bbox_inches='tight')
                    plt.close(fig)
    except Exception as _e:
        print(f"[warn] Failed to render triptych category scatter: {_e}")
    # Row-normalized probabilities per removed category
    counts = md.groupby(['removed_cat','added_cat']).size().reset_index(name='n')
    # Ensure we do not plot omitted categories
    if not counts.empty:
        counts = counts[~counts['removed_cat'].isin(omit_cats) & ~counts['added_cat'].isin(omit_cats)]
    row_tot = counts.groupby('removed_cat')['n'].transform('sum')
    counts['p_row'] = counts['n'] / row_tot.replace(0, np.nan)

    # Build matrix
    pivot = counts.pivot(index='removed_cat', columns='added_cat', values='p_row')

    # Common ordering: hierarchical clustering on symmetrized probability matrix
    cats = sorted(set(pivot.index.tolist()) | set(pivot.columns.tolist()))
    order = cats
    if use_hclust:
        try:
            import numpy as _np
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import pdist
            P_sq = pivot.reindex(index=cats, columns=cats).fillna(0.0)
            S = 0.5 * (P_sq.values + P_sq.values.T)
            # Guard for degenerate distances
            D = pdist(S, metric='euclidean')
            if _np.allclose(D, 0.0):
                order = cats
            else:
                Z = linkage(D, method='average')
                order = [cats[i] for i in leaves_list(Z)]
        except Exception:
            order = cats

    if same_axis_order:
        rem_order = order
        add_order = order
        # Diagonal alignment: reorder columns to maximize diagonal mass vs rem_order
        try:
            import numpy as _np
            from scipy.optimize import linear_sum_assignment
            cats_r = list(rem_order)
            cats_c = sorted(set(pivot.columns.tolist()))
            M = pivot.reindex(index=cats_r, columns=cats_c).fillna(0.0).values
            # Hungarian on negative probabilities to maximize sum
            cost = -M
            ri, ci = linear_sum_assignment(cost)
            chosen_cols = [cats_c[j] for j in ci]
            # Append any leftover columns not chosen, in original order
            leftover = [c for c in pivot.columns.tolist() if c not in chosen_cols]
            add_order = chosen_cols + leftover
        except Exception:
            # Fallback: keep clustering order
            add_order = order
    else:
        rem_order = sorted(pivot.index.tolist())
        add_order = sorted(pivot.columns.tolist())

    pivot = pivot.reindex(index=rem_order, columns=add_order)

    finite_vals = pivot.values[np.isfinite(pivot.values)]
    if finite_vals.size == 0:
        print("No finite values for category heatmap; skipping.")
        return

    # Render with dendrograms when clustering is enabled, otherwise a plain heatmap
    try:
        if use_hclust:
            # Build square matrix over common category set for clean dendrograms
            cats = sorted(set(pivot.index.tolist()) | set(pivot.columns.tolist()))
            P_sq = pivot.reindex(index=cats, columns=cats).fillna(0.0)
            # Use the same linkage for rows/cols from the symmetrized matrix
            from scipy.cluster.hierarchy import linkage
            from scipy.spatial.distance import pdist
            import numpy as _np
            S = 0.5 * (P_sq.values + P_sq.values.T)
            D = pdist(S, metric='euclidean')
            if _np.allclose(D, 0.0):
                P_sq_disp = P_sq.copy()
                P_sq_disp.index = [_pretty_cat(x) for x in P_sq_disp.index]
                P_sq_disp.columns = [_pretty_cat(x) for x in P_sq_disp.columns]
                cg = sns.clustermap(
                    P_sq_disp,
                    cmap='Blues', vmin=0.0, vmax=float(_np.nanmax(P_sq.values)),
                    linewidths=0.3, linecolor='#eeeeee',
                    row_cluster=False, col_cluster=False,
                    cbar_kws={'label':'Probability P(add | remove) (rows sum to 1)'}
                )
            else:
                Z = linkage(D, method='average')
                P_sq_disp = P_sq.copy()
                P_sq_disp.index = [_pretty_cat(x) for x in P_sq_disp.index]
                P_sq_disp.columns = [_pretty_cat(x) for x in P_sq_disp.columns]
                cg = sns.clustermap(
                    P_sq_disp,
                    cmap='Blues', vmin=0.0, vmax=float(_np.nanmax(P_sq.values)),
                    linewidths=0.3, linecolor='#eeeeee',
                    row_linkage=Z, col_linkage=Z,
                    cbar_kws={'label':'Probability P(add | remove) (rows sum to 1)'}
                )
            cg.ax_heatmap.set_xlabel('Added category')
            cg.ax_heatmap.set_ylabel('Removed category')
            cg.ax_heatmap.set_title('Category substitution heatmap — Probability of food added (given removal)')
            _ensure_dir(out_dir)
            cg.fig.savefig(out_dir / 'figure_category_heatmap.png', dpi=200)
            cg.fig.savefig(out_dir / 'figure_category_heatmap.svg')
            plt.close(cg.fig)
            return
    except Exception as e:
        print(f"[warn] Dendrogram rendering failed; falling back to plain heatmap: {e}")

    pivot_disp = pivot.copy()
    pivot_disp.index = [_pretty_cat(x) for x in pivot_disp.index]
    pivot_disp.columns = [_pretty_cat(x) for x in pivot_disp.columns]
    plt.figure(figsize=(13,10))
    sns.heatmap(
        pivot_disp,
        cmap='Blues',
        vmin=0.0,
        vmax=1.0,
        linewidths=0.3,
        linecolor='#eeeeee',
        cbar_kws={'label':'Probability P(add | remove) (rows sum to 1)'}
    )
    plt.xlabel('Added category'); plt.ylabel('Removed category')
    plt.title('Category substitution heatmap — Probability of food added (given removal)')
    plt.tight_layout()
    _ensure_dir(out_dir)
    plt.savefig(out_dir / 'figure_category_heatmap.png', dpi=200)
    plt.savefig(out_dir / 'figure_category_heatmap.svg')
    plt.close()


def _load_row_level_df(run_dir: Path) -> pd.DataFrame:
    """Prefer matches (has k_substitution, dRDI_pct, dCost_pct); fallback to examples if needed."""
    matches = Path(run_dir) / 'substitution_matches.csv'
    if matches.exists():
        try:
            df = pd.read_csv(matches)
            # Ensure percent columns exist
            if 'dRDI_pct' not in df.columns:
                df['dRDI_pct'] = pd.to_numeric(df.get('dRDI_pct', 100.0 * df.get('dRDI', df.get('improvement', np.nan))), errors='coerce')
            if 'dCost_pct' not in df.columns and {'real_cost','dCost'}.issubset(df.columns):
                with np.errstate(divide='ignore', invalid='ignore'):
                    df['dCost_pct'] = 100.0 * (pd.to_numeric(df['dCost'], errors='coerce') / pd.to_numeric(df['real_cost'], errors='coerce'))
            return df
        except Exception:
            pass
    examples = Path(run_dir) / 'substitution_examples_detailed.csv'
    if examples.exists():
        try:
            df = pd.read_csv(examples)
            if 'dRDI_pct' not in df.columns:
                df['dRDI_pct'] = pd.to_numeric(df.get('dRDI_pct', 100.0 * df.get('dRDI', df.get('improvement', np.nan))), errors='coerce')
            if 'dCost_pct' not in df.columns and {'real_cost','dCost'}.issubset(df.columns):
                with np.errstate(divide='ignore', invalid='ignore'):
                    df['dCost_pct'] = 100.0 * (pd.to_numeric(df['dCost'], errors='coerce') / pd.to_numeric(df['real_cost'], errors='coerce'))
            return df
        except Exception:
            pass
    # Last resort
    df = _read_matches(Path(run_dir))
    if 'dRDI_pct' not in df.columns:
        df['dRDI_pct'] = pd.to_numeric(df.get('dRDI_pct', 100.0 * df.get('dRDI', df.get('improvement', np.nan))), errors='coerce')
    if 'dCost_pct' not in df.columns and {'real_cost','dCost'}.issubset(df.columns):
        with np.errstate(divide='ignore', invalid='ignore'):
            df['dCost_pct'] = 100.0 * (pd.to_numeric(df['dCost'], errors='coerce') / pd.to_numeric(df['real_cost'], errors='coerce'))
    return df


def _get_numeric_series(df: pd.DataFrame, candidates: list[str], scale_100_if_fraction: bool = False) -> np.ndarray:
    for c in candidates:
        if c in df.columns:
            arr = pd.to_numeric(df[c], errors='coerce').to_numpy()
            if scale_100_if_fraction and np.nanmax(np.abs(arr)) <= 1.5:
                arr = 100.0 * arr
            return arr
    return np.full(len(df), np.nan)


def _aggregate_by_k(df: pd.DataFrame, winners_only: bool = False, ks: tuple[int, ...] = (1,2,3), bootstraps: int = 300, rng_seed: int = 1234) -> dict[int, dict[str, float]]:
    # Require k_substitution only
    if 'k_substitution' not in df.columns:
        raise ValueError("Expected 'k_substitution' column.")
    kcol = 'k_substitution'
    # Health as percent improvement; Cost as percent change
    H = _get_numeric_series(df, ['dRDI_pct', 'delta_rdi_pct'], scale_100_if_fraction=True)
    C = _get_numeric_series(df, ['dCost_pct', 'delta_cost_pct'], scale_100_if_fraction=True)
    # Filter to meaningful rows
    m = np.isfinite(H) & np.isfinite(C)
    df = df[m].copy()
    df['H_pct'] = H[m]
    df['C_pct'] = C[m]
    if winners_only:
        # Use policy winners to reflect θ (if available); otherwise fall back to dominance-like filter
        if 'selected' in df.columns:
            df = df[df['selected'] == True]
        else:
            df = df[(df['H_pct'] >= 0) & (df['C_pct'] <= 0)]
    results: dict[int, dict[str, float]] = {}
    for k in ks:
        sub = df[pd.to_numeric(df[kcol], errors='coerce') == k]
        if sub.empty:
            continue
        h = sub['H_pct'].to_numpy()
        c = sub['C_pct'].to_numpy()
        h_med = float(np.nanmedian(h))
        c_med = float(np.nanmedian(c))
        # Bootstrap 95% CIs on medians
        if bootstraps and len(sub) > 1:
            rng = np.random.default_rng(rng_seed + 10 * int(k))
            hb = []; cb = []
            n = len(sub)
            for _ in range(int(bootstraps)):
                idx = rng.integers(0, n, n)
                hb.append(np.nanmedian(h[idx]))
                cb.append(np.nanmedian(c[idx]))
            h_lo, h_hi = np.nanpercentile(hb, [2.5, 97.5])
            c_lo, c_hi = np.nanpercentile(cb, [2.5, 97.5])
        else:
            h_lo = h_hi = h_med
            c_lo = c_hi = c_med
        results[k] = {
            'h_med': h_med,
            'h_lo': float(h_lo),
            'h_hi': float(h_hi),
            'c_med': c_med,
            'c_lo': float(c_lo),
            'c_hi': float(c_hi),
        }
    return results


def figure_cost_benefit_points_by_theta_and_k(theta_to_dirs: dict[float, Path], out_dir: Path, ks: tuple[int, ...] = (1,2,3), winners_only: bool = False, bootstraps: int = 300, y_mode: str = 'savings') -> None:
    """Figure 4B: one operating point per (θ, k), the medians of ΔRDI% (x) and of a cost
    metric (y), joined across θ for each k, with bootstrap 95% CIs (error bars and ribbon)
    and a star at the knee of each k's path. main() passes winners_only=True.

    y_mode controls the vertical metric:
      - 'savings' (default): y = 100 * (old - new) / old = -dCost_pct  (higher = cheaper)
      - 'cost_remaining': y = 100 + dCost_pct = 100 - savings         (lower = cheaper)

    Writes figure_cost_benefit_points_by_theta_and_k.{png,svg}; with --dump_debug also the
    per-θ medians and CIs (debug_points_by_theta_and_k_theta_<θ>.csv and _all.csv).
    """
    _ensure_dir(out_dir)
    # Visual encoding: color by k, single point symbol
    cmap = plt.get_cmap('Set2')
    k_colors = {1: cmap(0), 2: cmap(1), 3: cmap(2)}
    point_marker = 'o'
    plt.figure(figsize=(8.8, 6.6))
    ax = plt.gca()

    xs = []
    ys = []
    points_by_k: dict[int, list[tuple[float, float, float, float, float]]] = {k: [] for k in ks}
    # Row counts per k for the legend
    counts_by_k: dict[int, int] = {k: 0 for k in ks}
    debug_rows_total = []  # accumulate across all thetas (for one consolidated CSV)
    for th, d in sorted(theta_to_dirs.items()):
        try:
            df_rows = _load_row_level_df(Path(d)).replace([np.inf, -np.inf], np.nan)
        except Exception:
            continue
        agg = _aggregate_by_k(df_rows, winners_only=winners_only, ks=ks, bootstraps=bootstraps, rng_seed=24680 + int(th * 100))
        # Count rows per k (the rows winners_only selects)
        try:
            H_all = _get_numeric_series(df_rows, ['dRDI_pct', 'delta_rdi_pct'], scale_100_if_fraction=True)
            C_all = _get_numeric_series(df_rows, ['dCost_pct', 'delta_cost_pct'], scale_100_if_fraction=True)
            if winners_only and 'selected' in df_rows.columns:
                df_eff = df_rows[df_rows['selected'] == True].copy()
                H_all = _get_numeric_series(df_eff, ['dRDI_pct', 'delta_rdi_pct'], scale_100_if_fraction=True)
                C_all = _get_numeric_series(df_eff, ['dCost_pct', 'delta_cost_pct'], scale_100_if_fraction=True)
            else:
                df_eff = df_rows
            if 'k_substitution' not in df_eff.columns:
                raise ValueError("Expected 'k_substitution' column.")
            kvals_eff = pd.to_numeric(df_eff['k_substitution'], errors='coerce').to_numpy()
            mfin = np.isfinite(H_all) & np.isfinite(C_all) & np.isfinite(kvals_eff)
            for k in ks:
                mk = mfin & (kvals_eff == k)
                if np.any(mk):
                    counts_by_k[k] += int(np.sum(mk))
        except Exception:
            pass
        for k in ks:
            if k not in agg:
                continue
            h_med = agg[k]['h_med']
            c_med = agg[k]['c_med']
            h_lo, h_hi = agg[k]['h_lo'], agg[k]['h_hi']
            c_lo, c_hi = agg[k]['c_lo'], agg[k]['c_hi']
            # y metric depending on mode
            y_metric = -c_med if y_mode == 'savings' else 100.0 + c_med
            xs.append(h_med); ys.append(y_metric)
            # Plot point: uniform styling per k
            ax.scatter([h_med], [y_metric], s=58,
                       color=k_colors.get(k, '#666'), marker=point_marker,
                       alpha=0.8, edgecolors='white', linewidths=0.7, zorder=3)
            # Error bars: transform cost CI to chosen metric
            y_lo_val = (-c_hi) if y_mode == 'savings' else (100.0 + c_lo)
            y_hi_val = (-c_lo) if y_mode == 'savings' else (100.0 + c_hi)
            ax.errorbar([h_med], [y_metric],
                        xerr=[[h_med - h_lo],[h_hi - h_med]],
                        yerr=[[y_metric - y_lo_val],[y_hi_val - y_metric]],
                        fmt='none', ecolor=k_colors.get(k, '#666'), elinewidth=1.0, alpha=0.6, capsize=2, zorder=5)
            # Store: (theta, x, y, y_lo, y_hi)
            y_val = y_metric
            y_lo = y_lo_val
            y_hi = y_hi_val
            points_by_k.setdefault(k, []).append((th, h_med, y_val, y_lo, y_hi))

        # Optional debug dump per-theta per-k medians/CI
        if _DUMP_DEBUG:
            try:
                rows = []
                for k in ks:
                    if k in agg:
                        rows.append({
                            'theta': th,
                            'k': k,
                            'H_med': agg[k]['h_med'],
                            'H_lo': agg[k]['h_lo'],
                            'H_hi': agg[k]['h_hi'],
                            'C_med': agg[k]['c_med'],
                            'C_lo': agg[k]['c_lo'],
                            'C_hi': agg[k]['c_hi']
                        })
                if rows:
                    _dump_df(Path(out_dir) / f"debug_points_by_theta_and_k_theta_{str(th).replace('.', 'p')}.csv", pd.DataFrame(rows))
                    debug_rows_total.extend(rows)
            except Exception:
                pass

    if xs and ys:
        X = np.asarray(xs); Y = np.asarray(ys)
        xlo, xhi = np.nanpercentile(X, [1, 99])
        ylo, yhi = np.nanpercentile(Y, [1, 99])
        pad_x = 0.08 * (xhi - xlo) if np.isfinite(xhi - xlo) else 5
        pad_y = 0.08 * (yhi - ylo) if np.isfinite(yhi - ylo) else 5
        # Set x-range starting at 0 and expand to cover points with padding
        ax.set_xlim(0.0, max(12.0, xhi + pad_x))
        ax.set_ylim(max(0.0, ylo - pad_y), yhi + pad_y)
    # Connect θ within each k; add CI ribbon per k
    y_all_los = []
    y_all_his = []
    for k, pts in points_by_k.items():
        if not pts:
            continue
        pts_theta_sorted = sorted(pts, key=lambda t: t[0])
        x_line = [p[1] for p in pts_theta_sorted]
        y_line = [p[2] for p in pts_theta_sorted]
        ax.plot(x_line, y_line, color=k_colors.get(k, '#666'), lw=1.8 if k==3 else 1.4, alpha=0.9 if k==3 else 0.75, zorder=3, label=f"k={k} path")
        # CI ribbon: sort by x for monotonic fill
        pts_x_sorted = sorted(pts, key=lambda t: t[1])
        x_fill = np.array([p[1] for p in pts_x_sorted])
        y_lo = np.array([p[3] for p in pts_x_sorted])
        y_hi = np.array([p[4] for p in pts_x_sorted])
        ax.fill_between(x_fill, y_lo, y_hi, color=k_colors.get(k, '#666'), alpha=(0.18 if k==3 else 0.10), zorder=2)
        y_all_los.extend(y_lo.tolist()); y_all_his.extend(y_hi.tolist())
        # Elbow (knee) detection on the θ path: max distance from line through endpoints
        try:
            import numpy as _np
            X = _np.asarray(x_line, dtype=float)
            Y = _np.asarray(y_line, dtype=float)
            if X.size >= 2:
                if X.size >= 3:
                    x0, y0 = X[0], Y[0]; x1, y1 = X[-1], Y[-1]
                    dx, dy = x1 - x0, y1 - y0
                    denom = float(_np.hypot(dx, dy)) + 1e-9
                    d = _np.abs(dy*X - dx*Y + x1*y0 - y1*x0) / denom
                    knee_idx = int(_np.nanargmax(d))
                else:
                    knee_idx = int(X.size // 2)
                kx, ky = float(X[knee_idx]), float(Y[knee_idx])
                # Draw using screen offsets so visibility is consistent
                dx, dy = 18, 18  # points (slightly longer arrow offset)
                # Arrow from offset label position to the point
                ax.annotate('', xy=(kx, ky), xytext=(dx, dy), textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', lw=1.6, color='black', shrinkA=8),
                            zorder=1000, clip_on=False)
                # Star marker at the label end (offset position) using transform offset
                import matplotlib.transforms as mtransforms
                trans = mtransforms.offset_copy(ax.transData, fig=plt.gcf(), x=dx, y=dy, units='points')
                ax.scatter([kx], [ky], marker='*', s=190, color=k_colors.get(k, '#666'),
                           edgecolors='black', linewidths=0.8, zorder=1001, clip_on=False, transform=trans)
                # Numeric label slightly to the right of the star
                trans_text = mtransforms.offset_copy(ax.transData, fig=plt.gcf(), x=dx+12, y=dy, units='points')
                ax.text(kx, ky, f"({kx:.1f}%, {ky:.1f}%)", fontsize=9, color='black',
                        zorder=1001, clip_on=False, transform=trans_text)
        except Exception:
            pass
    # Only vertical reference line at ΔRDI=0; y is Savings
    ax.axvline(0.0, ls='--', c='#999999', lw=1.0, zorder=1)
    ax.set_xlabel('Nutrition gain vs daily targets — (higher = healthier)')
    ax.set_ylabel('Cost savings (%) — (higher = cheaper)')
    # Title (without stats subtitle)
    ax.set_title('Best-Value Meal Swaps: Healthier Meals for Less (by k-hop food substitutions)')
    # Build custom legends: (1) k-hop substitutions; (2) point marker
    import matplotlib.lines as mlines
    k_handles = []
    for k in ks:
        lab_n = counts_by_k.get(k, 0)
        k_handles.append(mlines.Line2D([], [], color=k_colors.get(k, '#666'), marker=point_marker, linestyle='-', label=f"{k}-hop substitutions (n={lab_n:,})"))
    leg1 = ax.legend(handles=k_handles, title='k-hop Substitutions', frameon=True, loc='lower right', fancybox=True)
    try:
        fr = leg1.get_frame(); fr.set_facecolor('white'); fr.set_alpha(0.9); fr.set_edgecolor('#cccccc')
    except Exception:
        pass
    ax.add_artist(leg1)
    # Adjust y-limits to include full CI with padding, while clamping lower bound at 0
    if y_all_los and y_all_his:
        ylo_ci = float(np.nanmin(np.array(y_all_los)))
        yhi_ci = float(np.nanmax(np.array(y_all_his)))
        ymin, ymax = ax.get_ylim()
        pad = 0.06 * (yhi_ci - ylo_ci if np.isfinite(yhi_ci - ylo_ci) and (yhi_ci - ylo_ci) > 0 else (ymax - ymin if (ymax - ymin) > 0 else 1.0))
        new_lo = max(0.0, min(ymin, ylo_ci - pad))
        new_hi = max(ymax, yhi_ci + pad)
        ax.set_ylim(new_lo, new_hi)

    # Light dotted grid drawn below the data
    try:
        ax.grid(True, ls=':', lw=0.6, alpha=0.25)
        ax.set_axisbelow(True)
    except Exception:
        pass

    # Final clamp at zero for both axes (after all adjustments)
    try:
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
    except Exception:
        pass
    # Write consolidated debug file across all thetas (one row per (theta,k))
    if _DUMP_DEBUG and debug_rows_total:
        try:
            _dump_df(Path(out_dir) / "debug_points_by_theta_and_k_all.csv", pd.DataFrame(debug_rows_total))
        except Exception:
            pass
    plt.tight_layout()
    plt.savefig(out_dir / 'figure_cost_benefit_points_by_theta_and_k.png', dpi=200)
    plt.savefig(out_dir / 'figure_cost_benefit_points_by_theta_and_k.svg')
    plt.close()


def _compute_knee_optimal_theta_per_k(theta_to_dirs: dict[float, Path], ks: tuple[int, ...] = (1,2,3), winners_only: bool = True, bootstraps: int = 300) -> dict[int, float]:
    """Compute an "optimal" theta per k using knee detection on the path of operating points.

    We aggregate per-theta winners to get the median operating point per k, then
    choose the point that maximizes distance from the line through the path endpoints.
    Returns {k: theta_opt} for the ks that have data.
    """
    points_by_k: dict[int, list[tuple[float, float, float]]] = {}
    for th, d in sorted(theta_to_dirs.items()):
        try:
            df_rows = _load_row_level_df(Path(d)).replace([np.inf, -np.inf], np.nan)
        except Exception:
            continue
        agg = _aggregate_by_k(df_rows, winners_only=winners_only, ks=ks, bootstraps=bootstraps, rng_seed=1357 + int(th*100))
        for k in ks:
            if k not in agg:
                continue
            h_med = agg[k]['h_med']
            c_med = agg[k]['c_med']
            y_val = -c_med  # savings
            points_by_k.setdefault(k, []).append((float(th), float(h_med), float(y_val)))

    theta_opt_by_k: dict[int, float] = {}
    for k, pts in points_by_k.items():
        if len(pts) < 3:
            # fallback: choose theta with max y (savings) if insufficient points
            if pts:
                theta_opt_by_k[k] = sorted(pts, key=lambda t: t[2], reverse=True)[0][0]
            continue
        pts_sorted = sorted(pts, key=lambda t: t[0])
        X = np.asarray([p[1] for p in pts_sorted], dtype=float)
        Y = np.asarray([p[2] for p in pts_sorted], dtype=float)
        x0, y0 = X[0], Y[0]; x1, y1 = X[-1], Y[-1]
        dx, dy = x1 - x0, y1 - y0
        denom = float(np.hypot(dx, dy)) + 1e-9
        dists = np.abs(dy*X - dx*Y + x1*y0 - y1*x0) / denom
        knee_idx = int(np.nanargmax(dists))
        theta_opt_by_k[k] = float(pts_sorted[knee_idx][0])
    return theta_opt_by_k


from matplotlib.colors import LinearSegmentedColormap, to_rgb
def _single_hue_cmap(name: str, base_hex: str) -> LinearSegmentedColormap:
    rgb = to_rgb(base_hex)
    def _mix(c, w=(1.0,1.0,1.0), a=0.5):
        return tuple((1.0 - a)*w[i] + a*c[i] for i in range(3))
    deeper = tuple(max(0.0, v*0.85) for v in rgb)
    stops = [
        (0.0, (0.96, 0.96, 0.96)),
        (0.2, _mix(rgb, a=0.30)),
        (0.5, _mix(rgb, a=0.55)),
        (0.8, rgb),
        (1.0, deeper),
    ]
    return LinearSegmentedColormap.from_list(name, stops, N=256)


def figure_ternary_hcb_from_k_dirs(
    k_to_dir: dict[int, Path],
    out_dir: Path,
    ternary_style: str = 'objective',
    raw_use_winners: bool = False,
    raw_scale: str = 'per_k_p95',
    per_k_normalize: bool = False,
    per_panel_colorscale: bool = True,
):
    """Ternary panels (nutrient gain, cost savings, swap simplicity) for each k, each k read
    from its own θ run folder; writes figure_ternary_hcb_by_k_optimal_theta.{png,svg} and
    returns the figure.

    ternary_style: 'objective' (value-weighted KDE), 'density' (KDE density) or 'raw'
    (shares of unscaled ΔRDI%, savings% and effort; effort recomputed from the meal vectors)
    """
    try:
        import ternary  # type: ignore
    except Exception:
        print("Missing dependency 'python-ternary'; aborting.")
        return

    # Load per-k dataframes, keep only rows for that k
    dfs: list[pd.DataFrame] = []
    ks_order = []
    for k in sorted(k_to_dir.keys()):
        d = Path(k_to_dir[k])
        try:
            dfk = _read_matches(d).replace([np.inf, -np.inf], np.nan)
        except Exception:
            continue
        if 'k_substitution' not in dfk.columns:
            raise ValueError("Expected 'k_substitution' in per-k dataframe.")
        dfk = dfk[pd.to_numeric(dfk['k_substitution'], errors='coerce') == k].copy()
        dfk['k'] = k
        dfs.append(dfk)
        ks_order.append(k)
    if not dfs:
        print("No per-k data available for ternary.")
        return
    df = pd.concat(dfs, ignore_index=True)
    # Debug: overall sample counts per k
    try:
        vc = pd.to_numeric(df['k'], errors='coerce').value_counts().sort_index()
        print("[debug optimalθ] samples by k:", ", ".join([f"k={int(k)}: {int(n)}" for k, n in vc.items()]))
    except Exception:
        pass

    # Build encodings
    # Ensure ΔRDI is in percent (not pp) and prefer explicit dRDI_pct if present
    df['dRDI_pct'] = pd.to_numeric(df.get('dRDI_pct', 100.0 * df.get('dRDI', df.get('improvement', np.nan))), errors='coerce')
    if {'real_cost','dCost'}.issubset(df.columns):
        with np.errstate(divide='ignore', invalid='ignore'):
            df['dCost_pct'] = 100.0 * (pd.to_numeric(df['dCost'], errors='coerce') / pd.to_numeric(df['real_cost'], errors='coerce'))
    # Portion/composition if missing → derive from meal vectors
    portion_list, comp_list = [], []
    effort_kstable_list, effort_portion_list, effort_comp_list = [], [], []

    # For raw style, always recompute k-aware effort when meal vectors are available;
    # otherwise compute only if columns are missing.
    if (ternary_style == 'raw') or (not {'portion_shift_pct','composition_change_pct'}.issubset(df.columns)):
        try:
            data = _load_meals_and_features(_find_data_exports(Path(next(iter(k_to_dir.values())))))
        except Exception:
            data = None

        portion_list, comp_list, effort_kstable_list, effort_portion_list, effort_comp_list = [], [], [], [], []
        for _, r in df.iterrows():
            mt = r.get('Meal Type'); gi = int(r.get('gen_idx', -1)); ri = int(r.get('real_idx', -1))
            if (data is not None and mt in data.get('gen_by_type', {}) and mt in data.get('real_by_type', {})
                and 0 <= gi < len(data['gen_by_type'][mt]) and 0 <= ri < len(data['real_by_type'][mt])):
                gm = np.asarray(data['gen_by_type'][mt][gi], float)
                rm = np.asarray(data['real_by_type'][mt][ri], float)
                k_here = int(pd.to_numeric(r.get('k', r.get('k_substitution', np.nan)), errors='coerce')) if \
                        pd.notna(r.get('k', r.get('k_substitution', np.nan))) else 1
                E, E_port, E_comp, ps_pct, comp_pct, *_ = _effort_kaware(gm, rm, k_here, alpha=0.5)
                portion_list.append(float(np.clip(ps_pct, 0.0, 100.0)))
                comp_list.append(float(np.clip(comp_pct, 0.0, 100.0)))
                effort_kstable_list.append(E); effort_portion_list.append(E_port); effort_comp_list.append(E_comp)
            else:
                portion_list.append(np.nan); comp_list.append(np.nan)
                effort_kstable_list.append(np.nan); effort_portion_list.append(np.nan); effort_comp_list.append(np.nan)
        df['portion_shift_pct'] = portion_list
        df['composition_change_pct'] = comp_list
        df['effort_kstable'] = effort_kstable_list
        df['effort_kstable_pct'] = 100.0 * df['effort_kstable']
        df['effort_kstable_portion'] = effort_portion_list   # optional diagnostics
        df['effort_kstable_comp'] = effort_comp_list         # optional diagnostics

    # Use ΔRDI (%) not pp for Health
    H = np.maximum(0.0, pd.to_numeric(df.get('dRDI_pct', 100.0 * df.get('dRDI', df.get('improvement', np.nan))), errors='coerce').to_numpy(float))
    S = np.maximum(0.0, -pd.to_numeric(df['dCost_pct'], errors='coerce').to_numpy(float))
    P = pd.to_numeric(df['portion_shift_pct'], errors='coerce').to_numpy(float)
    C = pd.to_numeric(df['composition_change_pct'], errors='coerce').to_numpy(float)
    # Debug: raw feature ranges
    def _rng(x):
        x = np.asarray(x, float)
        return (float(np.nanmin(x)) if np.isfinite(np.nanmin(x)) else np.nan,
                float(np.nanmax(x)) if np.isfinite(np.nanmax(x)) else np.nan)
    print(f"[debug optimalθ] raw dRDI_pct range: {_rng(df.get('dRDI_pct'))}")
    print(f"[debug optimalθ] raw dCost_pct range: {_rng(df.get('dCost_pct'))}")
    print(f"[debug optimalθ] raw portion_shift_pct range: {_rng(P)}")
    print(f"[debug optimalθ] raw composition_change_pct range: {_rng(C)}")

    # Similarity (stability/ease) is simply 1 − Effort
    # Without an effort value, use the portion/composition similarity instead
    S_port = 1.0 - np.clip(pd.to_numeric(df['portion_shift_pct'], errors='coerce').to_numpy(float) /
                        max(float(np.nanpercentile(df['portion_shift_pct'], 99)), 1e-9), 0.0, 1.0)
    S_comp = 1.0 - np.clip(pd.to_numeric(df['composition_change_pct'], errors='coerce').to_numpy(float) /
                        max(float(np.nanpercentile(df['composition_change_pct'], 99)), 1e-9), 0.0, 1.0)
    valid = (~np.isnan(S_port)).astype(float) + (~np.isnan(S_comp)).astype(float)
    B_fallback = (np.nan_to_num(S_port) + np.nan_to_num(S_comp)) / np.maximum(valid, 1.0)
    B_fallback = np.clip(B_fallback, 0.0, 1.0)

    E = pd.to_numeric(df['effort_kstable'], errors='coerce').to_numpy(float)
    B = np.where(np.isfinite(E), 1.0 - E, B_fallback)

    # Normalization for objective/density
    if ternary_style == 'objective' and per_k_normalize:
        kvals = pd.to_numeric(df.get('k', pd.Series(dtype=float)), errors='coerce').to_numpy()
        Hn_all = np.zeros_like(H, dtype=float)
        Sn_all = np.zeros_like(S, dtype=float)
        Bn_all = np.clip(B, 0.0, 1.0)
        for k in np.unique(kvals[np.isfinite(kvals)]):
            mk = (kvals == k)
            if not np.any(mk):
                continue
            try:
                H_REF_k = float(np.nanpercentile(H[mk], 90)) or 1.0
            except Exception:
                H_REF_k = 1.0
            try:
                S_REF_k = float(np.nanpercentile(S[mk], 90)) or 1.0
            except Exception:
                S_REF_k = 1.0
            Hn_all[mk] = np.clip(H[mk] / max(H_REF_k, 1e-9), 0.0, 1.5)
            Sn_all[mk] = np.clip(S[mk] / max(S_REF_k, 1e-9), 0.0, 1.5)
    else:
        # Global anchors or density/other styles
        H_REF = 10.0   # meaningful ΔRDI (%)
        S_REF = 20.0   # material savings (%)
        STAB_REF = 100.0  # stability (%)
        Hn_all = np.clip(H / max(H_REF, 1e-9), 0.0, 1.0)
        Sn_all = np.clip(S / max(S_REF, 1e-9), 0.0, 1.0)
        Bn_all = np.clip((100.0 * B) / max(STAB_REF, 1e-9), 0.0, 1.0)
    if ternary_style != 'raw':
        print(f"[debug optimalθ] normalized Hn range: {_rng(Hn_all)}  Sn range: {_rng(Sn_all)}  Bn range: {_rng(Bn_all)}")
    kvals = pd.to_numeric(df.get('k', pd.Series(dtype=float)), errors='coerce').to_numpy()
    # Debug: counts per k in kvals
    try:
        from collections import Counter
        cnt = Counter(k for k in kvals if np.isfinite(k))
        print("[debug optimalθ] kvals counts:", ", ".join([f"k={int(k)}: {int(v)}" for k, v in sorted(cnt.items())]))
    except Exception:
        pass

    # Prepare panels
    scale = 100
    panels = [(f"{int(k)}-hop substitutions", (kvals==k)) for k in ks_order]
    heat_funcs = []
    # For raw style: prepare pooled KDE and global vmax for shared normalization
    pooled_XY_raw = None
    pooled_kde = None
    global_vmax98 = None
    bandwidth = 0.12
    vmax_seen = 0.0
    from sklearn.neighbors import KernelDensity
    for _, msk in panels:
        m = msk & np.isfinite(Hn_all) & np.isfinite(Sn_all) & np.isfinite(Bn_all)
        print(f"[optimalθ] k panel points: {int(np.sum(msk))} raw, {int(np.sum(m))} valid")

        if not np.any(m):
            heat_funcs.append(None)
            continue
        Hn, Cn, Bn = Hn_all[m], Sn_all[m], Bn_all[m]
        Ssum = Hn + Cn + Bn + 1e-9
        Hb, Cb, Bb = Hn/Ssum, Cn/Ssum, Bn/Ssum
        XY = _bary_to_xy(Hb, Cb, Bb)
        if XY.shape[0] == 0:
            heat_funcs.append(None)
            continue
        kde = KernelDensity(kernel='gaussian', bandwidth=float(bandwidth)).fit(XY)
        def f_factory(kde):
            def f(point: tuple[int,int,int]) -> float:
                c, b, h = point
                s = float(c + b + h) or 1.0
                h, c, b = h/s, c/s, b/s
                xy = _bary_to_xy(np.array([h]), np.array([c]), np.array([b]))
                return float(np.exp(kde.score_samples(xy.reshape(1, -2))))
            return f
        heat_funcs.append(f_factory(kde))
        probe = np.exp(kde.score_samples(XY))
        if probe.size:
            vmax_seen = max(vmax_seen, float(np.percentile(probe, 98)))

    # Note: pooled KDE is constructed below after df_eff_pool is defined

    # Pooled anchors for raw scaling, built once (respecting raw_use_winners)
    df_eff_pool = df.copy()
    if raw_use_winners and 'selected' in df_eff_pool.columns:
        df_eff_pool = df_eff_pool[df_eff_pool['selected'] == True]

    H_pool = np.maximum(0.0, pd.to_numeric(
        df_eff_pool.get('dRDI_pct', 100.0 * df_eff_pool.get('dRDI', df_eff_pool.get('improvement', np.nan))),
        errors='coerce').to_numpy(float))
    S_pool = np.maximum(0.0, -pd.to_numeric(
        df_eff_pool.get('dCost_pct', pd.Series(dtype=float)),
        errors='coerce').to_numpy(float))

    if 'effort_kstable' in df_eff_pool.columns:
        E_pool_pct = 100.0 * pd.to_numeric(df_eff_pool['effort_kstable'], errors='coerce').to_numpy(float)
    elif 'effort_bc_pct' in df_eff_pool.columns:
        E_pool_pct = pd.to_numeric(df_eff_pool['effort_bc_pct'], errors='coerce').to_numpy(float)
    elif 'effort_bc' in df_eff_pool.columns:
        E_pool_pct = 100.0 * pd.to_numeric(df_eff_pool['effort_bc'], errors='coerce').to_numpy(float)
    else:
        E_pool_pct = np.full(len(df_eff_pool), np.nan, dtype=float)

    m_pool = np.isfinite(H_pool) & np.isfinite(S_pool) & np.isfinite(E_pool_pct)
    H_pool, S_pool, E_pool_pct = H_pool[m_pool], S_pool[m_pool], E_pool_pct[m_pool]

    H_REF_GLOBAL = float(np.nanpercentile(H_pool, 95)) if H_pool.size else 1.0
    S_REF_GLOBAL = float(np.nanpercentile(S_pool, 95)) if S_pool.size else 1.0
    E_REF_GLOBAL = float(np.nanpercentile(E_pool_pct, 95)) if E_pool_pct.size else 1.0

    # Build pooled KDE/global vmax for raw style using the filtered pool
    if ternary_style == 'raw' and not per_panel_colorscale:
        try:
            H_raw_pool = np.maximum(0.0, pd.to_numeric(
                df_eff_pool.get('dRDI_pct', 100.0 * df_eff_pool.get('dRDI', df_eff_pool.get('improvement', np.nan))),
                errors='coerce').to_numpy(float))
            S_raw_pool = np.maximum(0.0, -pd.to_numeric(
                df_eff_pool.get('dCost_pct', pd.Series(dtype=float)),
                errors='coerce').to_numpy(float))
            if 'effort_kstable' in df_eff_pool.columns:
                E_frac_pool = pd.to_numeric(df_eff_pool['effort_kstable'], errors='coerce').to_numpy(float)
            elif 'effort_kstable_pct' in df_eff_pool.columns:
                E_frac_pool = pd.to_numeric(df_eff_pool['effort_kstable_pct'], errors='coerce').to_numpy(float) / 100.0
            elif 'effort_bc_pct' in df_eff_pool.columns:
                E_frac_pool = pd.to_numeric(df_eff_pool['effort_bc_pct'], errors='coerce').to_numpy(float) / 100.0
            elif 'effort_bc' in df_eff_pool.columns:
                E_frac_pool = pd.to_numeric(df_eff_pool['effort_bc'], errors='coerce').to_numpy(float)
            else:
                E_frac_pool = np.full(len(df_eff_pool), np.nan, dtype=float)
            m_all = np.isfinite(H_raw_pool) & np.isfinite(S_raw_pool) & np.isfinite(E_frac_pool) & (E_frac_pool >= 0) & (E_frac_pool <= 1)
            if np.any(m_all):
                E_pct_pool = 100.0 * E_frac_pool[m_all]
                denom_pool = H_raw_pool[m_all] + S_raw_pool[m_all] + E_pct_pool + 1e-12
                Hb_all, Cb_all, Bb_all = H_raw_pool[m_all]/denom_pool, S_raw_pool[m_all]/denom_pool, E_pct_pool/denom_pool
                pooled_XY_raw = _bary_to_xy(Hb_all, Cb_all, Bb_all)
                if pooled_XY_raw.shape[0] > 0:
                    pooled_kde = KernelDensity(kernel='gaussian', bandwidth=float(bandwidth)).fit(pooled_XY_raw)
                    probe_all = np.exp(pooled_kde.score_samples(pooled_XY_raw))
                    global_vmax98 = float(np.percentile(probe_all, 98)) if probe_all.size else 1.0
                    print(f"[raw pooled] KDE built with n={pooled_XY_raw.shape[0]} points; vmax98={global_vmax98:.3g}")
        except Exception:
            pooled_XY_raw = None; pooled_kde = None; global_vmax98 = None
    ncols = min(3, len(panels)); nrows = int(np.ceil(len(panels)/ncols))
    fig = plt.figure(figsize=(9.2*ncols, 6.4*nrows))
    try:
        fig.set_constrained_layout(True)
    except Exception:
        pass

    k_palette = {1: '#1b9e77', 2: '#d95f02', 3: '#7570b3'}
    axes_created = []
    for i, ((title,_), hf) in enumerate(zip(panels, heat_funcs), start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        axes_created.append(ax)
        tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
        try:
            k_here = int(str(title).split('-')[0])
        except Exception:
            k_here = None
        k_color = k_palette.get(k_here, '#444444')
        tax.boundary(linewidth=1.8, axes_colors={'l': k_color, 'r': k_color, 'b': k_color})
        tax.gridlines(multiple=10, color="#e8e8e8", linewidth=0.7)
        # Vertex labels (no parentheses)
        tax.left_corner_label("Cost savings", fontsize=14, offset=0.12)
        tax.right_corner_label("Swap simplicity", fontsize=14, offset=0.14)
        tax.top_corner_label("Nutrient adequacy gain", fontsize=14, offset=0.16)
        # Axis labels indicate direction (mapped to edges)
        tax.bottom_axis_label('More cost savings', fontsize=12, offset=0.06)
        tax.right_axis_label('Higher swap simplicity', fontsize=12, offset=0.14)
        tax.left_axis_label('Higher nutrient adequacy (ΔRDI)', fontsize=12, offset=0.14)
        tax.ticks(axis='lbr', multiple=100, linewidth=0.0, tick_formats="%d")
        tax.clear_matplotlib_ticks()
        if ternary_style == 'density' and hf is not None:
            def hf_norm(p: tuple[int,int,int]) -> float:
                v = hf(p)
                if not np.isfinite(v) or vmax_seen <= 0:
                    return 0.0
                return float(_norm01(v, float(vmax_seen), gamma=0.7))
            try:
                panel_cmap = _single_hue_cmap(f'kmap_opt_{k_here}', k_color)
            except Exception:
                panel_cmap = plt.cm.Greys
            _render_tri_heatmap_as_image(ax, hf_norm, panel_cmap, scale)
            # Single explicit colorbar for this panel (we control title and ticks)
            import matplotlib as _mpl
            sm_panel = _mpl.cm.ScalarMappable(cmap=panel_cmap); sm_panel.set_clim(0, 1)
            cbar_panel = fig.colorbar(sm_panel, ax=ax, fraction=0.03, pad=0.02)
            cbar_panel.ax.set_title("Density", fontsize=16, pad=8)
            cbar_panel.ax.tick_params(labelsize=14)
        elif ternary_style == 'raw':
            # Raw (unanchored) style: H = max(ΔRDI%,0), S = max(savings%,0), E = effort in [0,1]
            # Position uses shares from H,S,E; color shows density only
            m = (kvals==k_here)
            if np.sum(m) > 0:
                # Raw magnitudes
                df_eff = df.copy()
                if raw_use_winners and 'selected' in df_eff.columns:
                    df_eff = df_eff[df_eff['selected'] == True]
                    m = (pd.to_numeric(df_eff.get('k', pd.Series(dtype=float)), errors='coerce') == k_here)
                H_raw = np.maximum(0.0, pd.to_numeric(df_eff.get('dRDI_pct', 100.0 * df_eff.get('dRDI', df_eff.get('improvement', np.nan))), errors='coerce').to_numpy(float)[m])
                S_raw = np.maximum(0.0, -pd.to_numeric(df_eff.get('dCost_pct', pd.Series(dtype=float)), errors='coerce').to_numpy(float)[m])
                # Effort: prefer fraction if present, else percent/100
                if 'effort_kstable' in df_eff.columns:
                    E = pd.to_numeric(df_eff['effort_kstable'], errors='coerce').to_numpy(float)[m]  # fraction [0,1]
                else:
                    E = np.full(np.sum(m), np.nan, dtype=float)
                # Drop invalid rows
                valid = np.isfinite(H_raw) & np.isfinite(S_raw) & np.isfinite(E) & (E >= 0) & (E <= 1)
                H_raw, S_raw, E = H_raw[valid], S_raw[valid], E[valid]
                if H_raw.size > 0:
                    # Convert Effort to percent
                    E_pct = 100.0 * E
                    # Per-k scaling of raw magnitudes (based on winners if requested)
                    mode = str(raw_scale or 'per_k_p95')

                    # helpers for per-k refs
                    def ref_per_k(vals: np.ndarray, how: str) -> float:
                        if vals.size == 0:
                            return 1.0
                        if how == 'max':
                            r = float(np.nanmax(vals))
                        else:  # 'p95'
                            r = float(np.nanpercentile(vals, 95))
                        return r if r > 1e-9 else 1.0

                    if mode == 'global_p95':
                        H_ref = H_REF_GLOBAL
                        S_ref = S_REF_GLOBAL
                        E_ref = E_REF_GLOBAL
                    elif mode == 'none':
                        # treat inputs as absolute percentages
                        H_ref = 100.0
                        S_ref = 100.0
                        E_ref = 100.0
                    elif mode == 'per_k_max':
                        H_ref = ref_per_k(H_raw, 'max')
                        S_ref = ref_per_k(S_raw, 'max')
                        E_ref = ref_per_k(E_pct,  'max')
                    else:  # default: per_k_p95
                        H_ref = ref_per_k(H_raw, 'p95')
                        S_ref = ref_per_k(S_raw, 'p95')
                        E_ref = ref_per_k(E_pct,  'p95')
                    print(f"[raw scale] refs H={H_ref:.3g} S={S_ref:.3g} E={E_ref:.3g}  mode={mode} winners={raw_use_winners}")
                    Hs = H_raw / H_ref
                    Ss = S_raw / S_ref
                    Es = E_pct / E_ref
                    denom = Hs + Ss + Es + 1e-12
                    Hb, Cb, Bb = Hs/denom, Ss/denom, Es/denom
                    # Debug dump of actual raw values and shares
                    if _DUMP_DEBUG:
                        try:
                            out = pd.DataFrame({
                                'k': int(k_here) if k_here is not None else np.nan,
                                'H_raw_pct': H_raw,
                                'S_raw_pct': S_raw,
                                'E_effort_pct': E_pct,
                                'E_effort_frac': E,
                                'H_ref': H_ref, 'S_ref': S_ref, 'E_ref': E_ref,
                                'H_scaled': Hs, 'S_scaled': Ss, 'E_scaled': Es,
                                'Hb': Hb, 'Cb': Cb, 'Bb': Bb
                            })
                            _dump_df(Path(out_dir) / f"debug_ternary_raw_k{k_here}.csv", out)
                        except Exception:
                            pass
                    # Map to XY for KDE density
                    XY = _bary_to_xy(Hb, Cb, Bb)
                    try:
                        
                        if per_panel_colorscale or pooled_kde is None:
                            kde_use = KernelDensity(kernel='gaussian', bandwidth=float(bandwidth)).fit(XY)
                            probe = np.exp(kde_use.score_samples(XY))
                            vmax98 = float(np.percentile(probe, 98)) if probe.size else 1.0
                        else:
                            kde_use = pooled_kde
                            vmax98 = global_vmax98 if isinstance(global_vmax98, float) and global_vmax98 > 0 else 1.0

                        def hf_raw_norm(p: tuple[int,int,int]) -> float:
                            c, b, h = p
                            s = float(c + b + h) or 1.0
                            h, c, b = h/s, c/s, b/s
                            xy = _bary_to_xy(np.array([h]), np.array([c]), np.array([b]))
                            if kde_use is None:
                                return 0.0
                            v = float(np.exp(kde_use.score_samples(xy.reshape(1, -2))))
                            return float(_norm01(v, float(vmax98), gamma=0.7))
                        panel_cmap = _single_hue_cmap(f'kmap_raw_{k_here}', k_color)
                        _render_tri_heatmap_as_image(ax, hf_raw_norm, panel_cmap, scale)
                        import matplotlib as _mpl
                        sm_panel = _mpl.cm.ScalarMappable(cmap=panel_cmap); sm_panel.set_clim(0, 1)
                        label = "Density (per-panel)" if (per_panel_colorscale or pooled_kde is None) else "Density (pooled)"
                        cbar_panel = fig.colorbar(sm_panel, ax=ax, fraction=0.03, pad=0.02)
                        cbar_panel.ax.set_title(label, fontsize=16, pad=8)
                        cbar_panel.ax.tick_params(labelsize=14)
                    except Exception:
                        pass
                    # Overlay median shares and IQR whiskers
                    try:
                        Hb_med = float(np.nanmedian(Hb)); Cb_med = float(np.nanmedian(Cb)); Bb_med = float(np.nanmedian(Bb))
                        Hb_q = np.nanpercentile(Hb, [25, 75]); Cb_q = np.nanpercentile(Cb, [25, 75]); Bb_q = np.nanpercentile(Bb, [25, 75])
                        # Convert shares to ternary XY for plotting
                        med_xy = _bary_to_xy(np.array([Hb_med]), np.array([Cb_med]), np.array([Bb_med]))[0]
                        ax.plot([med_xy[0]], [med_xy[1]], marker='o', color='k', ms=9, mec='white', mew=0.9, zorder=6)
                        # Whiskers: from median toward each vertex along share axes
                        def share_to_xy(h, c, b):
                            return _bary_to_xy(np.array([h]), np.array([c]), np.array([b]))[0]
                        # Health whisker
                        xy_h25 = share_to_xy(Hb_q[0], Cb_med, Bb_med); xy_h75 = share_to_xy(Hb_q[1], Cb_med, Bb_med)
                        ax.plot([xy_h25[0], xy_h75[0]], [xy_h25[1], xy_h75[1]], color='#000000', lw=1.2, zorder=6)
                        # Savings whisker
                        xy_c25 = share_to_xy(Hb_med, Cb_q[0], Bb_med); xy_c75 = share_to_xy(Hb_med, Cb_q[1], Bb_med)
                        ax.plot([xy_c25[0], xy_c75[0]], [xy_c25[1], xy_c75[1]], color='#000000', lw=1.2, zorder=6)
                        # Effort whisker
                        xy_b25 = share_to_xy(Hb_med, Cb_med, Bb_q[0]); xy_b75 = share_to_xy(Hb_med, Cb_med, Bb_q[1])
                        ax.plot([xy_b25[0], xy_b75[0]], [xy_b25[1], xy_b75[1]], color='#000000', lw=1.2, zorder=6)
                    except Exception:
                        pass
        elif ternary_style != 'density':
            # Objective style with adjustable alpha (weight on Health)
            m = (kvals==k_here) & np.isfinite(Hn_all) & np.isfinite(Sn_all) & np.isfinite(Bn_all)
            print(f"[debug optimalθ] k={k_here} included={int(np.sum(m))} / raw={int(np.sum(kvals==k_here))}")
            if np.sum(m) > 0:
                Hn, Cn, Bn = Hn_all[m], Sn_all[m], Bn_all[m]
                denom = Hn + Cn + Bn + 1e-9
                Hb, Cb, Bb = Hn/denom, Cn/denom, Bn/denom
                # Use B (similarity) -> effort = 1 - B
                effort = 1.0 - np.clip(Bn, 0.0, 1.0)      # in [0,1]
                v_arr  = (np.clip(Hn,0.0,1.5) + np.clip(Cn,0.0,1.5)) / (1e-6 + effort)

                # KDE over barycentric XY weighted by value to produce a smooth gradient
                try:
                    XY = _bary_to_xy(Hb, Cb, Bb)
                    import numpy as _np
                    vn = _np.clip(v_arr, 0.0, 1.0)
                    wscale = 8
                    reps = _np.rint(vn * wscale).astype(int)
                    mask_nz = reps > 0
                    XYw = _np.repeat(XY[mask_nz], reps[mask_nz], axis=0) if _np.any(mask_nz) else XY[:0]
                    if XYw.shape[0] == 0 and XY.shape[0] > 0:
                        XYw = XY
                    bw = 0.12
                    kde = KernelDensity(kernel='gaussian', bandwidth=float(bw)).fit(XYw)
                    probe = _np.exp(kde.score_samples(XY)) if XY.shape[0] else _np.array([1.0])
                    vmax98 = float(_np.percentile(probe, 98)) if probe.size else 1.0
                    try:
                        from matplotlib.colors import LinearSegmentedColormap, to_rgb
                        base_rgb = to_rgb(k_color)
                        panel_cmap = LinearSegmentedColormap.from_list(
                            f'hcb_k_{k_here}' if k_here is not None else 'hcb_k_default',
                            [(0.96,0.96,0.96), (0.98*base_rgb[0],0.98*base_rgb[1],0.98*base_rgb[2]), base_rgb],
                            N=256
                        )
                    except Exception:
                        panel_cmap = plt.cm.magma
                    def _hf_val(p: tuple[int,int,int]) -> float:
                        c, b, h = p
                        s = float(c + b + h) or 1.0
                        h, c, b = h/s, c/s, b/s
                        xy = _bary_to_xy(_np.array([h]), _np.array([c]), _np.array([b]))
                        v = float(_np.exp(kde.score_samples(xy.reshape(1, -2))))
                        return float(_norm01(v, float(max(vmax98, 1e-9)), gamma=0.7))
                    _render_tri_heatmap_as_image(ax, _hf_val, panel_cmap, scale)
                    import matplotlib as _mpl
                    sm_panel = _mpl.cm.ScalarMappable(cmap=panel_cmap); sm_panel.set_clim(0.0, 1.0)
                    cbar_panel = fig.colorbar(sm_panel, ax=ax, fraction=0.03, pad=0.02)
                    cbar_panel.ax.set_title("Benefit–effort efficiency", fontsize=16, pad=8)
                    cbar_panel.ax.tick_params(labelsize=14)
                except Exception:
                    # Fallback to lattice smoothing if KDE fails
                    from collections import defaultdict
                    coords = _to_lattice(Cb, Bb, Hb, scale=scale)
                    sum_grid: dict[tuple[int,int,int], float] = defaultdict(float)
                    count_grid: dict[tuple[int,int,int], int] = defaultdict(int)
                    for (h, c, b), v in zip(coords, v_arr):
                        key = (int(h), int(c), int(b))
                        sum_grid[key] += float(np.clip(v, 0.0, 1.0))
                        count_grid[key] += 1
                    grid = {k: (sum_grid[k] / max(count_grid[k], 1)) for k in sum_grid}
                    grid_sm = _smooth_lattice_grid(grid, scale, passes=3, include_self=True)
                    import numpy as _np
                    _vals = _np.array(list(grid_sm.values()), float)
                    pos = _vals[_vals > 0]
                    vmax98 = float(_np.percentile(pos, 98)) if pos.size else 1.0
                    vmax98 = vmax98 if vmax98 > 0 else 1.0
                    grid_norm = {k: float(_norm01(v, float(vmax98), gamma=0.7)) for k, v in grid_sm.items()}
                    try:
                        from matplotlib.colors import LinearSegmentedColormap, to_rgb
                        base_rgb = to_rgb(k_color)
                        panel_cmap = LinearSegmentedColormap.from_list(
                            f'hcb_k_{k_here}' if k_here is not None else 'hcb_k_default',
                            [(0.96,0.96,0.96), (0.98*base_rgb[0],0.98*base_rgb[1],0.98*base_rgb[2]), base_rgb],
                            N=256
                        )
                    except Exception:
                        panel_cmap = plt.cm.magma
                    hm = tax.heatmap(grid_norm, scale=scale, style='triangular', cmap=panel_cmap, colorbar=False, vmin=0.0, vmax=1.0, scientific=False)
                    try:
                        hm.set_rasterized(True)
                    except Exception:
                        pass
                    import matplotlib as _mpl
                    sm_panel = _mpl.cm.ScalarMappable(cmap=panel_cmap); sm_panel.set_clim(0.0, 1.0)
                    cbar_panel = fig.colorbar(sm_panel, ax=ax, fraction=0.03, pad=0.02)
                    cbar_panel.ax.set_title("Value score\n(within-k)", fontsize=14, pad=8)
                    cbar_panel.ax.tick_params(labelsize=14)

        # Titles & numbers
        # Title: add N for raw and other styles consistently
        try:
            k_here_int = int(str(title).split('-')[0])
            N_here = int(np.sum(kvals==k_here_int))
            tax.set_title(f"{title}\nN={N_here}", pad=16, fontsize=15, color='#222222', fontweight='bold', y=1.06)
        except Exception:
            tax.set_title(title, pad=16, fontsize=15, color='#222222', fontweight='bold', y=1.06)
        tax.clear_matplotlib_ticks(); tax._redraw_labels()
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_frame_on(False)
        H_CAP, C_CAP, STAB_CAP = 100, 100, 100
        ax.text(0.48, 0.97, f"{H_CAP}%", transform=ax.transAxes, fontsize=12, ha='right', va='top')
        ax.text(0.52, 0.97, "0%", transform=ax.transAxes, fontsize=12, ha='left', va='top')
        ax.text(0.070, 0.055, "0%", transform=ax.transAxes, fontsize=12, ha='right', va='bottom')
        ax.text(0.110, 0.055, f"{C_CAP}%", transform=ax.transAxes, fontsize=12, ha='left', va='bottom')
        ax.text(0.890, 0.055, f"{STAB_CAP}%", transform=ax.transAxes, fontsize=12, ha='right', va='bottom')
        ax.text(0.930, 0.055, "0%", transform=ax.transAxes, fontsize=12, ha='left', va='bottom')

    _ensure_dir(out_dir)
    png_path = out_dir / 'figure_ternary_hcb_by_k_optimal_theta.png'
    svg_path = out_dir / 'figure_ternary_hcb_by_k_optimal_theta.svg'
    try:
        fig.canvas.draw()
    except Exception:
        pass
    print(f"[optimalθ] Saving PNG: {png_path}")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"[optimalθ] Saving SVG: {svg_path}")
    fig.savefig(svg_path, bbox_inches='tight', dpi=300)
    return fig


def generate_ternary_at_optimal_thetas(
    theta_to_dirs: dict[float, Path],
    out_dir: Path,
    ks: tuple[int, ...]=(1,2,3),
    ternary_style: str = 'objective',
    raw_use_winners: bool = False,
    raw_scale: str = 'per_k_p95',
) -> dict[int, float]:
    """Optional (--with_ternary): ternary plot with each k at its knee θ, also copied to
    figure_ternary_hcb_by_k_substitution_k3run.*. Returns {k: knee θ}."""
    theta_by_k = _compute_knee_optimal_theta_per_k(theta_to_dirs, ks=ks, winners_only=True)
    k_to_dir: dict[int, Path] = {}
    for k, th in theta_by_k.items():
        if th in theta_to_dirs:
            k_to_dir[k] = Path(theta_to_dirs[th])
        else:
            # Fall back to closest available theta
            th_closest = sorted(theta_to_dirs.keys(), key=lambda t: abs(t - th))[0]
            k_to_dir[k] = Path(theta_to_dirs[th_closest])
    print(f"[optimalθ] Rendering ternary to: {out_dir}")
    print(f"[optimalθ] ternary_style={ternary_style} raw_use_winners={raw_use_winners} raw_scale={raw_scale}")

    figure_ternary_hcb_from_k_dirs(
        k_to_dir,
        out_dir,
        ternary_style=ternary_style,
        raw_use_winners=raw_use_winners,
        raw_scale=raw_scale,
        per_panel_colorscale=True,  # colors normalized per panel
    )
    # Copy to k3run names for convenience
    png = out_dir / 'figure_ternary_hcb_by_k_optimal_theta.png'
    svg = out_dir / 'figure_ternary_hcb_by_k_optimal_theta.svg'
    try:
        import shutil
        if png.exists():
            shutil.copyfile(png, out_dir / 'figure_ternary_hcb_by_k_substitution_k3run.png')
        if svg.exists():
            shutil.copyfile(svg, out_dir / 'figure_ternary_hcb_by_k_substitution_k3run.svg')
    except Exception as _e:
        print(f"[optimalθ] Copy step failed: {_e}")
    return theta_by_k


def figure_category_triptych_from_k_dirs(
    k_to_dir: dict[int, Path],
    out_dir: Path,
    omit_cats: set[str] | None = None,
) -> None:
    """Figure 4C-D: for each k (rows), the heatmap of P(added category | removed category)
    and the scatter of added vs removed category probabilities, from the selected winners
    in k_to_dir[k]. Writes figure_category_scatter_plus_heatmap_k123.{png,svg} to out_dir;
    needs at least two k folders and food_to_group.json in a nearby data_exports/."""
    # Default omissions: hide miscellaneous and beverages from plots
    omit_cats = omit_cats or {"other", "beverages"}
    # Prepare category mapping and helpers.
    # When k_to_dir points to grouped temp dirs (e.g., figures_theta_sweep/_tmp_theta_grouped),
    # `data_exports` lives higher up at the run root (outputs/runs/<RUN_NAME>/data_exports).
    candidate_dirs: list[Path] = []
    for p in k_to_dir.values():
        if not p:
            continue
        p = Path(p)
        for anc in [p, p.parent, p.parent.parent, p.parent.parent.parent, p.parent.parent.parent.parent]:
            candidate_dirs.append(anc / 'data_exports')
    candidate_dirs.extend([
        Path(out_dir) / 'data_exports',
        Path(out_dir).parent / 'data_exports',
        Path(out_dir).parent.parent / 'data_exports',
        Path.cwd() / 'data_exports',
    ])
    # De-duplicate while preserving order
    seen = set()
    deduped: list[Path] = []
    for c in candidate_dirs:
        k = str(c.resolve()) if c.exists() else str(c)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(c)

    cat_map = {}
    for cand in deduped:
        try:
            cat_map = _read_food_category_map(cand)
            if cat_map:
                break
        except Exception:
            continue
    if not cat_map:
        raise RuntimeError("Missing WWEIA mapping: food_to_group.json not found or empty. Cannot build category heatmap without WWEIA Main Categories.")

    def cat_of(code: str) -> str:
        _s = str(code)
        try:
            if '.' in _s:
                _s = _s.split('.')[0]
            _s = ''.join(ch for ch in _s if ch.isdigit())
        except Exception:
            pass
        c = _s.zfill(8)
        cat = cat_map.get(c)
        if cat:
            return str(cat)
        return "other"
    # Build per-k scatter dataframes using pairwise-marginal counting (to match heatmaps)
    def _scatter_df_for_dir(run_dir: Path) -> pd.DataFrame:
        dfk = _matches_df(run_dir, winners_only=True)
        rows_h = []
        for _, rr in dfk.iterrows():
            rem = str(rr.get('removed_foods', '') or '')
            add = str(rr.get('added_foods', '') or '')
            if not rem and not add:
                continue
            r_codes = [x for x in rem.split(';') if x]
            a_codes = [x for x in add.split(';') if x]
            r_cats = [c for c in {cat_of(x) for x in r_codes} if c not in omit_cats]
            a_cats = [c for c in {cat_of(x) for x in a_codes} if c not in omit_cats]
            r_cats = [c for c in r_cats if c not in omit_cats]
            a_cats = [c for c in a_cats if c not in omit_cats]
            for rc in r_cats:
                for ac in a_cats:
                    rows_h.append({'removed_cat': rc, 'added_cat': ac})
        if not rows_h:
            return pd.DataFrame(columns=['category','p_add','p_remove','row_sum','col_sum'])
        mdh = pd.DataFrame(rows_h)
        counts = mdh.groupby(['removed_cat','added_cat']).size().reset_index(name='n')
        total_pairs = float(counts['n'].sum()) or 1.0
        row_sums = counts.groupby('removed_cat')['n'].sum()
        col_sums = counts.groupby('added_cat')['n'].sum()
        cats_all_k = sorted(set(row_sums.index.tolist()) | set(col_sums.index.tolist()))
        rows_k = []
        for c in cats_all_k:
            rows_k.append({
                'category': c,
                'p_remove': float(row_sums.get(c, 0.0)) / total_pairs,
                'p_add': float(col_sums.get(c, 0.0)) / total_pairs,
                'row_sum': int(row_sums.get(c, 0.0) or 0),
                'col_sum': int(col_sums.get(c, 0.0) or 0),
            })
        return pd.DataFrame(rows_k)

    def _pivot_for_dir(run_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float, list[str]]:
        dfk = _matches_df(run_dir, winners_only=True)
        rows_h = []
        for _, rr in dfk.iterrows():
            rem = str(rr.get('removed_foods', '') or '')
            add = str(rr.get('added_foods', '') or '')
            if not rem and not add:
                continue
            r_codes = [x for x in rem.split(';') if x]
            a_codes = [x for x in add.split(';') if x]
            r_cats = [c for c in {cat_of(x) for x in r_codes} if c not in omit_cats]
            a_cats = [c for c in {cat_of(x) for x in a_codes} if c not in omit_cats]
            r_cats = [c for c in r_cats if c not in omit_cats]
            a_cats = [c for c in a_cats if c not in omit_cats]
            for rc in r_cats:
                for ac in a_cats:
                    rows_h.append({'removed_cat': rc, 'added_cat': ac})
        if not rows_h:
            return pd.DataFrame(), np.array([]), np.array([]), 0.0, []
        mdh = pd.DataFrame(rows_h)
        counts = mdh.groupby(['removed_cat','added_cat']).size().reset_index(name='n')
        row_tot = counts.groupby('removed_cat')['n'].transform('sum')
        counts['p_row'] = counts['n'] / row_tot.replace(0, np.nan)
        pivot = counts.pivot(index='removed_cat', columns='added_cat', values='p_row')
        total_pairs = float(counts['n'].sum()) or 1.0
        rsum = counts.groupby('removed_cat')['n'].sum()
        csum = counts.groupby('added_cat')['n'].sum()
        cats = sorted(set(pivot.index.tolist()) | set(pivot.columns.tolist()))
        rarr = np.array([float(rsum.get(c, 0.0)) / total_pairs for c in cats], float)
        carr = np.array([float(csum.get(c, 0.0)) / total_pairs for c in cats], float)
        return pivot, rarr, carr, total_pairs, cats

    existings = [k for k, p in k_to_dir.items() if p and p.exists() and (p / 'substitution_matches.csv').exists()]
    if len(existings) < 2:
        return
    data_by_k = {k: _scatter_df_for_dir(p) for k, p in k_to_dir.items() if k in existings}
    if not data_by_k:
        return
    all_cats = sorted({c for dfk in data_by_k.values() for c in dfk['category'].tolist()})
    ncat = len(all_cats)
    try:
        import seaborn as sns  # type: ignore
        pal = sns.color_palette('tab20', ncat) if ncat <= 20 else sns.color_palette('hls', ncat)
    except Exception:
        pal = [(0.12,0.47,0.71)] * ncat
    cat2color = {c: pal[i % len(pal)] for i, c in enumerate(all_cats)}
    # Local pretty-name helper to avoid cross-scope issues
    def _pretty_local(c: str) -> str:
        mapping = {
            'snacks_sweets': 'Snacks',
            'fats_oils': 'Fats & oils',
            'condiments_sauces': 'Condiments & sauces',
            'mixed_dishes': 'Mixed dishes',
            'coffee_tea': 'Coffee & tea',
            'plain_water': 'Water',
            'diet_beverages': 'Diet beverages',
        }
        c0 = str(c)
        if c0 in mapping:
            return mapping[c0]
        return c0.replace('_',' ').capitalize() if c0 else c0

    piv_and_marg_by_k = {k: _pivot_for_dir(p) for k, p in k_to_dir.items() if k in existings}
    piv_by_k = {k: piv_and_marg_by_k[k][0] for k in piv_and_marg_by_k}
    marg_by_k = {k: {'row': piv_and_marg_by_k[k][1], 'col': piv_and_marg_by_k[k][2], 'cats': piv_and_marg_by_k[k][4]} for k in piv_and_marg_by_k}
    cats_union = sorted({c for piv in piv_by_k.values() for c in (list(piv.index) + list(piv.columns))}) if piv_by_k else []
    order = list(cats_union)
    if cats_union:
        try:
            import numpy as _np  # type: ignore
            from scipy.cluster.hierarchy import linkage, leaves_list  # type: ignore
            from scipy.spatial.distance import pdist  # type: ignore
            mats = []
            for _k, _piv in piv_by_k.items():
                P = _piv.reindex(index=cats_union, columns=cats_union).fillna(0.0).values
                S = 0.5 * (P + P.T)
                mats.append(S)
            if mats:
                S_avg = _np.mean(_np.stack(mats, axis=0), axis=0)
                D = pdist(S_avg, metric='euclidean')
                if not _np.allclose(D, 0.0):
                    Z = linkage(D, method='average')
                    leaves = leaves_list(Z)
                    order = [cats_union[i] for i in leaves]
        except Exception:
            order = list(cats_union)
    for k in list(piv_by_k.keys()):
        piv_by_k[k] = piv_by_k[k].reindex(index=order, columns=order).fillna(0.0) if order else piv_by_k[k].fillna(0.0)
        if k in marg_by_k and order:
            rmap = {c: float(v) for c, v in zip(marg_by_k[k]['cats'], marg_by_k[k]['row'])}
            cmap = {c: float(v) for c, v in zip(marg_by_k[k]['cats'], marg_by_k[k]['col'])}
            marg_by_k[k]['row'] = np.array([rmap.get(c, 0.0) for c in order], float)
            marg_by_k[k]['col'] = np.array([cmap.get(c, 0.0) for c in order], float)
            marg_by_k[k]['cats'] = list(order)
    vmax = 0.0
    for piv in piv_by_k.values():
        try:
            vmax = max(vmax, float(np.nanmax(piv.values)))
        except Exception:
            pass
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=False, sharey=False, constrained_layout=False)
    labels_by_k = {1: '1-hop Substitution Patterns', 2: '2-hop Substitution Patterns', 3: '3-hop Substitution Patterns'}

    def _theta_for_dir(d: Path) -> str:
        try:
            tfile = d / 'theta.txt'
            if tfile.exists():
                tval = tfile.read_text().strip()
                if tval:
                    return tval
            import re as _re
            m = _re.search(r'tradeoff_theta_([0-9p\.]+)_k[123]', str(d.name))
            if m:
                return m.group(1).replace('p', '.')
        except Exception:
            pass
        return ''
    theta_label_by_k = {k: _theta_for_dir(p) for k, p in k_to_dir.items()}

    for ridx, k in enumerate((1,2,3)):
        # Right column: scatter
        ax_sc = axes[ridx, 1]
        dfk = data_by_k.get(k, pd.DataFrame(columns=['category','p_add','p_remove','row_sum','col_sum']))
        if not dfk.empty:
            colors = [cat2color.get(c, (0.12,0.47,0.71)) for c in dfk['category']]
            mass = dfk['row_sum'].astype(float) + dfk['col_sum'].astype(float)
            m99 = float(np.nanpercentile(mass.to_numpy(), 99)) if len(mass) else 1.0
            m99 = m99 or 1.0
            sizes = 25.0 + 160.0 * (np.clip(mass.to_numpy() / m99, 0.0, 1.0))
            ax_sc.scatter(dfk['p_add'], dfk['p_remove'], s=sizes, c=colors, edgecolors='white', linewidths=0.5)
            xmax = float(np.nanmax(dfk['p_add'].to_numpy())) if len(dfk) else 0.0
            ymax = float(np.nanmax(dfk['p_remove'].to_numpy())) if len(dfk) else 0.0
            px = float(np.nanpercentile(dfk['p_add'].to_numpy(), 99)) if len(dfk) else 0.0
            py = float(np.nanpercentile(dfk['p_remove'].to_numpy(), 99)) if len(dfk) else 0.0
            lim = float(max(0.05, 1.05 * max(px, py)))
            lim_use = max(lim, xmax, ymax) + 0.03
            ax_sc.plot([0, lim_use], [0, lim_use], ls='--', lw=1.0, c='#aaaaaa')
            ax_sc.set_xlim(-0.01, lim_use)
            ax_sc.set_ylim(-0.01, lim_use)
            try:
                ax_sc.set_xticks(np.arange(0.0, lim_use + 0.0001, 0.10))
                ax_sc.set_yticks(np.arange(0.0, lim_use + 0.0001, 0.10))
            except Exception:
                pass
            ax_sc.set_aspect('equal', adjustable='box')
        ax_sc.set_title(labels_by_k[k], fontsize=12)
        ax_sc.set_ylabel('Probability of food removed', fontsize=12)
        ax_sc.set_xlabel('Probability of food added', fontsize=12)
        ax_sc.grid(alpha=0.15, linestyle=':')

        # Left column: heatmap
        ax_hm = axes[ridx, 0]
        piv = piv_by_k.get(k, pd.DataFrame())
        if piv is not None and not piv.empty:
            import numpy as _np
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable
            # Determine order safely
            cats_here = sorted(set(piv.index.tolist()) | set(piv.columns.tolist()))
            order_names = cats_here
            try:
                from scipy.cluster.hierarchy import linkage, leaves_list
                P_sq = piv.reindex(index=cats_here, columns=cats_here).fillna(0.0)
                S = 0.5 * (P_sq.values + P_sq.values.T)
                if S.size and not _np.allclose(S, 0.0):
                    Z = linkage(S, method='average')
                    order_idx = leaves_list(Z)
                    order_names = [cats_here[i] for i in order_idx]
            except Exception:
                order_names = cats_here
            piv_ord = piv.reindex(index=order_names, columns=order_names).fillna(0.0)
            vmax_local = float(_np.nanmax(piv_ord.values)) if _np.isfinite(_np.nanmax(piv_ord.values)) and _np.nanmax(piv_ord.values) > 0 else 1.0
            piv_disp = piv_ord.copy(); piv_disp.index = [_pretty_local(x) for x in piv_ord.index]; piv_disp.columns = [_pretty_local(x) for x in piv_ord.columns]
            # Draw heatmap regardless of dendrogram success
            k_gradients = {1: ['#ffffff', '#b4e1d3', '#68c3a6'], 2: ['#ffffff', '#fec6b1', '#fc8d62'], 3: ['#ffffff', '#c8d1e6', '#90a3cc']}
            from matplotlib.colors import LinearSegmentedColormap
            colors = k_gradients[k]
            n_bins = 100
            custom_cmap = LinearSegmentedColormap.from_list(f'custom_{k}', colors, N=n_bins)
            import seaborn as sns  # type: ignore
            hm = sns.heatmap(
                piv_disp,
                ax=ax_hm,
                cmap=custom_cmap, vmin=0.0, vmax=vmax_local,
                linewidths=0.3, linecolor='#eeeeee',
                cbar=False,
                square=True,
            )
            ax_hm.set_xticks(_np.arange(len(piv_disp.columns))+0.5)
            ax_hm.set_yticks(_np.arange(len(piv_disp.index))+0.5)
            ax_hm.set_xticklabels(list(piv_disp.columns), rotation=45, ha='right', fontsize=11)
            ax_hm.set_yticklabels(list(piv_disp.index), rotation=0, fontsize=11)
            # Try dendrogram and colorbar; failures should not hide heatmap
            try:
                from scipy.cluster.hierarchy import dendrogram
                fig_ref = ax_hm.figure
                pos = ax_hm.get_position()
                top_h = 0.10 * pos.height
                left_w = 0.10 * pos.width
                ax_top = fig_ref.add_axes([pos.x0, pos.y1, pos.width, top_h])
                ax_left = fig_ref.add_axes([pos.x1, pos.y0, left_w, pos.height])
                ax_top.set_axis_off(); ax_left.set_axis_off()
                try:
                    if 'Z' in locals():
                        dendrogram(Z, ax=ax_top, no_labels=True, color_threshold=0, above_threshold_color='black', link_color_func=lambda k: 'black')
                        dendrogram(Z, ax=ax_left, orientation='right', no_labels=True, color_threshold=0, above_threshold_color='black', link_color_func=lambda k: 'black')
                except Exception:
                    pass
                cax_w = 0.015
                cax = fig_ref.add_axes([pos.x1 + left_w + 0.005, pos.y0, cax_w, pos.height])
                sm = ScalarMappable(norm=Normalize(0.0, vmax_local), cmap=custom_cmap); sm.set_array([])
                cb = fig_ref.colorbar(sm, cax=cax)
                cb.set_label('Substitution Probability')
                cb.set_ticks(np.arange(0.0, vmax_local + 0.05, 0.1))
            except Exception:
                pass
        tlabel = theta_label_by_k.get(k) or ''
        if tlabel:
            ax_hm.set_title(f"θ={tlabel}", fontsize=11, loc='left')
        ax_hm.set_ylabel('Removed category', fontsize=12)
        ax_hm.set_xlabel('Added category', fontsize=12)

    try:
        fig.subplots_adjust(left=0.07, right=0.80, top=0.95, bottom=0.08, wspace=0.35, hspace=0.45)
        from matplotlib.lines import Line2D
        leg_ax = fig.add_axes([0.82, 0.10, 0.16, 0.40]); leg_ax.axis('off')
        pretty_names = [_pretty_local(c) for c in all_cats]
        handles = [Line2D([0],[0], marker='o', linestyle='', markersize=6,
                           markerfacecolor=cat2color[c], markeredgecolor='white', markeredgewidth=0.5)
                   for c in all_cats]
        leg = leg_ax.legend(handles, pretty_names, title='Food Categories', loc='center left', frameon=True, borderaxespad=0.0, handletextpad=0.4, labelspacing=0.4)
        try:
            leg.get_frame().set_edgecolor('#444444'); leg.get_frame().set_linewidth(0.8)
        except Exception:
            pass
    except Exception:
        plt.tight_layout()
    _ensure_dir(out_dir)
    fig.savefig(out_dir / 'figure_category_scatter_plus_heatmap_k123.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'figure_category_scatter_plus_heatmap_k123.svg', bbox_inches='tight')
    plt.close(fig)


def _to_lattice(H: np.ndarray, S: np.ndarray, B: np.ndarray, scale: int) -> list[tuple[int,int,int]]:
    W = np.vstack([H, S, B]).T
    denom = np.maximum(W.sum(axis=1, keepdims=True), 1e-9)
    W = W / denom
    L = np.rint(W * scale).astype(int)
    diffs = scale - L.sum(axis=1)
    for i, d in enumerate(diffs):
        if d != 0:
            j = int(np.argmax(L[i]))
            L[i, j] += int(d)
    return [(int(h), int(s), int(b)) for h, s, b in L]


# Map barycentric (Health=h, Cost savings=c, Stability=b) with h+c+b=1 to 2D triangle
def _bary_to_xy(h: np.ndarray, c: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = b * 1.0 + h * 0.5
    y = h * (np.sqrt(3) / 2.0)
    return np.column_stack([x, y])


def _discover_theta_dirs(base: Path) -> dict[float, Path]:
    theta_to_dirs: dict[float, Path] = {}
    if not base.exists() or not base.is_dir():
        return theta_to_dirs
    import re
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        m = re.match(r'^(?:tradeoff|policy)_theta_([0-9p\.]+)_k3(?:run)?$', name)
        if not m:
            continue
        t_str = m.group(1)
        t_val = float(t_str.replace('p', '.'))
        if (p / 'substitution_matches.csv').exists():
            theta_to_dirs[t_val] = p
    return theta_to_dirs


def _discover_theta_dirs_group_any(base: Path, tmp_base: Path) -> dict[float, Path]:
    """Discover theta run directories even when each k is saved separately (…_k1, …_k2, …_k3).

    If a canonical folder containing all ks exists (…_k3run), prefer it. Otherwise, group
    per-theta k-folders and materialize a temporary combined folder with concatenated
    substitution_matches.csv so downstream plotting (which expects a single dir per θ)
    sees all k values.
    """
    import re
    from pathlib import Path as _Path
    import pandas as _pd
    theta_to_dirs: dict[float, _Path] = {}
    if not base.exists() or not base.is_dir():
        return theta_to_dirs
    # First pass: canonical _k3run if present (will be overridden by grouped if we detect multiple ks)
    primary = _discover_theta_dirs(base)
    theta_to_dirs.update(primary)
    # Second pass: group any …_k{1,2,3} per theta (including when k3 exists) so we can combine when multiple ks are present
    groups: dict[str, list[_Path]] = {}
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        m = re.match(r'^(?:tradeoff|policy)_theta_([0-9p\.]+)_k([123])$', name)
        if not m:
            continue
        t_str = m.group(1)
        if (p / 'substitution_matches.csv').exists():
            groups.setdefault(t_str, []).append(p)
    # Materialize combined dirs for any theta with at least one k-folder
    for t_str, dirs in groups.items():
        if not dirs:
            continue
        t_val = float(t_str.replace('p', '.'))
        # If only one k-folder exists and no need to group, keep existing mapping unless absent
        if len(dirs) == 1 and t_val in theta_to_dirs:
            # Already mapped (likely to canonical k3run); skip combining since only one k folder was found
            continue
        # Otherwise, create tmp combined folder that concatenates all available ks
        tmp_dir = tmp_base / f"theta_combined_{t_str}_k123"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # Concatenate matches across ks
        dfs = []
        for d in dirs:
            try:
                df = _pd.read_csv(d / 'substitution_matches.csv')
                dfs.append(df)
            except Exception:
                pass
        if dfs:
            try:
                df_all = _pd.concat(dfs, ignore_index=True)
                df_all.replace([float('inf'), float('-inf')], _pd.NA, inplace=True)
                df_all.to_csv(tmp_dir / 'substitution_matches.csv', index=False)
                # Override any prior mapping (so that k=1 and k=2 are included)
                theta_to_dirs[t_val] = tmp_dir
            except Exception:
                pass
    return theta_to_dirs


def main():
    ap = argparse.ArgumentParser(description="Plot dynamic nutrition–cost tradeoffs from substitution analysis")
    ap.add_argument("--input_dir", type=str, required=True, help="Directory containing substitution_matches.csv")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory to write figure panels")
    ap.add_argument("--mode", type=str, default="k_from_input", choices=["k_from_input","optimal_theta"],
                    help="k_from_input: plots for one run folder (category heatmap with --with_category_heatmap, ternary with --with_ternary); "
                         "optimal_theta: knee θ per k, Figure 4B and Figure 4C-D from the θ runs under --theta_runs_base (ternary with --with_ternary)")
    ap.add_argument("--with_category_heatmap", action="store_true", help="Also render category substitution heatmap in k_from_input mode")
    ap.add_argument("--k_dirs", type=str, default="", help="Optional explicit mapping for triptych: 'k1=PATH,k2=PATH,k3=PATH'. Used with --mode k_from_input --with_ternary --with_category_heatmap: figure_category_scatter_plus_heatmap_k123 is built from these dirs and saved into out_dir, regardless of input_dir naming.")
    ap.add_argument("--with_ternary", action="store_true", help="Render ternary plots (optional). If not set, ternary plots are skipped.")
    ap.add_argument("--theta_runs_base", type=str, default="", help="Base directory containing multiple θ run folders (defaults to parent of input_dir)")
    ap.add_argument("--ks", type=str, default="1,2,3", help="Comma-separated k values to include (default: 1,2,3)")
    ap.add_argument("--ternary_style", type=str, default="objective", choices=["objective","density","raw"], help="optimal_theta ternary style: objective (default), density, or raw (unanchored H,S,E with density color)")
    ap.add_argument("--raw_scale", type=str, default="per_k_p95", choices=["per_k_p95","per_k_max","global_p95","none"], help="Scaling mode for raw ternary before share normalization")
    ap.add_argument("--raw_use_winners", action="store_true", help="Use winners-only cohort (selected==True) for raw ternary")
    ap.add_argument("--overwrite_k3run", action="store_true", help="Also overwrite figure_ternary_hcb_by_k_substitution_k3run.* with the generated optimal-θ figure")
    ap.add_argument("--dump_debug", action="store_true", help="Write CSVs of data actually used to render figures")
    args = ap.parse_args()
    global _DUMP_DEBUG
    _DUMP_DEBUG = bool(getattr(args, 'dump_debug', False))

    try:
        from datetime import datetime
        log_path = Path.cwd() / 'logs' / 'substitutions' / 'substitutions.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _logger = _setup_logger('substitutions', str(log_path), file_mode='a')
        sys.stdout = _StreamToLogger(_logger, logging.INFO)
        sys.stderr = _StreamToLogger(_logger, logging.ERROR)
        _logger.info(
            "\n" + "=" * 60 + "\n[session] %s plot_substitution_figures %s",
            datetime.now().isoformat(),
            " ".join(sys.argv),
        )
    except Exception:
        pass

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    # optimal_theta: knee θ per k, Figure 4B and 4C-D (ternary only with --with_ternary)
    if args.mode == "optimal_theta":
        base = Path(args.theta_runs_base) if args.theta_runs_base else input_dir.parent
        # Discover canonical k3run dirs; if absent, group per-theta k1/k2/k3 into a temporary combined dir
        tmp_base = (out_dir / '_tmp_theta_grouped')
        theta_to_dirs = _discover_theta_dirs_group_any(base, tmp_base)
        if not theta_to_dirs:
            print(f"No theta run directories discovered under {base}")
            return
        try:
            ks = tuple(int(k.strip()) for k in str(args.ks).split(',') if k.strip())
        except Exception:
            ks = (1,2,3)
        used = _compute_knee_optimal_theta_per_k(theta_to_dirs, ks=ks, winners_only=True)
        # The .txt (a Python dict literal) is read by scripts/generate_manuscript_numbers_log.py,
        # the .json by `make figure4a` and analyses/primary_outcome_cis.py.
        try:
            txt_path = out_dir / "optimal_theta_by_k.txt"
            txt_path.write_text(str(used), encoding="utf-8")
            json_path = out_dir / "optimal_theta_by_k.json"
            json_path.write_text(json.dumps(used, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as e:
            print(f"[warn] Failed to write optimal_theta_by_k artifacts: {e}")

        if args.with_ternary:
            used = generate_ternary_at_optimal_thetas(
                theta_to_dirs,
                out_dir,
                ks=ks,
                ternary_style=args.ternary_style,
                raw_use_winners=args.raw_use_winners,
                raw_scale=args.raw_scale,
            )
            print("Optimal θ by k:", used)
        else:
            print("[info] --with_ternary not set; skipping optimal-θ ternary plots.")
        # Also produce the cost-benefit figure for the same theta runs
        try:
            figure_cost_benefit_points_by_theta_and_k(theta_to_dirs, out_dir, ks=ks, winners_only=True, bootstraps=300)
        except Exception as e:
            print(f"Failed to render cost-benefit figure: {e}")
        # Category scatter + heatmap at each k's knee theta (nearest swept theta if not swept).
        try:
            k_to_dir = {k: theta_to_dirs[min(theta_to_dirs, key=lambda t: abs(t - th))] for k, th in used.items()}
            figure_category_triptych_from_k_dirs(k_to_dir, out_dir)
            print(f"Category triptych written to: {out_dir}")
        except Exception as e:
            print(f"Failed to render category triptych: {e}")
        if args.overwrite_k3run:
            try:
                src_png = out_dir / 'figure_ternary_hcb_by_k_optimal_theta.png'
                src_svg = out_dir / 'figure_ternary_hcb_by_k_optimal_theta.svg'
                dst_png = out_dir / 'figure_ternary_hcb_by_k_substitution_k3run.png'
                dst_svg = out_dir / 'figure_ternary_hcb_by_k_substitution_k3run.svg'
                if src_png.exists():
                    shutil.copyfile(src_png, dst_png)
                if src_svg.exists():
                    shutil.copyfile(src_svg, dst_svg)
            except Exception:
                pass
        return

    # If ternary plots are not requested, skip directly to optional heatmap
    if not args.with_ternary:
        if args.with_category_heatmap:
            try:
                figure_category_heatmap(input_dir, out_dir)
            except Exception as e:
                print(f"[warn] Failed to render category heatmap: {e}")
        print(f"Saved new figures to: {out_dir}")
        return

    # Only generate gradient-style HCB ternary by k_substitution with Health gain (%) and fixed scaling
    try:
        import ternary  # type: ignore
        from sklearn.neighbors import KernelDensity
    except Exception:
        print("Missing dependencies for gradient ternary; skipping ternary plots.")
        # Still allow category heatmap to render
        if args.with_category_heatmap:
            try:
                figure_category_heatmap(input_dir, out_dir)
            except Exception as e:
                print(f"[warn] Failed to render category heatmap: {e}")
        print(f"Saved new figures to: {out_dir}")
        return
    df = _read_matches(input_dir).replace([np.inf, -np.inf], np.nan)
    # Prefer explicit k_substitution if present
    if 'k_substitution' in df.columns:
        df['k'] = pd.to_numeric(df['k_substitution'], errors='coerce')
    # Ensure ΔRDI percent available for health component and stats
    df['dRDI_pct'] = pd.to_numeric(df.get('dRDI_pct', 100.0 * df.get('dRDI', df.get('improvement', np.nan))), errors='coerce')
    if {'real_cost','dCost'}.issubset(df.columns):
        with np.errstate(divide='ignore', invalid='ignore'):
            df['dCost_pct'] = 100.0 * (pd.to_numeric(df['dCost'], errors='coerce') / pd.to_numeric(df['real_cost'], errors='coerce'))
    # Stability from portion/composition; if missing, derive from meal vectors
    if not {'portion_shift_pct','composition_change_pct'}.issubset(df.columns):
        try:
            data = _load_meals_and_features(_find_data_exports(Path(input_dir)))
        except Exception:
            data = None
        portion, comp = [], []
        for _, r in df.iterrows():
            mt = r.get('Meal Type'); gi = int(r.get('gen_idx', -1)); ri = int(r.get('real_idx', -1))
            if (data is not None and mt in data.get('gen_by_type', {}) and mt in data.get('real_by_type', {}) and 0 <= gi < len(data['gen_by_type'][mt]) and 0 <= ri < len(data['real_by_type'][mt])):
                gm = data['gen_by_type'][mt][gi]; rm = data['real_by_type'][mt][ri]
                l1 = float(np.sum(np.abs(gm - rm))); base = float(np.sum(rm))
                ps = (l1 / max(base, 1e-6)) * 100.0
                portion.append(float(np.clip(ps, 0.0, 100.0)))
                r_items = (rm > 0).astype(np.int32); g_items = (gm > 0).astype(np.int32)
                removed = int(np.sum((r_items == 1) & (g_items == 0))); added = int(np.sum((r_items == 0) & (g_items == 1)))
                items_in_real = int(max(np.sum(r_items), 1))
                comp_pct = 100.0 * (max(added, removed) / items_in_real)
                comp.append(float(np.clip(comp_pct, 0.0, 100.0)))
            else:
                portion.append(np.nan); comp.append(np.nan)
        df['portion_shift_pct'] = portion
        df['composition_change_pct'] = comp
    # Health component uses ΔRDI (%)
    H = np.maximum(0.0, pd.to_numeric(df['dRDI_pct'], errors='coerce').to_numpy(float))
    S = np.maximum(0.0, -pd.to_numeric(df['dCost_pct'], errors='coerce').to_numpy(float))
    P = pd.to_numeric(df['portion_shift_pct'], errors='coerce').to_numpy(float)
    C = pd.to_numeric(df['composition_change_pct'], errors='coerce').to_numpy(float)
    # Debug: raw feature ranges for k_from_input
    def _rng(x):
        x = np.asarray(x, float)
        return (float(np.nanmin(x)) if np.isfinite(np.nanmin(x)) else np.nan,
                float(np.nanmax(x)) if np.isfinite(np.nanmax(x)) else np.nan)
    print(f"[debug k_from_input] raw dRDI_pct range: {_rng(df.get('dRDI_pct'))}")
    print(f"[debug k_from_input] raw dCost_pct range: {_rng(df.get('dCost_pct'))}")
    print(f"[debug k_from_input] raw portion_shift_pct range: {_rng(P)}")
    print(f"[debug k_from_input] raw composition_change_pct range: {_rng(C)}")

    # Similarity (stability/ease) is simply 1 − Effort
    # Without an effort value, use the portion/composition similarity instead
    S_port = 1.0 - np.clip(pd.to_numeric(df['portion_shift_pct'], errors='coerce').to_numpy(float) /
                        max(float(np.nanpercentile(df['portion_shift_pct'], 99)), 1e-9), 0.0, 1.0)
    S_comp = 1.0 - np.clip(pd.to_numeric(df['composition_change_pct'], errors='coerce').to_numpy(float) /
                        max(float(np.nanpercentile(df['composition_change_pct'], 99)), 1e-9), 0.0, 1.0)
    valid = (~np.isnan(S_port)).astype(float) + (~np.isnan(S_comp)).astype(float)
    B_fallback = (np.nan_to_num(S_port) + np.nan_to_num(S_comp)) / np.maximum(valid, 1.0)
    B_fallback = np.clip(B_fallback, 0.0, 1.0)

    # The matches files carry no effort column
    if 'effort_kstable' in df.columns:
        E = pd.to_numeric(df['effort_kstable'], errors='coerce').to_numpy(float)
    else:
        E = np.full(len(df), np.nan)
    B = np.where(np.isfinite(E), 1.0 - E, B_fallback)

    # Per-k robust normalization so each panel uses its own 95th-pct anchor
    kvals = pd.to_numeric(df.get('k', pd.Series(dtype=float)), errors='coerce').to_numpy()
    Hn_all = np.zeros_like(H, dtype=float)
    Sn_all = np.zeros_like(S, dtype=float)
    Bn_all = np.clip(B, 0.0, 1.0)

    for kk in [1, 2, 3]:
        mk = (kvals == kk) & np.isfinite(H) & np.isfinite(S)
        if not np.any(mk):
            continue
        Hpos = H[mk][H[mk] > 0]
        Spos = S[mk][S[mk] > 0]
        Href = float(np.percentile(Hpos, 95)) if Hpos.size else float(np.percentile(H[mk], 95) or 1.0)
        Sref = float(np.percentile(Spos, 95)) if Spos.size else float(np.percentile(S[mk], 95) or 1.0)
        Hn_all[mk] = np.clip(H[mk] / max(Href, 1e-9), 0.0, 1.5)
        Sn_all[mk] = np.clip(S[mk] / max(Sref, 1e-9), 0.0, 1.5)

    print(f"[debug k_from_input] normalized Hn range: {_rng(Hn_all)}  Sn range: {_rng(Sn_all)}  Bn range: {_rng(Bn_all)}")
    kvals = pd.to_numeric(df.get('k', pd.Series(dtype=float)), errors='coerce').to_numpy()
    ks = [k for k in [1,2,3] if np.sum(kvals==k) > 0]
    if not ks:
        print("No k_substitution in {1,2,3}; nothing to plot.")
        print(f"Saved new figures to: {out_dir}")
        return

    # Build panels and anchored value heatmaps on triangular lattice (colour scale per panel)
    scale = 100
    panels = [(f"{int(k)} swaps", (kvals==k)) for k in ks]
    ncols = min(3, len(panels)); nrows = int(np.ceil(len(panels)/ncols))
    fig = plt.figure(figsize=(9.2*ncols, 6.4*nrows))
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    k_palette = {1: '#1b9e77', 2: '#d95f02', 3: '#7570b3'}
    for i, (title, msk) in enumerate(panels, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
        tax.clear_matplotlib_ticks()
        try:
            k_here = int(str(title).split()[0])
        except Exception:
            k_here = None
        k_color = k_palette.get(k_here, '0.25')
        tax.boundary(linewidth=1.8, axes_colors={'l': k_color, 'r': k_color, 'b': k_color})
        tax.gridlines(multiple=10, color="0.85", linewidth=0.6)
        # Vertex labels at corners (add more vertical padding)
        tax.left_corner_label('Cost savings', fontsize=16, offset=0.20)
        tax.right_corner_label('Swap simplicity', fontsize=16, offset=0.20)
        tax.top_corner_label('Nutrient adequacy gain', fontsize=16, offset=0.10)
        # Axis labels indicate direction (mapped to edges)
        tax.bottom_axis_label('More cost savings', fontsize=12, offset=0.08)
        tax.right_axis_label('Higher swap simplicity', fontsize=12, offset=0.10)
        tax.left_axis_label('Higher nutrient adequacy (ΔRDI)', fontsize=12, offset=0.10)
        # Show only end tick labels (0 and 100), hide intermediates
        # Show only the first and last tick values (0 and 100)
        tax.ticks(axis='lbr', multiple=100, linewidth=0.0, tick_formats="%d")



        m = msk & np.isfinite(Hn_all) & np.isfinite(Sn_all) & np.isfinite(Bn_all)
        print(f"[debug k_from_input] k={int(str(title).split()[0])} included={int(np.sum(m))} / raw={int(np.sum(msk))}")
        if np.sum(m) > 0:
            Hn, Cn, Bn = Hn_all[m], Sn_all[m], Bn_all[m]
            # Convert to shares and lattice coordinates
            denom = Hn + Cn + Bn + 1e-9
            Hb, Cb, Bb = Hn/denom, Cn/denom, Bn/denom
            print(f"[debug optimalθ] k={k_here} shares Hb range: {_rng(Hb)} Cb range: {_rng(Cb)} Bb range: {_rng(Bb)}")
            # python-ternary expects (left, right, bottom) = (Savings, Similarity, Health)
            coords = _to_lattice(Cb, Bb, Hb, scale=scale)
            # Anchored value per point
            v_arr = (Hn + Cn) / (1e-6 + (1.0 - Bn))  # higher = more health+savings per unit effort

            use_fallback = False
            if not np.any(v_arr > 1e-8):
                use_fallback = True

            print(f"[debug optimalθ] k={k_here} color v range: {_rng(v_arr)}  coords sample: {coords[:3] if len(coords)>2 else coords}")
            # Optional debug dump: per-point normalized inputs
            if _DUMP_DEBUG:
                try:
                    out = pd.DataFrame({
                        'k': int(k_here) if k_here is not None else np.nan,
                        'Hn': Hn, 'Sn': Cn, 'Bn': Bn,
                        'Hb': Hb, 'Cb': Cb, 'Bb': Bb,
                        'value': np.clip(v_arr, 0.0, 1.0)
                    })
                    _dump_df(Path(out_dir) / f"debug_ternary_k_from_input_k{k_here}.csv", out)
                except Exception:
                    pass
            # Aggregate by coordinate using mean; then apply multi-pass neighbor smoothing
            from collections import defaultdict
            sum_grid: dict[tuple[int,int,int], float] = defaultdict(float)
            count_grid: dict[tuple[int,int,int], int] = defaultdict(int)
            for (h, c, b), v in zip(coords, v_arr):
                key = (int(h), int(c), int(b))
                sum_grid[key] += float(np.clip(v, 0.0, 1.0))
                count_grid[key] += 1
            grid = {k: (sum_grid[k] / max(count_grid[k], 1)) for k in sum_grid}
            # multi-pass smoothing for gradient-like look
            grid_sm = _smooth_lattice_grid(grid, scale, passes=3, include_self=True)
            # robust per-panel normalization so color uses the full 0–1 range
            import numpy as _np
            _vals = _np.array(list(grid_sm.values()), float)
            pos = _vals[_vals > 0]
            if pos.size:
                vmax98 = float(_np.percentile(pos, 98)) or 1.0
                grid_sm = {k: float(_norm01(v, float(vmax98), gamma=0.7)) for k, v in grid_sm.items()}
            else:
                use_fallback = True
            try:
                base_rgb = to_rgb(k_color)
                panel_cmap = LinearSegmentedColormap.from_list(
                    f'hcb_k_{k_here}' if k_here is not None else 'hcb_k_default',
                    [(0.96,0.96,0.96), (0.98*base_rgb[0],0.98*base_rgb[1],0.98*base_rgb[2]), base_rgb],
                    N=256
                )
            except Exception:
                panel_cmap = plt.cm.magma
            # If the smoothed grid is effectively empty, fall back to density-based heatmap
            import numpy as _np
            _nz = _np.array(list(grid_sm.values()), float)
            _nz = _nz[_np.isfinite(_nz)]
            use_fallback = (_nz.size == 0) or (float(_np.nanmax(_nz)) <= 1e-8)
            if use_fallback:
                XY = _bary_to_xy(Hb, Cb, Bb)
                if XY.shape[0] > 0:
                    # adaptive bandwidth — spreads a near-point/line cloud enough to be visible
                    bw = 0.35 * float(np.mean(np.std(XY, axis=0)))
                    if not np.isfinite(bw) or bw < 0.12:
                        bw = 0.18
                    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(XY)
                    probe = _np.exp(kde.score_samples(XY)) if XY.shape[0] else _np.array([1.0])
                    vmax98 = float(_np.percentile(probe, 98)) if probe.size else 1.0
                    def _hf(p):
                        c, b, h = p
                        s = float(c + b + h) or 1.0
                        h, c, b = h/s, c/s, b/s
                        xy = _bary_to_xy(_np.array([h]), _np.array([c]), _np.array([b]))
                        v = float(_np.exp(kde.score_samples(xy.reshape(1, -2))))
                        return float(_norm01(v, float(max(vmax98, 1e-9)), gamma=0.7))
                    _render_tri_heatmap_as_image(ax, _hf, panel_cmap, scale)
                    import matplotlib as _mpl
                    sm = _mpl.cm.ScalarMappable(cmap=panel_cmap); sm.set_clim(0.0, 1.0)
                    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
                    cbar.ax.set_title("Benefit–effort efficiency", fontsize=16, pad=8)
                    cbar.ax.tick_params(labelsize=14)
                else:
                    hm = tax.heatmap(
                        grid_sm, scale=scale, style='triangular',
                        cmap=panel_cmap, colorbar=False, vmin=0.0, vmax=1.0, scientific=False
                    )
                    try:
                        hm.set_rasterized(True)
                    except Exception:
                        pass
                    import matplotlib as _mpl
                    sm = _mpl.cm.ScalarMappable(cmap=panel_cmap); sm.set_clim(0.0, 1.0)
                    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
                    cbar.ax.set_title("Benefit–effort efficiency", fontsize=16, pad=8)
                    cbar.ax.tick_params(labelsize=14)
            else:
                hm = tax.heatmap(
                    grid_sm, scale=scale, style='triangular',
                    cmap=panel_cmap, colorbar=False, vmin=0.0, vmax=1.0, scientific=False
                )
                try:
                    hm.set_rasterized(True)
                except Exception:
                    pass
                # Create a single explicit colorbar for this panel so we control title and ticks
                import matplotlib as _mpl
                sm = _mpl.cm.ScalarMappable(cmap=panel_cmap); sm.set_clim(0.0, 1.0)
                cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
                cbar.ax.set_title("Benefit–effort efficiency", fontsize=16, pad=8)
                cbar.ax.tick_params(labelsize=14)
        # Titles with summary stats
        # Panel mask for stats in original df index space
        m_df = pd.to_numeric(df.get('k', pd.Series(dtype=float)), errors='coerce') == int(str(title).split()[0])
        N = int(np.sum(m_df))
        print(f"[k_from_input] k={int(str(title).split()[0])}: N={N}")

        dRDI_median_pct = float(np.nanmedian(pd.to_numeric(df.loc[m_df, 'dRDI_pct'], errors='coerce')))
        dCost_median_pct = float(np.nanmedian(pd.to_numeric(df.loc[m_df, 'dCost_pct'], errors='coerce')))
        stab_median_pct = float(np.nanmedian(100.0 - (0.5 * pd.to_numeric(df.loc[m_df, 'portion_shift_pct'], errors='coerce') +
                                                        0.5 * pd.to_numeric(df.loc[m_df, 'composition_change_pct'], errors='coerce'))))

        clean_title = f"{int(str(title).split()[0])}-hop substitutions"
        tax.set_title(f"{clean_title}\nN={N}  ΔRDI={dRDI_median_pct:+.1f}%  ΔCost={dCost_median_pct:+.1f}%  Stability={stab_median_pct:.0f}%",
                      pad=12, fontsize=18, color='#222222', fontweight='bold')
        # Keep gridlines; do not re-add ticks to avoid intermediate labels
        tax.clear_matplotlib_ticks(); tax._redraw_labels()
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_frame_on(False)

    fig.set_facecolor("white")

    _ensure_dir(out_dir)
    fig.savefig(out_dir / 'figure_ternary_hcb_by_k_substitution_k3run.png', dpi=300)
    fig.savefig(out_dir / 'figure_ternary_hcb_by_k_substitution_k3run.svg', dpi=300)

    # Optionally render category heatmap and triptych via explicit k_dirs mapping
    if args.with_category_heatmap:
        figure_category_heatmap(input_dir, out_dir)
        # If an explicit k mapping was provided, build the triptych using those paths directly into out_dir
        kmap_str = str(getattr(args, 'k_dirs', '') or '').strip()
        if kmap_str:
            k_to_dir: dict[int, Path] = {}
            for part in kmap_str.split(','):
                part = part.strip()
                if not part or '=' not in part:
                    continue
                key, pstr = part.split('=', 1)
                key = key.strip().lower(); p = Path(pstr.strip())
                if key == 'k1': k_to_dir[1] = p
                if key == 'k2': k_to_dir[2] = p
                if key == 'k3': k_to_dir[3] = p
            if any((p and (p/ 'substitution_matches.csv').exists()) for p in k_to_dir.values()):
                figure_category_triptych_from_k_dirs(k_to_dir, out_dir)

    print(f"Saved new figures to: {out_dir}")


if __name__ == "__main__":
    main()