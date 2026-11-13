#!/usr/bin/env python3
"""Figure 3 plots written by `cli/evaluate.py` (`make eval`) into `<run>/figures/`.

- `_plot_cluster_deviation_boxplots`: per-archetype RDI deviation of real vs generated meals
  (Figure 3C), `boxplots_rdi_deviation_real_vs_generated.{png,svg}` plus `..._stats.csv`.
- `create_spider_plots`: 3x3 spider-plot grid of nutrition metrics, macronutrients and
  micronutrients per meal type (Figure 3B), `spider_plots_comprehensive.{png,svg}`.
- `create_unified_plot_layout`: combined preview `fig_main_cvae_rdi.png` (not used in the paper).
"""

from __future__ import annotations

import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path


# ---------- Spider-plot layout helpers (compact & crisp) ----------
import textwrap as _tw

def _wrap_label(txt: str, width: int = 11) -> str:
    return "\n".join(_tw.wrap(str(txt), width=width, break_long_words=False, break_on_hyphens=False)) or str(txt)

def _beautify_polar_compact(ax, labels, *,
                            wrap=11,
                            tick_labelsize=22,     # larger x labels
                            r_ticklabelsize=18,    # larger r labels
                            pad=15):               # more pixels away from the web
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=r_ticklabelsize)

    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    ax.set_xticks(angles)
    ax.set_xticklabels([_wrap_label(l, wrap) for l in labels],
                       fontsize=tick_labelsize, linespacing=1.0,
                       ha='center', va='center')  # center align labels
    ax.tick_params(axis="x", pad=pad)

    # Put r labels where they won't clash with long xticks
    ax.set_rlabel_position(67.5)
    ax.grid(alpha=0.5, lw=0.8)

def _beautify_polar_diagonal(ax, labels, *,
                            wrap=11,
                            tick_labelsize=22,     # larger x labels
                            r_ticklabelsize=18,    # larger r labels
                            pad=15):               # more pixels away from the web
    """Beautify polar plot with diagonal axis orientation (45-degree offset)"""
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=r_ticklabelsize)

    # Set diagonal orientation (45-degree offset)
    ax.set_theta_offset(np.pi / 4.0)  # 45 degrees instead of 90
    ax.set_theta_direction(-1)

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    ax.set_xticks(angles)
    ax.set_xticklabels([_wrap_label(l, wrap) for l in labels],
                       fontsize=tick_labelsize, linespacing=1.0,
                       ha='center', va='center')  # center align labels
    ax.tick_params(axis="x", pad=pad)

    # Put r labels where they won't clash with long xticks
    ax.set_rlabel_position(67.5)
    ax.grid(alpha=0.5, lw=0.8)


# ====== Unified scoring helpers (0–1, where 1 = meets target / desirable) ======
import numpy as _np

def _score_max_target(x, M):
    """Limit: lower is better. x <= M ⇒ score≈1."""
    x = _np.asarray(x, float); M = _np.asarray(M, float)
    with _np.errstate(divide="ignore", invalid="ignore"):
        s = M / x
    return _np.clip(s, 0.0, 1.0)

def _score_band_pct(x_pct, L, U):
    """AMDR band: 1 inside [L,U], linear falloff to 0 outside."""
    x = _np.asarray(x_pct, float); L = float(L); U = float(U)
    s = _np.ones_like(x)
    left = x < L; right = x > U
    s[left]  = _np.clip(x[left] / L, 0.0, 1.0)
    s[right] = _np.clip((100. - x[right]) / (100. - U), 0.0, 1.0)
    return s

# Aggregation used for the spider-plot polygons:
def _safe_median(arr):
    """Safe median that handles NaN values"""
    arr = _np.asarray(arr, float)
    arr = arr[~_np.isnan(arr)]
    if len(arr) == 0:
        return 0.0
    return float(_np.median(arr))

# Each spider-plot vertex is the median of the per-meal scores.
_PLOT_CENTER = _safe_median


# --- Helpers to make metrics robustly 0–1 and proper 1-D arrays ---

def _as_1d_array(x, n_rows):
    """Return a 1-D float array. If x is a scalar, broadcast to length n_rows."""
    a = _np.asarray(x, float)
    if a.ndim == 0:
        return _np.full(int(n_rows), float(a))
    return a.reshape(-1)

def _mer_to_score01(mer_like):
    """
    Ensure MER is a 0–1 'goodness' score.
    If input already in [0,1], clip & return.
    If it's an excess ratio (1 = at limit, >1 = worse), map via 1/ratio then clip to [0,1].
    """
    a = _np.asarray(mer_like, float)
    if a.size == 0:
        return a
    mx = _np.nanmax(a)
    if _np.isfinite(mx) and mx <= 1.0:
        return _np.clip(a, 0.0, 1.0)
    with _np.errstate(divide="ignore", invalid="ignore"):
        s = 1.0 / _np.maximum(a, 1e-6)
    return _np.clip(_np.nan_to_num(s, nan=0.0, posinf=1.0), 0.0, 1.0)



class VisualizationManager:
    """Figure 3 plotting entry points called by `cli/evaluate.py`."""
    
    def __init__(self, processed: Dict, code2name: Dict[str, str]):
        self.processed = processed
        self.code2name = code2name
    
    def plot_main_figure(self, real_aligned: Dict[str, np.ndarray], 
                        gen_aligned: Dict[str, np.ndarray], 
                        adj_by_type: Dict[str, np.ndarray],
                        clusters_aligned: Dict[str, np.ndarray],
                        figures_dir: Path):
        """Write fig_main_cvae_rdi.png and, on the way, the box-plot and spider-plot panels.

        `adj_by_type` is unused. Errors are turned into a warning.
        """
        print("→ Creating unified figures...")
        figures_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Create unified plot layout
            create_unified_plot_layout(
                meal_types=[mt for mt in ["Breakfast", "Lunch", "Dinner"] if mt in real_aligned],
                real_data=real_aligned,
                gen_data=gen_aligned,
                real_clusters=clusters_aligned,
                gen_clusters=clusters_aligned,
                W=self.processed["W"],
                nutrient_names=self.processed["nutrient_names"],
                RDI=self.processed["RDI"],
                meal_fraction_map=self.processed["meal_fraction_map"],
                food_to_group=self.processed["food_to_group"],
                feature_names=self.processed.get("feature_names"),
                limits=self.processed["limits"],
                out_path=str(figures_dir / "fig_main_cvae_rdi.png"),
                figsize=(24, 16)
            )
            print(f"✓ Unified main figure saved to {figures_dir}")
        except Exception as e:
            warnings.warn(f"Unified figure creation failed: {e}")
    
    def plot_cluster_deviation_boxplots(self, real_aligned: Dict[str, np.ndarray],
                                      gen_aligned: Dict[str, np.ndarray],
                                      clusters_aligned: Dict[str, np.ndarray],
                                      figures_dir: Path):
        """Write boxplots_rdi_deviation_real_vs_generated.{png,svg} and its _stats.csv (Figure 3C)."""
        def _cluster_name(meal, cid: int) -> str:
            # Use human-friendly cluster names when available
            try:
                from cli.run_multi_meal_cvae import _get_cluster_name as _human
                return _human(meal, int(cid))
            except Exception:
                return f"Cluster {int(cid)}"

        _plot_cluster_deviation_boxplots(
            meal_types=[mt for mt in ["Breakfast", "Lunch", "Dinner"] if mt in real_aligned],
            real_data=real_aligned,            # aligned grams
            gen_data=gen_aligned,              # aligned grams
            real_clusters=clusters_aligned,    # aligned cluster ids
            gen_clusters=clusters_aligned,     # aligned cluster ids (pairwise match)
            W=self.processed["W"],
            nutrient_names=self.processed["nutrient_names"],
            RDI=self.processed["RDI"],
            meal_fraction_map=self.processed["meal_fraction_map"],
            get_cluster_name_fn=_cluster_name,
            include_nutrients=None,            # or a list like ["energy_kcal","protein_g",...]
            deviation_mode="log2",              # <-- shows over/under clearly and symmetrically
            clip=2.0,                           # cap at ±2 in log2 space (i.e., 4×/¼×)
            out_path=str(figures_dir / "boxplots_rdi_deviation_real_vs_generated.png"),
        )
        print(f"✓ Deviation boxplots saved to {figures_dir}")


def _bxp_stats_from_samples(samples: np.ndarray, label: str):
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return None
    q1, med, q3 = np.percentile(s, [25, 50, 75])
    lo, hi      = np.percentile(s, [10, 90])   # whiskers as p10–p90 (robust)
    return dict(label=label, q1=q1, med=med, q3=q3, whislo=lo, whishi=hi, fliers=[])

def _plot_cluster_deviation_boxplots(
    *,
    meal_types: list[str],
    real_data: dict[str, np.ndarray],         # X_real[meal] (N x D grams)
    gen_data: dict[str, np.ndarray],          # X_gen[meal]  (N x D grams)
    real_clusters: dict[str, np.ndarray],     # cluster ids per real row
    gen_clusters: dict[str, np.ndarray],      # cluster ids per gen row
    W: np.ndarray,
    nutrient_names: list[str],
    RDI: np.ndarray,
    meal_fraction_map: dict[str, float],
    get_cluster_name_fn=None,                 # callable (meal_name, cluster_id) -> str
    include_nutrients: list[str] | None = None,
    figsize=(20, 22),
    out_path: str | None = None,
    deviation_mode: str = "log2",
    clip: float | None = 2.0,
    left_margin: float = 0.40,                # left margin for outside labels
    x_bracket: float = -0.065,                # bracket position (axes fraction)
    x_label: float = -0.155                   # label position (axes fraction)
):
    """
    Single aggregated horizontal boxplot with different colors for each meal type.
    Y-axis lists clusters; X-axis is the deviation score.

    Each box summarizes `per_meal_deviation_from_rdi` over the real or generated meals of one
    archetype; pairs are compared with Mann-Whitney U and BH-FDR. Writes `out_path` (PNG),
    the same figure as SVG, and `<stem>_stats.csv`. `left_margin`, `x_bracket` and `x_label`
    are not used: the bracket and label positions are computed from the tick-label widths.
    """
    if get_cluster_name_fn is None:
        get_cluster_name_fn = lambda meal, cid: f"Cluster {int(cid)}"

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[1.0, 0.22],  # plots + legend
        wspace=0.12
    )

    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[0, 1])


    # Colors (same as spider plots)
    meal_colors = {
        'Breakfast': {'real': '#1f4e79', 'gen': '#5b9bd5'},
        'Lunch':     {'real': '#2d5016', 'gen': '#70ad47'},
        'Dinner':    {'real': '#d2691e', 'gen': '#ff8c00'},
    }
    # vertical spacing between paired boxes along Y
    dy = 0.32

    # Horizontal significance connectors (vertical bars) helper
    def _add_significance_markers_h(ax, rows_r_g, pos_real, pos_gen):
        """
        rows_r_g: list of (r_dev, g_dev) arrays, aligned 1:1 with the plotted rows
        """
        try:
            from scipy import stats
            from statsmodels.stats.multitest import multipletests
        except Exception:
            return

        pvals, idx = [], []
        for i, (r_dev, g_dev) in enumerate(rows_r_g):
            if r_dev is None or g_dev is None or len(r_dev) == 0 or len(g_dev) == 0:
                continue
            try:
                _, p = stats.mannwhitneyu(r_dev, g_dev, alternative="two-sided")
                pvals.append(p); idx.append(i)
            except Exception:
                continue

        if not pvals:
            return

        _, qvals, _, _ = multipletests(pvals, method="fdr_bh")

        # Use the *current* limits for connector position + text padding
        x_min, x_max = ax.get_xlim()
        x_span = x_max - x_min
        xr = x_max - 0.35                 # vertical bar x
        dx = 0.04 * x_span                # label padding from the bar
        x_text = min(x_max - 0.02 * x_span, xr + dx)

        for i, q in zip(idx, qvals):
            label = "***" if q < 0.01 else ("**" if q < 0.05 else ("*" if q < 0.10 else "n.s."))
            y1, y2 = pos_real[i], pos_gen[i]
            ax.plot([xr, xr], [y1, y2], color="black", lw=1.0, zorder=3, clip_on=False)
            ax.text(x_text, 0.5*(y1+y2), label, ha="left", va="center",
                    fontsize=9, fontweight="bold", zorder=4, clip_on=False)

    # Collect all clusters from all meals
    all_clusters_info = []

    for meal in meal_types:
        Xr = real_data.get(meal)
        Xg = gen_data.get(meal)
        Cr = real_clusters.get(meal)
        Cg = gen_clusters.get(meal)
        
        if Xr is None or Xg is None or Cr is None or Cg is None:
            continue

        # Get unique clusters for this meal
        uniq = sorted({int(c) for c in np.concatenate([Cr, Cg]) if int(c) != -1})
        if not uniq:
            continue

        # Process each cluster for this meal
        for cid in uniq:
            r_mask = (Cr == cid)
            g_mask = (Cg == cid)

            r_dev = g_dev = None
            if r_mask.any():
                r_dev = per_meal_deviation_from_rdi(
                    Xr[r_mask], meal, W, nutrient_names, RDI, meal_fraction_map,
                    include_nutrients, mode=deviation_mode, clip=clip
                )
            if g_mask.any():
                g_dev = per_meal_deviation_from_rdi(
                    Xg[g_mask], meal, W, nutrient_names, RDI, meal_fraction_map,
                    include_nutrients, mode=deviation_mode, clip=clip
                )

            all_clusters_info.append({
                "cid": int(cid),
                "meal": meal,
                "label": f"{meal} - {get_cluster_name_fn(meal, cid)}",
                "r_dev": r_dev,
                "g_dev": g_dev,
            })

    # --- Order: Breakfast (top), Lunch (middle), Dinner (bottom)
    meal_order = ['Breakfast', 'Lunch', 'Dinner']
    sorted_clusters = []

    for meal in meal_order:
        # clusters for this meal
        meal_clusters = [info for info in all_clusters_info if info['meal'] == meal]

        # sort by smallest -> largest median absolute deviation of REAL meals
        def real_dev_magnitude(info):
            r = info['r_dev']
            if r is None or len(r) == 0:
                return float('inf')  # push clusters with no real data to the end
            return float(np.nanmedian(np.abs(r)))

        meal_clusters.sort(key=real_dev_magnitude)
        sorted_clusters.extend(meal_clusters)

    all_clusters_info = sorted_clusters

    # Calculate meal boundaries for outside-left brackets
    meal_boundaries = []
    current_meal = None
    start_idx = 0
    
    for i, info in enumerate(all_clusters_info):
        if current_meal is None:
            current_meal = info['meal']
            start_idx = i
        elif info['meal'] != current_meal:
            # End of current meal group
            meal_boundaries.append((current_meal, start_idx, i - 1))
            current_meal = info['meal']
            start_idx = i
    
    # Add the last meal group
    if current_meal is not None:
        meal_boundaries.append((current_meal, start_idx, len(all_clusters_info) - 1))

    # --- ensure enough vertical pixels per row ---
    rows = len(all_clusters_info)

    dpi = fig.get_dpi()
    h_in, w_in = fig.get_size_inches()[1], fig.get_size_inches()[0]

    # target: ~18 px per row + ~120 px for margins
    min_px_per_row = 18
    margin_px = 120
    need_h_in = max(h_in, (rows * min_px_per_row + margin_px) / dpi)

    if need_h_in > h_in:
        fig.set_size_inches(w_in, need_h_in, forward=True)

    # Y positions for each cluster row (Breakfast at top, Dinner at bottom, best clusters at top within each meal)
    base = np.arange(len(all_clusters_info), dtype=float)[::-1]  # Reverse so Breakfast appears at top
    pos_real = base - dy/2
    pos_gen = base + dy/2

    stats_list = []
    positions = []
    colors_seq = []
    yticklabels = []
    # rows for CSV export of boxplot statistics
    csv_rows = []

    for i, info in enumerate(all_clusters_info):
        r_dev = info["r_dev"]
        g_dev = info["g_dev"]
        meal = info["meal"]
        colors = meal_colors.get(meal, {'real': '#1f77b4', 'gen': '#ff7f0e'})

        # Real data
        if r_dev is not None and len(r_dev) > 0:
            st = _bxp_stats_from_samples(r_dev, "Real")
            if st:
                stats_list.append(st)
                positions.append(pos_real[i])
                colors_seq.append(colors["real"])
                # Collect detailed stats for CSV
                arr = np.asarray(r_dev, float)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    p10, q1, med, q3, p90 = np.percentile(arr, [10, 25, 50, 75, 90])
                    csv_rows.append({
                        "meal": meal,
                        "cluster_id": int(info["cid"]),
                        "cluster_name": get_cluster_name_fn(meal, int(info["cid"])),
                        "type": "Real",
                        "n": int(arr.size),
                        "p10": float(p10),
                        "q1": float(q1),
                        "median": float(med),
                        "q3": float(q3),
                        "p90": float(p90),
                        "mean": float(np.nanmean(arr)),
                        "std": float(np.nanstd(arr))
                    })

        # Generated data
        if g_dev is not None and len(g_dev) > 0:
            st = _bxp_stats_from_samples(g_dev, "Generated")
            if st:
                stats_list.append(st)
                positions.append(pos_gen[i])
                colors_seq.append(colors["gen"])
                # Collect detailed stats for CSV
                arr = np.asarray(g_dev, float)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    p10, q1, med, q3, p90 = np.percentile(arr, [10, 25, 50, 75, 90])
                    csv_rows.append({
                        "meal": meal,
                        "cluster_id": int(info["cid"]),
                        "cluster_name": get_cluster_name_fn(meal, int(info["cid"])),
                        "type": "Generated",
                        "n": int(arr.size),
                        "p10": float(p10),
                        "q1": float(q1),
                        "median": float(med),
                        "q3": float(q3),
                        "p90": float(p90),
                        "mean": float(np.nanmean(arr)),
                        "std": float(np.nanstd(arr))
                    })

        yticklabels.append(info["label"])

    # Draw the boxplot
    if stats_list:
        b = ax.bxp(stats_list, positions=positions, widths=dy, showfliers=False,
                   patch_artist=True, vert=False)  # HORIZONTAL
        for patch, col in zip(b['boxes'], colors_seq):
            patch.set(facecolor=col, alpha=0.55, edgecolor="black", linewidth=0.9)
        for coll in ("whiskers", "caps", "medians"):
            for artist in b[coll]:
                artist.set(color="black", linewidth=0.9)

        # Build rows for significance in the exact plotted order
        rows_for_sig = [(info["r_dev"], info["g_dev"]) for info in all_clusters_info]

    
    # Write CSV with boxplot statistics, aligned with figure path
    try:
        if out_path is not None and csv_rows:
            import os
            import pandas as pd
            out_dir_base, fname = os.path.split(out_path)
            stem, _ = os.path.splitext(fname)
            csv_path = os.path.join(out_dir_base, f"{stem}_stats.csv")
            pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
            print(f"✓ Boxplot stats table saved to {csv_path}")
    except Exception as e:
        print(f"[warn] Failed to write boxplot stats CSV: {e}")

    # Y-axis = clusters (remove meal type from labels, keep only cluster names)
    from textwrap import wrap as _wrap

    def _wrap_tick(s, width=24):
        return "\n".join(_wrap(s, width=width, break_long_words=False, break_on_hyphens=False))

    clean_labels = [label.split(' - ')[1] if ' - ' in label else label for label in yticklabels]
    wrapped_labels = [_wrap_tick(lbl, width=24) for lbl in clean_labels]

    ax.set_yticks(base)
    ax.set_yticklabels(wrapped_labels, rotation=0)
    ax.tick_params(axis='y', labelsize=14)  # larger font
    for t in ax.get_yticklabels():
        t.set_linespacing(0.95)            # tighter line spacing for wrapped labels

    # --- Draw outside-left meal brackets + labels ---
    from matplotlib.transforms import blended_transform_factory as _btf

    # nudge y-tick labels closer to the plot
    ax.tick_params(axis="y", pad=0)

    # make sure text extents are computed
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # widest y-tick label (in pixels)
    yticks = ax.get_yticklabels()
    max_w_px = max((t.get_window_extent(renderer=renderer).width for t in yticks), default=0.0)

    # convert pixel gaps to Axes-fraction (negative x = outside-left)
    ax_w_px = ax.bbox.width

    # push bracket farther left, but reduce gap to the meal text
    gap_from_ticks_px  = 24   # space between tick labels and bracket
    gap_to_mealtext_px = 18   # space between bracket and rotated meal text

    x_bracket = - (max_w_px + gap_from_ticks_px) / ax_w_px - 0.25
    x_label   = - (max_w_px + gap_from_ticks_px + gap_to_mealtext_px) / ax_w_px - 0.25

    print(x_label)

    # ensure there is enough left margin so nothing gets clipped
    fig.subplots_adjust(left=0.54)

    # blended transform: x in Axes fraction, y in data coords
    trans = _btf(ax.transAxes, ax.transData)

    label_fs = 12
    for meal, start_idx, end_idx in meal_boundaries:
        y_top    = base[start_idx] + dy/2
        y_bottom = base[end_idx]   - dy/2
        y_mid    = 0.5*(y_top + y_bottom)

        # bracket
        ax.plot([x_bracket, x_bracket], [y_bottom, y_top],
                transform=trans, color='black', linewidth=2, alpha=0.8,
                clip_on=False, zorder=4)

        # meal label
        c = meal_colors.get(meal, {'real': 'black'})['real']
        ax.text(x_label, y_mid, meal, transform=trans, rotation=90,
                ha='right', va='center', fontsize=label_fs, fontweight='bold',
                color=c, clip_on=False, zorder=5)

    # X label (shared) + baseline line for signed/log2
    xlabel = {
        "abs":   "Average nutrient deviation from target (|coverage − 1|)",
        "signed":"Average nutrient deviation from target (coverage − 1)",
        "log2":  "Average nutrient deviation from target (log₂ coverage)"
    }[deviation_mode]

    # X label + baseline line for signed/log2
    if deviation_mode in ("signed", "log2"):
        ax.axvline(0.0, color="0.6", lw=1, ls="--", alpha=0.8)
        # Add label for RDI line at the top of the plot
        ax.text(0.0, ax.get_ylim()[1], "USDA Recommended\nDaily Intake", 
                ha="center", va="bottom", fontsize=12, color="0.6")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.tick_params(axis='x', labelsize=14)

    # Set x-limits with minimum at -2.5, maximum at 1.3 (to accommodate meal type labels and significance markers)
    ax.set_xlim(-2.5, 1.3)

    # Add significance markers after x-limits are set
    _add_significance_markers_h(ax, rows_for_sig, pos_real, pos_gen)

    # Legend panel on the right
    try:
        from matplotlib.patches import Patch
        legend_ax.axis('off')

        # Source legend: Real / Generated in grayscale
        src_handles = [
            Patch(facecolor="#555555", edgecolor="black", alpha=0.85, label="Real"),
            Patch(facecolor="#BBBBBB", edgecolor="black", alpha=0.85, label="Generated"),
        ]
        legend_ax.legend(
            handles=src_handles,
            title="Source",
            loc="upper left",
            bbox_to_anchor=(0.10, 0.72, 0.80, 0.28),
            bbox_transform=legend_ax.transAxes,
            fontsize=12, frameon=True, borderpad=0.4, labelspacing=0.4,
            handletextpad=0.6, borderaxespad=0.0, ncol=1, title_fontsize=14,
        )

        # Stats legend (same font size, placed underneath)
        stats_text = (
            "Significance (FDR):\n"
            "  ***  q < 0.01\n"
            "  **   q < 0.05\n"
            "  *    q < 0.10\n"
            "  n.s. q ≥ 0.10"
        )
        legend_ax.text(
            0.10, 0.46, stats_text, transform=legend_ax.transAxes,
            ha="left", va="top", fontsize=12
        )
    except Exception:
        pass

    try:
        ax.margins(x=0.04)
        fig.tight_layout(rect=(0.01, 0.03, 0.82, 0.985))
    except Exception:
        fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        # Also save as SVG
        svg_path = out_path.replace('.png', '.svg')
        fig.savefig(svg_path, format='svg', bbox_inches="tight")
        print(f"✓ Boxplot also saved as SVG: {svg_path}")
        plt.close(fig)
    else:
        return fig


def _diagnostic_effect_sizes(real_data, gen_data, W, nutrient_names, RDI, meal_fraction_map):
    """Print effect sizes (Cohen's d) for diagnostic purposes"""
    from src.analysis_rdi_eval.nutrition_metrics import grams_to_nutrients, per_meal_macro_percentages, kcal_from_nutrients
    from tabulate import tabulate
    
    print("Spider Plot Diagnostic - Effect Sizes (Cohen's d):")
    print("=" * 60)
    
    # Test vitamins (using recommended keys)
    vitamin_keys = ["vitamin_d_iu", "vitamin_a_mcg", "vitamin_c_mg", 
                    "vitamin_b12_mcg", "folate_mcg", "vitamin_k_mcg"]
    print("Vitamins (% of meal target coverage):")
    from src.analysis_rdi_eval.nutrition_metrics import per_meal_coverage
    
    # Collect data for tabulation
    vitamin_data = []
    
    for meal in ['Breakfast', 'Lunch', 'Dinner']:
        meal_fraction = meal_fraction_map[meal]
        real_nmat = grams_to_nutrients(real_data[meal], W)
        gen_nmat  = grams_to_nutrients(gen_data[meal], W)
        
        # Debug: Number of meals
        n_real = len(real_data[meal])
        n_gen = len(gen_data[meal])
        
        for key in vitamin_keys:
            r = per_meal_coverage(real_nmat, nutrient_names, RDI, meal_fraction, key)
            g = per_meal_coverage(gen_nmat,  nutrient_names, RDI, meal_fraction, key)
            if r.size and g.size:
                pooled_std = np.sqrt(((len(r)-1)*np.var(r) + (len(g)-1)*np.var(g)) / max(len(r)+len(g)-2, 1))
                if pooled_std > 0:
                    d = (np.mean(g) - np.mean(r)) / pooled_std
                    vitamin_data.append([
                        meal,
                        key,
                        f"{np.mean(r):.3f}±{np.std(r):.3f}",
                        f"{np.mean(g):.3f}±{np.std(g):.3f}",
                        f"{d:.3f}",
                        f"{n_real}/{n_gen}"
                    ])
    
    # Print tabulated vitamin data
    print(tabulate(vitamin_data, 
                   headers=["Meal", "Vitamin", "Real (mean±std)", "Generated (mean±std)", "Cohen's d", "Meals (R/G)"],
                   tablefmt="grid"))
    
    # Test macros
    print("\nMacronutrients (coverage vs AMDR midpoint targets):")
    targets = {'Protein_%': 22.5, 'Fat_%': 27.5, 'Carb_%': 55.0}
    
    # Collect data for tabulation
    macro_data = []
    
    for meal in ['Breakfast', 'Lunch', 'Dinner']:
        real_nmat = grams_to_nutrients(real_data[meal], W)
        gen_nmat  = grams_to_nutrients(gen_data[meal], W)
        real_pct  = per_meal_macro_percentages(real_nmat, nutrient_names, None)
        gen_pct   = per_meal_macro_percentages(gen_nmat,  nutrient_names, None)
        
        # Debug: Number of meals
        n_real = len(real_data[meal])
        n_gen = len(gen_data[meal])
        
        for i, name in enumerate(['Protein_%','Fat_%','Carb_%']):
            r_cov = real_pct[:, i] / targets[name]
            g_cov = gen_pct[:, i]  / targets[name]
            pooled_std = np.sqrt(((len(r_cov)-1)*np.var(r_cov) + (len(g_cov)-1)*np.var(g_cov)) / max(len(r_cov)+len(g_cov)-2, 1))
            if pooled_std > 0:
                d = (np.mean(g_cov) - np.mean(r_cov)) / pooled_std
                macro_data.append([
                    meal,
                    name,
                    f"{np.mean(r_cov):.3f}±{np.std(r_cov):.3f}",
                    f"{np.mean(g_cov):.3f}±{np.std(g_cov):.3f}",
                    f"{d:.3f}",
                    f"{n_real}/{n_gen}"
                ])
    
    # Print tabulated macro data
    print(tabulate(macro_data, 
                   headers=["Meal", "Macro", "Real (mean±std)", "Generated (mean±std)", "Cohen's d", "Meals (R/G)"],
                   tablefmt="grid"))
    
    # Sanity check: low kcal meals
    print("\nSanity Check - Low kcal meals:")
    
    # Collect data for tabulation
    kcal_data = []
    
    for meal in ['Breakfast', 'Lunch', 'Dinner']:
        real_nmat = grams_to_nutrients(real_data[meal], W)
        gen_nmat = grams_to_nutrients(gen_data[meal], W)
        
        real_kcal = kcal_from_nutrients(real_nmat, nutrient_names)
        gen_kcal = kcal_from_nutrients(gen_nmat, nutrient_names)
        
        real_low = np.sum(real_kcal < 50) / len(real_kcal) * 100
        gen_low = np.sum(gen_kcal < 50) / len(gen_kcal) * 100
        kcal_data.append([
            meal,
            f"{real_low:.1f}%",
            f"{gen_low:.1f}%",
            f"{len(real_data[meal])}/{len(gen_data[meal])}"
        ])
    
    # Print tabulated kcal data
    print(tabulate(kcal_data, 
                   headers=["Meal", "Real < 50 kcal", "Generated < 50 kcal", "Meals (R/G)"],
                   tablefmt="grid"))


def create_spider_plots(real_data: Dict[str, np.ndarray], 
                       gen_data: Dict[str, np.ndarray],
                       W: np.ndarray, 
                       nutrient_names: List[str],
                       RDI: np.ndarray,
                       meal_fraction_map: Dict[str, float],
                       food_to_group: Dict[str, str] = None,
                       feature_names: List[str] = None,
                       output_dir: str = "outputs/runs/eval_run_001/figures",
                       limits: Dict[str, float] = None) -> None:
    """
    Create comprehensive spider plots for nutrition analysis in a single 3x3 figure
    (rows: Breakfast, Lunch, Dinner; columns: nutrition metrics, macronutrients,
    micronutrients). Each vertex is the median per-meal score on a 0-1 scale. Prints
    effect-size tables first, then writes <output_dir>/spider_plots_comprehensive.{png,svg}.

    Args:
        real_data: Dict mapping meal types to real meal data
        gen_data: Dict mapping meal types to generated meal data
        W: Nutrient matrix (D, M)
        nutrient_names: List of nutrient names
        RDI: Reference daily intakes
        meal_fraction_map: Dict mapping meal types to fractions
        food_to_group: Food group per food code (with feature_names: food-group diversity)
        feature_names: Food codes, in the column order of the meal matrices
        output_dir: Output directory for figures
        limits: Moderation limits for MER (defaults: sodium, SFA and sugars)
    """
    meal_types = ['Breakfast', 'Lunch', 'Dinner']
    
    # Run diagnostic first
    _diagnostic_effect_sizes(real_data, gen_data, W, nutrient_names, RDI, meal_fraction_map)
    
    # Create single 3x3 figure with all spider plots
    fig, axes = plt.subplots(3, 3, figsize=(20, 18), subplot_kw=dict(projection='polar'))
    
    plt.tight_layout()
    plt.subplots_adjust(
        left=0.035, right=0.995, top=0.96, bottom=0.06,
        wspace=0.12,   # tight columns
        hspace=0.30    # room between the 3 rows
    )
    
    # ---------- Transposed layout ----------
    # Rows = meals (Breakfast, Lunch, Dinner)
    # Cols = categories (Metrics, Macros, Micronutrients)

    # Col 0: Nutrition Metrics
    _plot_metrics_row(
        axes[:, 0], real_data, gen_data, W, nutrient_names, RDI,
        meal_fraction_map, food_to_group, feature_names, meal_types,
        limits=limits
    )

    # Col 1: Macronutrients
    _plot_macronutrients_row(
        axes[:, 1], real_data, gen_data, W, nutrient_names, RDI,
        meal_fraction_map, meal_types
    )

    # Col 2: Micronutrients
    _plot_micronutrients_row(
        axes[:, 2], real_data, gen_data, W, nutrient_names, RDI,
        meal_fraction_map, meal_types
    )

    # Column labels (categories), centered
    col_labels = ['Nutrition Metrics', 'Macronutrients', 'Micronutrients']
    for j, label in enumerate(col_labels):
        fig.text(0.165 + j * 0.33, 0.97, label, ha='center', va='center',
                 fontsize=18, fontweight='bold')

    plt.savefig(f"{output_dir}/spider_plots_comprehensive.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/spider_plots_comprehensive.svg", format='svg', bbox_inches='tight')
    print(f"✓ Spider plots also saved as SVG: {output_dir}/spider_plots_comprehensive.svg")
    plt.close()
    


def _plot_metrics_row(axes, real_data, gen_data, W, nutrient_names, RDI, 
                      meal_fraction_map, food_to_group, feature_names, meal_types,
                      limits: Dict[str, float] = None):
    from src.analysis_rdi_eval.nutrition_metrics import (
        compute_nutrition_metrics_comprehensive, grams_to_nutrients,
        per_meal_macro_percentages, _canon_map
    )
    labels = ["Energy density", "AMDR composite", "MER", "MAR", "Diversity"]

    for i, meal in enumerate(meal_types):
        ax = axes[i]
        frac = meal_fraction_map[meal]
        cmap = _canon_map(nutrient_names)

        # nutrient matrices
        Xr = grams_to_nutrients(real_data[meal], W)
        Xg = grams_to_nutrients(gen_data[meal],  W)

        # defaults for moderation (if not supplied)
        if limits is None:
            limits_ = {"sodium_mg": 2300.0, "sfa_pct_kcal": 10.0}
            if "added_sugars_g" in cmap:
                limits_["added_sugars_pct_kcal"] = 10.0
            elif "sugars_total_g" in cmap:
                limits_["sugars_total_pct_kcal"] = 25.0
        else:
            limits_ = dict(limits)

        # Calculate target groups from real data for this meal type
        target_groups = 3  # default fallback
        if food_to_group is not None and feature_names is not None:
            from src.analysis_rdi_eval.nutrition_metrics import _group_kcal_shares
            try:
                p_real, _ = _group_kcal_shares(real_data[meal], W, nutrient_names,
                                             feature_names, food_to_group, min_kcal=10.0)
                target_groups = int(_np.median((p_real > 0).sum(axis=1))) or 3
                print(f"Target groups for {meal}: {target_groups}")
            except Exception as e:
                print(f"Warning: Could not calculate target groups for {meal}: {e}")
                target_groups = 3

        r = compute_nutrition_metrics_comprehensive(real_data[meal], W, nutrient_names, RDI, frac,
                                                    food_to_group=food_to_group, feature_names=feature_names,
                                                    limits=limits_, target_groups=target_groups)
        g = compute_nutrition_metrics_comprehensive(gen_data[meal],  W, nutrient_names, RDI, frac,
                                                    food_to_group=food_to_group, feature_names=feature_names,
                                                    limits=limits_, target_groups=target_groups)
        
        # Debug comprehensive metrics results
        if meal == 'Breakfast':
            print(f"\n=== COMPREHENSIVE METRICS DEBUG for {meal} ===")
            print(f"Real diversity from comprehensive: {r['diversity'][:5]}")
            print(f"Gen diversity from comprehensive:  {g['diversity'][:5]}")
            print(f"Real diversity type: {type(r['diversity'])}, shape: {r['diversity'].shape if hasattr(r['diversity'], 'shape') else 'scalar'}")
            print(f"Gen diversity type: {type(g['diversity'])}, shape: {g['diversity'].shape if hasattr(g['diversity'], 'shape') else 'scalar'}")
            print("=" * 50)

        # --- Scores (arrays per meal) on 0–1 ---
        # Energy density (kcal/g): lower is better; reference E* = 1.25 kcal/g.
        def _ed_scores(Xn, grams):
            kcal = Xn[:, cmap["energy_kcal"]]
            ed = kcal / _np.maximum(_np.sum(grams, axis=1), 1e-6)
            return _score_max_target(ed, 1.25), ed  # (scores, raw)

        ed_real_s, _ = _ed_scores(Xr, real_data[meal])
        ed_gen_s,  _ = _ed_scores(Xg, gen_data[meal])

        # AMDR composite from %kcal bands
        pL,pU = 10,35; fL,fU = 20,35; cL,cU = 45,65
        r_pct = per_meal_macro_percentages(Xr, nutrient_names, None)
        g_pct = per_meal_macro_percentages(Xg, nutrient_names, None)
        amdr_r = _np.nanmean([_score_band_pct(r_pct[:,0], pL,pU),
                              _score_band_pct(r_pct[:,1], fL,fU),
                              _score_band_pct(r_pct[:,2], cL,cU)], axis=0)
        amdr_g = _np.nanmean([_score_band_pct(g_pct[:,0], pL,pU),
                              _score_band_pct(g_pct[:,1], fL,fU),
                              _score_band_pct(g_pct[:,2], cL,cU)], axis=0)

        # --- MAR / MER / Diversity as proper 1-D [0,1] score arrays ---
        nR = Xr.shape[0]
        nG = Xg.shape[0]

        # MAR should already be an adequacy score. Make it 1-D and clip to [0,1].
        mar_r = _np.clip(_as_1d_array(r.get("MAR", _np.nan), nR), 0.0, 1.0)
        mar_g = _np.clip(_as_1d_array(g.get("MAR", _np.nan), nG), 0.0, 1.0)

        # MER may be a ratio (>1). Convert to a 0–1 score where 1 = at/below limit.
        mer_r = _mer_to_score01(_as_1d_array(r.get("MER", _np.nan), nR))
        mer_g = _mer_to_score01(_as_1d_array(g.get("MER", _np.nan), nG))

        # Diversity: Use the comprehensive metrics diversity (Group-based Hill diversity) consistently
        print("Using comprehensive metrics diversity (Group-based Hill diversity)...")
        div_r = _np.clip(_as_1d_array(r.get("diversity", _np.nan), nR), 0.0, 1.0)
        div_g = _np.clip(_as_1d_array(g.get("diversity", _np.nan), nG), 0.0, 1.0)

        # Debug diversity calculation for all meals
        print(f"\n=== DIVERSITY ANALYSIS for {meal} ===")
        print(f"Real data shape: {real_data[meal].shape}, Gen data shape: {gen_data[meal].shape}")
        
        # Check raw data characteristics
        real_grams = real_data[meal]
        gen_grams = gen_data[meal]
        print(f"Real grams - sum: {_np.sum(real_grams):.1f}, non-zero items: {_np.count_nonzero(real_grams)}")
        print(f"Gen grams  - sum: {_np.sum(gen_grams):.1f}, non-zero items: {_np.count_nonzero(gen_grams)}")
        
        # Check items >= 5g threshold
        real_above_5g = _np.count_nonzero(real_grams >= 5.0, axis=1)
        gen_above_5g = _np.count_nonzero(gen_grams >= 5.0, axis=1)
        print(f"Real items >= 5g per meal: mean={_np.mean(real_above_5g):.1f}, std={_np.std(real_above_5g):.1f}, min={_np.min(real_above_5g)}, max={_np.max(real_above_5g)}")
        print(f"Gen items >= 5g per meal:  mean={_np.mean(gen_above_5g):.1f}, std={_np.std(gen_above_5g):.1f}, min={_np.min(gen_above_5g)}, max={_np.max(gen_above_5g)}")
        
        # Check for NaN values in raw data
        real_nan_count = _np.sum(_np.isnan(real_grams))
        gen_nan_count = _np.sum(_np.isnan(gen_grams))
        print(f"Real data NaN count: {real_nan_count}, Gen data NaN count: {gen_nan_count}")
        
        # Check comprehensive metrics diversity
        print(f"Comprehensive diversity - Real: mean={_np.nanmean(div_r):.3f}, std={_np.nanstd(div_r):.3f}, min={_np.nanmin(div_r):.3f}, max={_np.nanmax(div_r):.3f}")
        print(f"Comprehensive diversity - Gen:  mean={_np.nanmean(div_g):.3f}, std={_np.nanstd(div_g):.3f}, min={_np.nanmin(div_g):.3f}, max={_np.nanmax(div_g):.3f}")
        print(f"Real diversity NaN count: {_np.sum(_np.isnan(div_r))}, Gen diversity NaN count: {_np.sum(_np.isnan(div_g))}")

        # Handle any remaining NaN values
        div_r = _np.nan_to_num(div_r, nan=0.0)
        div_g = _np.nan_to_num(div_g, nan=0.0)
        
        # Calculate both mean±std and median[IQR] for comparison
        real_mean, real_std = _np.mean(div_r), _np.std(div_r)
        real_med, real_iqr = _np.nanmedian(div_r), _np.nanpercentile(div_r, 75) - _np.nanpercentile(div_r, 25)
        gen_mean, gen_std = _np.mean(div_g), _np.std(div_g)
        gen_med, gen_iqr = _np.nanmedian(div_g), _np.nanpercentile(div_g, 75) - _np.nanpercentile(div_g, 25)
        
        print(f"Final diversity - Real: mean±std={real_mean:.3f}±{real_std:.3f}, median[IQR]={real_med:.3f}[{real_iqr:.3f}], min={_np.min(div_r):.3f}, max={_np.max(div_r):.3f}")
        print(f"Final diversity - Gen:  mean±std={gen_mean:.3f}±{gen_std:.3f}, median[IQR]={gen_med:.3f}[{gen_iqr:.3f}], min={_np.min(div_g):.3f}, max={_np.max(div_g):.3f}")
        print("=" * 60)

        real_series = [ed_real_s, amdr_r, mer_r, mar_r, div_r]
        gen_series  = [ed_gen_s,  amdr_g, mer_g, mar_g, div_g]

        # Debug the series data before plotting
        print(f"\n=== SERIES DEBUG for {meal} ===")
        print(f"Labels: {labels}")
        print(f"Real series lengths: {[len(s) if hasattr(s, '__len__') else 'scalar' for s in real_series]}")
        print(f"Gen series lengths:  {[len(s) if hasattr(s, '__len__') else 'scalar' for s in gen_series]}")
        
        # Show both mean±std and median[IQR] for each series
        for i, label in enumerate(labels):
            s_r = real_series[i]
            s_g = gen_series[i]
            if hasattr(s_r, '__len__') and len(s_r) > 0:
                r_mean, r_std = _np.mean(s_r), _np.std(s_r)
                r_med, r_iqr = _np.nanmedian(s_r), _np.nanpercentile(s_r, 75) - _np.nanpercentile(s_r, 25)
                g_mean, g_std = _np.mean(s_g), _np.std(s_g)
                g_med, g_iqr = _np.nanmedian(s_g), _np.nanpercentile(s_g, 75) - _np.nanpercentile(s_g, 25)
                print(f"{label}: Real mean±std={r_mean:.3f}±{r_std:.3f}, median[IQR]={r_med:.3f}[{r_iqr:.3f}]")
                print(f"{label}: Gen  mean±std={g_mean:.3f}±{g_std:.3f}, median[IQR]={g_med:.3f}[{g_iqr:.3f}]")
            else:
                print(f"{label}: Real={s_r}, Gen={s_g}")
        print("=" * 50)

        # plot
        _plot_spider_data(ax, _np.array([_PLOT_CENTER(s) for s in real_series]),
                             _np.array([_PLOT_CENTER(s) for s in gen_series]),
                             labels, meal)


def _plot_macronutrients_row(axes, real_data, gen_data, W, nutrient_names, RDI, 
                             meal_fraction_map, meal_types):
    from src.analysis_rdi_eval.nutrition_metrics import grams_to_nutrients, per_meal_coverage
    labels = ["Protein", "Carbs", "Fat", "Fiber"]

    # Uses 'protein_g', 'carb_g', 'fat_g', 'fiber_g'; a nutrient missing from nutrient_names plots as 0
    for i, meal in enumerate(meal_types):
        ax = axes[i]; frac = meal_fraction_map[meal]
        Xr = grams_to_nutrients(real_data[meal], W)
        Xg = grams_to_nutrients(gen_data[meal],  W)

        # coverage arrays (value / per-meal target)
        cov_r = [
            per_meal_coverage(Xr, nutrient_names, RDI, frac, 'protein_g'),
            per_meal_coverage(Xr, nutrient_names, RDI, frac, 'carb_g'),
            per_meal_coverage(Xr, nutrient_names, RDI, frac, 'fat_g'),
            per_meal_coverage(Xr, nutrient_names, RDI, frac, 'fiber_g'),
        ]
        cov_g = [
            per_meal_coverage(Xg, nutrient_names, RDI, frac, 'protein_g'),
            per_meal_coverage(Xg, nutrient_names, RDI, frac, 'carb_g'),
            per_meal_coverage(Xg, nutrient_names, RDI, frac, 'fat_g'),
            per_meal_coverage(Xg, nutrient_names, RDI, frac, 'fiber_g'),
        ]

        # cap to 0..1 for consistent display
        real_series = [ _np.clip(c, 0.0, 1.0) for c in cov_r ]
        gen_series  = [ _np.clip(c, 0.0, 1.0) for c in cov_g ]

        _plot_spider_data_diagonal(ax,
            _np.array([_PLOT_CENTER(s) for s in real_series]),
            _np.array([_PLOT_CENTER(s) for s in gen_series]),
            labels, meal, show_legend=False
        )


def _plot_micronutrients_row(axes, real_data, gen_data, W, nutrient_names, RDI, 
                             meal_fraction_map, meal_types):
    from src.analysis_rdi_eval.nutrition_metrics import grams_to_nutrients, per_meal_coverage, _canon_map
    cmap = _canon_map(nutrient_names)
    
    # Combine vitamins and minerals into one row
    vit_d_key = 'vitamin_d_iu' if 'vitamin_d_iu' in cmap else 'vitamin_d_mcg'
    vitamin_keys = [vit_d_key, 'vitamin_a_mcg', 'vitamin_c_mg', 'vitamin_b12_mcg', 'folate_mcg', 'vitamin_k_mcg']
    mineral_keys = ["calcium_mg","iron_mg","zinc_mg","magnesium_mg","potassium_mg"]
    
    # Combine all micronutrient keys and labels
    all_keys = vitamin_keys + mineral_keys
    all_labels = ["Vitamin D","Vitamin A","Vitamin C","Vitamin B12","Folate","Vitamin K",
                  "Calcium","Iron","Zinc","Magnesium","Potassium"]

    for i, meal in enumerate(meal_types):
        ax = axes[i]; frac = meal_fraction_map[meal]
        Xr = grams_to_nutrients(real_data[meal], W)
        Xg = grams_to_nutrients(gen_data[meal],  W)
        real_series = []; gen_series = []

        for key in all_keys:
            r = _np.clip(per_meal_coverage(Xr, nutrient_names, RDI, frac, key), 0.0, 1.0)
            g = _np.clip(per_meal_coverage(Xg, nutrient_names, RDI, frac, key), 0.0, 1.0)
            real_series.append(r); gen_series.append(g)

        _plot_spider_data(ax,
            _np.array([_PLOT_CENTER(s) for s in real_series]),
            _np.array([_PLOT_CENTER(s) for s in gen_series]),
            all_labels, meal
        )


def create_unified_plot_layout(
    meal_types: list[str],
    real_data: dict[str, np.ndarray],
    gen_data: dict[str, np.ndarray],
    real_clusters: dict[str, np.ndarray],
    gen_clusters: dict[str, np.ndarray],
    W: np.ndarray,
    nutrient_names: list[str],
    RDI: np.ndarray,
    meal_fraction_map: dict[str, float],
    food_to_group: dict[str, str] = None,
    feature_names: list[str] = None,
    limits: dict[str, float] = None,
    out_path: str | None = None,
    figsize: tuple = (30, 22),                        # taller to accommodate larger boxplots
):
    """
    Build one aggregated figure: A) overview scatter image (see _create_overview_scatter_plot);
    B) cluster deviation boxplots; C) 3×3 spider-grid. B and C are first written to the
    output folder and then embedded from those PNGs.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.offsetbox import AnchoredText

    # Resolve output dir and the two panel image paths
    out_dir = Path(out_path).parent if out_path else Path(".")
    boxplot_path = out_dir / "boxplots_rdi_deviation_real_vs_generated.png"
    spider_path  = out_dir / "spider_plots_comprehensive.png"

    # --- 1) Generate B and C files (overwrite if present) ---
    # Define cluster name function for human-friendly names
    def _cluster_name(meal, cid: int) -> str:
        try:
            from cli.run_multi_meal_cvae import _get_cluster_name as _human
            return _human(meal, int(cid))
        except Exception:
            return f"Cluster {int(cid)}"
    
    _plot_cluster_deviation_boxplots(
        meal_types=meal_types,
        real_data=real_data,
        gen_data=gen_data,
        real_clusters=real_clusters,
        gen_clusters=gen_clusters,
        W=W,
        nutrient_names=nutrient_names,
        RDI=RDI,
        meal_fraction_map=meal_fraction_map,
        get_cluster_name_fn=_cluster_name,
        figsize=(20, 26),  # Larger to fit full column length
        out_path=str(boxplot_path),
        deviation_mode="log2",
        clip=2.0,
    )

    create_spider_plots(
        real_data=real_data,
        gen_data=gen_data,
        W=W,
        nutrient_names=nutrient_names,
        RDI=RDI,
        meal_fraction_map=meal_fraction_map,
        food_to_group=food_to_group,
        feature_names=feature_names,
        limits=limits,
        output_dir=str(out_dir)
    )

    # --- 2) Build the unified mosaic and embed the images ---
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.2, 1.2],   # equal width for left and right columns
        height_ratios=[1.0, 1.0],  # equal height for top and bottom rows
        wspace=0.06, hspace=0.08
    )
    axA = fig.add_subplot(gs[0, 0])  # Top left: Scatter plot
    axB = fig.add_subplot(gs[:, 1])  # Right column: Boxplot (spans both rows)
    axC = fig.add_subplot(gs[1, 0])  # Bottom left: Spider plots

    # Panel A: overview scatter image (top left)
    # Panel A: the training stage's overview plot, <run>/sanity_samples/analysis/
    _create_overview_scatter_plot(
        axA, out_dir.parent / "sanity_samples" / "analysis" / "overview_real_vs_generated.png"
    )

    # Helper to drop an image on an axes
    def _imshow_panel(ax, img_path: Path, aspect_ratio='auto'):
        ax.set_axis_off()
        if img_path.exists():
            img = plt.imread(str(img_path))
            ax.imshow(img, aspect=aspect_ratio)
        else:
            ax.text(0.5, 0.5, f"Missing: {img_path.name}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)

    # Panel B: boxplot image (right column, spans both rows)
    _imshow_panel(axB, boxplot_path, aspect_ratio='equal')

    # Panel C: spider-grid image (bottom left)
    _imshow_panel(axC, spider_path, aspect_ratio='equal')

    # Anchored panel letters - aligned horizontally for A,B and vertically for A,C
    # A and B horizontally aligned
    atA = AnchoredText("A", prop=dict(size=16, weight="bold"),
                      frameon=False, loc="upper left")
    axA.add_artist(atA)
    
    atB = AnchoredText("B", prop=dict(size=16, weight="bold"),
                      frameon=False, loc="upper left")
    axB.add_artist(atB)
    
    # A and C vertically aligned
    atC = AnchoredText("C", prop=dict(size=16, weight="bold"),
                      frameon=False, loc="upper left")
    axC.add_artist(atC)


    # Tighten layout
    try:
        fig.tight_layout(rect=(0.01, 0.03, 0.995, 0.99))
    except Exception:
        pass

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig


def _create_overview_scatter_plot(ax, image_path: Path):
    """Show the overview plot written by `make cvae_train` (image_path) in `ax`, or a note if
    it is missing."""
    import matplotlib.image as mpimg

    if image_path.exists():
        try:
            img = mpimg.imread(str(image_path))
            ax.imshow(img, aspect='equal')  # Maintain original aspect ratio
            ax.axis('off')  # Hide axes since we're showing an image
        except Exception as e:
            print(f"Warning: Could not load overview image: {e}")
            ax.text(0.5, 0.5, 'Overview plot not available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')
    else:
        print(f"Warning: Overview image not found at {image_path}")
        ax.text(0.5, 0.5, 'Overview plot not found', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.axis('off')


def _plot_spider_data(ax, real_values, gen_values, labels, meal, show_legend=False):
    """Plot spider plot data for real vs generated comparison"""
    # Debug spider plot data
    print(f"\nSPIDER DEBUG for {meal}:")
    print(f"  Labels: {labels}")
    print(f"  Real values: {real_values}")
    print(f"  Gen values:  {gen_values}")
    print(f"  Real values shape: {real_values.shape}, Gen values shape: {gen_values.shape}")
    
    # Set up angles
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Complete the data circles
    real_values = np.concatenate((real_values, [real_values[0]]))
    gen_values = np.concatenate((gen_values, [gen_values[0]]))

    # Tighten polygons inward so bigger labels never touch the web
    _R_SCALE = 0.86   # try 0.86–0.88 if needed
    real_values = real_values * _R_SCALE
    gen_values = gen_values * _R_SCALE
    
    # Define meal-specific color palette
    meal_colors = {
        'Breakfast': {'real': '#1f4e79', 'gen': '#5b9bd5'},      # Dark blue / Light blue
        'Lunch': {'real': '#2d5016', 'gen': '#70ad47'},          # Dark green / Light green  
        'Dinner': {'real': '#d2691e', 'gen': '#ff8c00'}          # Dark orange / Bright orange
    }
    
    colors = meal_colors.get(meal, {'real': '#1f77b4', 'gen': '#ff7f0e'})  # fallback colors
    
    # Plot
    ax.plot(angles, real_values, 'o-', linewidth=2, label='Real', color=colors['real'], alpha=0.8)
    ax.fill(angles, real_values, alpha=0.25, color=colors['real'])
    
    ax.plot(angles, gen_values, 'o-', linewidth=2, label='Generated', color=colors['gen'], alpha=0.8)
    ax.fill(angles, gen_values, alpha=0.25, color=colors['gen'])
    
    # Ensure fills sit below lines so boundaries stay crisp
    for coll in ax.collections:
        coll.set_zorder(2)
    for line in ax.lines:
        line.set_zorder(3)
    
    # Professional polar styling that does NOT expand the subplot box
    _beautify_polar_compact(ax, labels, wrap=11, tick_labelsize=22, r_ticklabelsize=18, pad=15)
    
    # Add legend only if requested
    if show_legend:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)

def _plot_spider_data_diagonal(ax, real_values, gen_values, labels, meal, show_legend=False):
    """Plot spider plot data for real vs generated comparison with diagonal axis orientation"""
    # Debug spider plot data
    print(f"\nSPIDER DEBUG for {meal}:")
    print(f"  Labels: {labels}")
    print(f"  Real values: {real_values}")
    print(f"  Gen values:  {gen_values}")
    print(f"  Real values shape: {real_values.shape}, Gen values shape: {gen_values.shape}")
    
    # Set up angles
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Complete the data circles
    real_values = np.concatenate((real_values, [real_values[0]]))
    gen_values = np.concatenate((gen_values, [gen_values[0]]))

    # Tighten polygons inward so bigger labels never touch the web
    _R_SCALE = 0.86   # try 0.86–0.88 if needed
    real_values = real_values * _R_SCALE
    gen_values = gen_values * _R_SCALE
    
    # Define meal-specific color palette
    meal_colors = {
        'Breakfast': {'real': '#1f4e79', 'gen': '#5b9bd5'},      # Dark blue / Light blue
        'Lunch': {'real': '#2d5016', 'gen': '#70ad47'},          # Dark green / Light green  
        'Dinner': {'real': '#d2691e', 'gen': '#ff8c00'}          # Dark orange / Bright orange
    }
    
    colors = meal_colors.get(meal, {'real': '#1f77b4', 'gen': '#ff7f0e'})  # fallback colors
    
    # Plot
    ax.plot(angles, real_values, 'o-', linewidth=2, label='Real', color=colors['real'], alpha=0.8)
    ax.fill(angles, real_values, alpha=0.25, color=colors['real'])
    
    ax.plot(angles, gen_values, 'o-', linewidth=2, label='Generated', color=colors['gen'], alpha=0.8)
    ax.fill(angles, gen_values, alpha=0.25, color=colors['gen'])
    
    # Ensure fills sit below lines so boundaries stay crisp
    for coll in ax.collections:
        coll.set_zorder(2)
    for line in ax.lines:
        line.set_zorder(3)
    
    # Professional polar styling with diagonal orientation
    _beautify_polar_diagonal(ax, labels, wrap=11, tick_labelsize=22, r_ticklabelsize=18, pad=15)
    
    # Add legend only if requested
    if show_legend:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)


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
    Duplicate of `per_meal_deviation_from_rdi` in cli/evaluate.py; keep the two in sync.
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
