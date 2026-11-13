"""Figure 3A: how close each archetype's generated meals are to its real meals.

One point per (meal type, archetype):

- x: Jensen-Shannon divergence between the food-presence distributions of generated and real
  meals, on an inverted axis labelled "Similarity to Real Meals" (0 to 1 across the plotted
  range, so 1 marks the archetype closest to its real meals);
- y: the share of generated meals with two or more foods (normalised Shannon diversity with
  every food counted as 100 g);
- marker: meal type; size: number of real meals; edge width: coverage@N.

`make cvae_train` writes the per-archetype table (`sanity_samples/analysis/
overview_real_vs_generated.csv`) and a version of the plot coloured by meal type.
`make eval` draws Figure 3A (`figures/overview_real_vs_generated_rdi.{png,svg}`), colouring each
archetype by its change in RDI deviation from the Figure 3C statistics:
(|median generated| - |median real|) / |median real|, so -0.47 means generated meals deviate 47%
less from the USDA targets than the real meals of that archetype (the per-archetype values behind
the headline median reduction).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Colours of generated meals in Figures 3B and 3C
MEAL_COLOURS = {"Breakfast": "#5b9bd5", "Lunch": "#70ad47", "Dinner": "#ff8c00"}
MARKERS = {"Breakfast": "o", "Lunch": "s", "Dinner": "^"}
RDI_COLOUR_LABEL = "Change in RDI deviation vs real meals (lower = better)"


def plot_cluster_overview(sumdf: pd.DataFrame, out_png, out_svg, *, colour_col: str | None = None,
                          colour_label: str = RDI_COLOUR_LABEL,
                          title: str = "Meal Cluster Quality: Fidelity, Diversity, and RDI Compliance") -> bool:
    """Draw the overview scatter from a per-archetype table.

    `sumdf` needs js_divergence, food_diversity, n_real_meals, coverage_at_N, meal_name, meal and
    cluster_name. With `colour_col`, markers are coloured by that column on a -1..1 scale
    (red-yellow-green, lower = better, missing values grey) with a colour bar; without it, by
    meal type. Returns False when there is nothing to plot.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D

    sumdf = sumdf[np.isfinite(sumdf["js_divergence"]) & np.isfinite(sumdf["food_diversity"])]
    if not len(sumdf):
        print("[overview] No finite rows to plot in summary.")
        return False

    x = sumdf["js_divergence"].to_numpy(float)
    y = sumdf["food_diversity"].to_numpy(float)
    meals = sumdf["meal_name"].fillna(sumdf["meal"].astype(str))

    cnorm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    cmap = mpl.colormaps["RdYlGn_r"].with_extremes(bad="0.7")   # green = good, red = poor
    colour = sumdf[colour_col].to_numpy(float) if colour_col else None

    # edge thickness = coverage@N (map ~[0,1.2] -> [0.4, 2.6])
    cov = np.clip(sumdf["coverage_at_N"].to_numpy(float), 0.0, 1.2)
    lw = 0.4 + 2.2 * (cov / 1.2)

    # size = number of real meals
    n_real = sumdf["n_real_meals"].to_numpy(float)
    n_real = np.clip(n_real, 1.0, np.nanmax(n_real) if np.isfinite(np.nanmax(n_real)) else 1.0)
    sizes = np.clip(30.0 + 170.0 * (n_real / np.nanmax(n_real)), 30.0, 220.0)

    def _corner_box(ax, text, xy, ha, va, fc, ec="0.3", alpha=0.12):
        ax.text(*xy, text, transform=ax.transAxes, ha=ha, va=va, fontsize=12, weight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec=ec, alpha=alpha))

    fig, ax = plt.subplots(figsize=(10, 7.5))
    last_scatter = None
    for meal_name, mk in MARKERS.items():
        msk = (meals == meal_name).to_numpy()
        if not np.any(msk):
            continue
        if colour is not None:
            last_scatter = ax.scatter(x[msk], y[msk], c=colour[msk], s=sizes[msk], marker=mk,
                                      cmap=cmap, norm=cnorm, edgecolors="k", linewidths=lw[msk],
                                      alpha=0.95, label=meal_name)
        else:
            ax.scatter(x[msk], y[msk], color=MEAL_COLOURS[meal_name], s=sizes[msk], marker=mk,
                       edgecolors="k", linewidths=lw[msk], alpha=0.95, label=meal_name)
    if last_scatter is not None:
        cb = plt.colorbar(last_scatter, ax=ax, shrink=0.7, aspect=20, anchor=(0.0, 1.0))
        cb.set_label(colour_label)
        cb.ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Limits: zoom to the point cloud, with room for labels and the corner boxes
    xq_lo, xq_hi = np.nanquantile(x, [0.02, 0.98])
    yq_lo, yq_hi = np.nanquantile(y, [0.02, 0.98])
    x_pad = 0.20 * (xq_hi - xq_lo + 1e-9)
    y_pad = 0.10 * (yq_hi - yq_lo + 1e-9)
    x_min, x_max = max(0.0, xq_lo - x_pad), min(50.0, xq_hi + x_pad)
    y_min, y_max = max(0.0, yq_lo - y_pad * 2.0), 1.025
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Inverted x axis (divergence 0 = most real-like on the right), labelled as similarity 0..1
    ax.invert_xaxis()
    try:
        tick_pos = np.linspace(x_min, x_max, 6)
        denom = max(1e-9, (x_max - x_min))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([f"{(x_max - t) / denom:.1f}" for t in tick_pos])
    except Exception:
        pass
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))

    _corner_box(ax, "High Diversity\nSynthetic", (0.02, 0.98), "left", "top", fc="#58a55a")
    _corner_box(ax, "High Diversity\nReal-like", (0.98, 0.98), "right", "top", fc="#f3cc5b")
    _corner_box(ax, "Low Diversity\nSynthetic", (0.02, 0.02), "left", "bottom", fc="#8ecae6")
    _corner_box(ax, "Low Diversity\nReal-like", (0.98, 0.02), "right", "bottom", fc="#f28b82")

    # Legends: meal types and sample sizes
    marker_fc = (lambda m: "grey") if colour is not None else (lambda m: MEAL_COLOURS[m])
    meal_handles = [Line2D([0], [0], marker=mk, color="w", label=m, markerfacecolor=marker_fc(m),
                           markeredgecolor="k", markersize=8) for m, mk in MARKERS.items()]
    meal_title = Line2D([0], [0], marker="", color="w", label="Meal Types")
    max_n = float(np.nanmax(n_real)) if np.isfinite(np.nanmax(n_real)) else 1.0
    ticks = np.array([250, 1000, 3000], dtype=float)
    size_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markeredgecolor="k",
                           markersize=np.sqrt(30.0 + 170.0 * (n / max_n)) / 2) for n in ticks]
    size_title = Line2D([0], [0], marker="", color="w", label="Sample Sizes")
    all_h = [meal_title] + meal_handles + [size_title] + size_handles
    all_l = ["Meal Types"] + [h.get_label() for h in meal_handles] + ["Sample Sizes"] + \
            [f"{int(n):,} meals" for n in ticks]
    leg = ax.legend(all_h, all_l, loc="lower left", bbox_to_anchor=(1.02, 0.0), frameon=True,
                    fancybox=False, shadow=False, ncol=1, borderpad=0.3, columnspacing=1.0,
                    handletextpad=0.5, prop={"size": 9})
    t = leg.get_texts()
    if len(t) >= 1:
        t[0].set_weight("bold")
    if len(t) > len(meal_handles) + 1:
        t[len(meal_handles) + 1].set_weight("bold")

    ax.set_xlabel("Similarity to Real Meals", fontsize=14)
    ax.set_ylabel("Meal Diversity (higher = more foods per meal)", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Archetype labels with short connectors (needs adjustText)
    try:
        from adjustText import adjust_text
        rng = np.random.default_rng(7)
        texts, anchors = [], []
        for xi, yi, lbl in zip(x, y, sumdf["cluster_name"].astype(str).tolist()):
            dx = (rng.random() - 0.5) * 0.02
            dy = (rng.random() - 0.5) * 0.02
            tt = ax.text(xi + dx, yi + dy, lbl, fontsize=10, ha="left", va="bottom",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.9),
                         zorder=3, clip_on=True)
            texts.append(tt)
            anchors.append((xi, yi))
        # A fixed number of iterations: without one, adjustText stops after 1 s, so the label
        # layout would depend on CPU speed and load. 300 is about what 1 s gives on a 3.9 GHz core.
        adjust_text(texts, ax=ax, autoalign="xy", expand_points=(2.5, 2.5), expand_text=(1.5, 1.5),
                    force_text=(1.5, 1.5), only_move={"points": "", "text": "xy"},
                    lim=(ax.get_xlim(), ax.get_ylim()), precision=0.001, iter_lim=300)
        to_disp = ax.transData.transform
        to_data = ax.transData.inverted().transform
        for tt, (xi, yi) in zip(texts, anchors):
            tx, ty = tt.get_position()
            pa, pt = to_disp((xi, yi)), to_disp((tx, ty))
            v = pt - pa
            d = float(np.hypot(*v))
            max_px = 80.0
            if d > max_px:
                v *= (max_px / max(d, 1e-6))
                tx, ty = to_data(pa + v)
                tt.set_position((tx, ty))
            ax.plot([xi, tx], [yi, ty], "-", lw=0.8, color="0.5", alpha=0.8, zorder=2, clip_on=True)
    except Exception as e:
        print(f"[overview] label placement skipped: {e}")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"🖼  Overview → {out_png}")
    return True


def rdi_change_by_archetype(stats_csv) -> pd.DataFrame:
    """Per (meal type, archetype): (|median generated| - |median real|) / |median real| from the
    Figure 3C statistics (boxplots_rdi_deviation_real_vs_generated_stats.csv)."""
    st = pd.read_csv(stats_csv)
    piv = st.pivot_table(index=["meal", "cluster_id"], columns="type", values="median", aggfunc="first")
    real = piv["Real"].abs().replace(0.0, np.nan)
    change = (piv["Generated"].abs() - real) / real
    return change.rename("rdi_change").reset_index().rename(columns={"meal": "meal_name"})


def draw_figure3a(run_dir, figures_dir=None):
    """Draw Figure 3A for a run: the training-stage overview table coloured by each archetype's
    change in RDI deviation. Writes figures/overview_real_vs_generated_rdi.{png,svg,csv}; returns
    the SVG path, or None when an input is missing."""
    run_dir = Path(run_dir)
    figures_dir = Path(figures_dir) if figures_dir else run_dir / "figures"
    table = run_dir / "sanity_samples" / "analysis" / "overview_real_vs_generated.csv"
    stats = figures_dir / "boxplots_rdi_deviation_real_vs_generated_stats.csv"
    missing = [str(p) for p in (table, stats) if not p.exists()]
    if missing:
        print(f"[figure 3A] skipped, missing: {', '.join(missing)}")
        return None
    sumdf = pd.read_csv(table).merge(rdi_change_by_archetype(stats), on=["meal_name", "cluster_id"], how="left")
    out = figures_dir / "overview_real_vs_generated_rdi"
    sumdf.to_csv(out.with_suffix(".csv"), index=False)
    if not plot_cluster_overview(sumdf, out.with_suffix(".png"), out.with_suffix(".svg"), colour_col="rdi_change"):
        return None
    return out.with_suffix(".svg")
