"""
MealClusterViz: cluster figures drawn from a fitted MealClusterEngine (read-only use of its results).

Used by cli/run_clustering.py (`make clustering`), writing to --viz-out (outputs/clustering):
- compose_cluster_landscape: <meal>_cluster_landscape.png/.svg (t-SNE of the cluster features,
  archetypes in colour and the other clusters and noise in grey, plus a food-category donut per
  archetype) and <meal>_dot_matrix.png/.svg;
- create_combined_tsne_plot: combined_tsne_landscape.png/.svg, Figure 2 of the paper.
t-SNE embeddings are cached in <engine output_dir>/_tsne_cache/. The archetypes come from
config/cluster_filter_default.json and their names from config/cluster_names.json.
"""
import hashlib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath   # for geometric paths (convex hulls etc.)
from sklearn.manifold import TSNE
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.patches import Wedge
from typing import Optional, Set, Dict

from dataclasses import dataclass


class MealClusterViz:
    """
    Visualization layer that consumes MealClusterEngine outputs (read-only).
    """
    @dataclass
    class LayoutCfg:
        # grid proportions
        height_top: float = 6.0
        height_bottom: float = 2.7
        width_left: float = 2.30
        width_main: float = 4.70
        width_right: float = 3.70
        tsne_to_pies_ratio: float = 3.0

        # inset stack inside left column (insets are off: show_insets=False)
        inset_inner_left: float = 0.03
        inset_inner_width: float = 0.98   # width of each inset frame inside the column
        inset_vpad: float = 0.014        # gap between insets
        inset_box_aspect: float = 0.88   # height/width of each inset; 1.0 is square
        inset_min_cell_h: float = 0.26   # minimum height (fraction of the left column) per inset

        # donut grid; _render_cluster_arcs_panel resets arcs_pad_y, arcs_radius,
        # arcs_label_font and arcs_ncols, and always draws 6 rows (arcs_even_rows)
        arcs_ncols: int = 2          # columns in the donut grid
        arcs_nrows: int = 6
        arcs_pad_blanks: bool = True
        arcs_topk: int = 5
        arcs_inner_frac: float = 0.55
        arcs_label_font: int = 7
        arcs_max_clusters: int = 999
        
        wspace: float = 0.28
        hspace: float = 0.15
        show_insets: bool = False
        max_insets: int = 6

        arcs_pad_x: float = 0.01      # horizontal gap between donuts
        arcs_pad_y: float = 0.037     # vertical gap between donuts
        arcs_radius: float = 0.60    # outer donut radius in the cell
        arcs_label_pad: float = 0.14  # label offset below donut (in axes fraction)

    @dataclass
    class FilterCfg:
        allowed_clusters: Dict[str, Optional[Set[int]]]
        include_noise: bool = False
        noise_label: int = -1
        background_gray_rgba: tuple = (0.7, 0.7, 0.7, 0.15)  # for non-selected + noise (slightly darker and more visible)
        # Fraction of noise points to render in background (1.0 = all noise, 0.1 = 10%)
        noise_fraction: float = 1.0
        # Seed for deterministic subsampling of noise points
        noise_seed: int = 42
        # Fraction of all gray background points to render (applies after noise subsampling)
        gray_fraction: float = 1.0
        # Seed for deterministic subsampling of gray background points
        gray_seed: int = 43

    def __init__(self, engine, use_tsne_cache: bool = True):
        self.E = engine
        self.output_dir = os.path.normpath(
            getattr(engine, "output_dir", "outputs/clustering")
        )
        self.use_tsne_cache = use_tsne_cache
        self.layout = self.LayoutCfg()

        # Share of the daily targets per meal: USDA school-meal precedent
        # (≈1/4 breakfast, 1/3 lunch), remainder (≈0.4167) -> dinner
        self.meal_fraction_by_type = {"breakfast": 0.25, "lunch": 1/3, "dinner": 1 - (0.25 + 1/3)}
        
        # Initialize food category RDI targets (daily amounts in grams, converted from USDA guidelines)
        # Based on USDA MyPlate 2024 recommendations for 2000-calorie diet
        self._food_category_rdi_daily = {
            "fruits": 400,        # 2 cups × 200g per cup (average fruit density)
            "vegetables": 500,    # 2.5 cups × 200g per cup (average vegetable density)  
            "grains": 170,        # 6 oz × 28.35g per oz (grain density)
            "protein_foods": 156, # 5.5 oz × 28.35g per oz (protein density)
            "milk_dairy": 720,    # 3 cups × 240g per cup (milk density)
            "fats_oils": 27       # 27g per day (direct from USDA)
        }

        self.filter = self.FilterCfg(
            allowed_clusters={
                "breakfast":    {0, 2, 3, 4, 6, 7, 9, 10, 11, 12, 23, 27},
                "lunch":        {2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14},
                "dinner":       {3, 4, 5, 6, 7, 9, 10, 11, 13, 14}
            },
            include_noise=False,
            noise_label=-1,
        )

        # If the default cluster filter config exists, prefer it as the
        # single source of truth for which clusters to render per meal.
        try:
            from pathlib import Path
            import json as _json
            fpath = Path('config/cluster_filter_default.json')
            if fpath.exists():
                spec = _json.loads(fpath.read_text())
                # Normalize keys and convert to sets of ints
                override = {}
                for k, v in (spec or {}).items():
                    mk = str(k).strip().lower()
                    try:
                        ids = {int(x) for x in (v or [])}
                    except Exception:
                        ids = set()
                    if mk in ("breakfast","lunch","dinner"):
                        override[mk] = ids
                # Apply only if non-empty to avoid wiping defaults unintentionally
                if override:
                    self.filter.allowed_clusters.update(override)
        except Exception:
            # Non-fatal: fall back to built-in defaults
            pass

        self.cluster_label_overrides = {
            "breakfast": {
                0:  "Pizza Meals",
                6:  "Soups & Broths",
                9:  "Snack Bars",
                10: "Simple Dairy & Grain",
                11: "Yogurt Meals",
                12: "Cereal Bowls",
                14: "Sandwiches",
                15: "Fruits",
                16: "Plant Proteins",
                17: "Sweet Bakery",
                18: "Quick Breads",
                28: "Breads & Rolls",
            },
            "lunch": {
                2:  "Snacks",
                3:  "Bean & Veggie Bowls",
                4:  "Cereal Bowls",
                5:  "Yogurt Meals",
                6:  "Pizza Snack Plates",
                7:  "Breads & Spreads",
                8:  "Pizza Meals",
                9:  "Mexican Entrées",
                11: "Pasta & Mac Plates",
                13: "Soups & Broths",
                14: "Deli Sandwich Plates",
                15: "Sandwiches",
            },
            # Insertion order = donut order in Figure 2C.
            "dinner": {
                3:  "Flavored Milk",
                4:  "Yogurt Meals",
                5:  "Bean & Veggie Bowls",
                6:  "Cereal Bowls",
                9:  "Pizza Meals",
                10: "Soups & Broths",
                11: "Mexican Plates",
                13: "Mixed-Grain Staples (Low Variety)",
                8:  "Sandwiches",
                12: "Mixed Meat Dishes",
            }
        }
        # Names for every cluster (config/cluster_names.json); existing keys keep
        # their position above, so the figure order is unchanged.
        try:
            import json
            from pathlib import Path
            names_path = Path("config") / "cluster_names.json"
            if names_path.exists():
                data = json.loads(names_path.read_text(encoding="utf-8"))
                for mt, mapping in data.items():
                    self.cluster_label_overrides.setdefault(mt, {})
                    for cid_str, label in mapping.items():
                        self.cluster_label_overrides[mt][int(cid_str)] = label
        except Exception:
            pass

    def _category_palette(self):
        # Distinct, publication-quality colors for the 9 main food categories
        palette = {
            "grains":              "#D4A574",  # warm golden brown (bread, cereal)
            "fruits":              "#FF6B6B",  # vibrant coral red (apples, berries)
            "vegetables":          "#4ECDC4",  # teal green (leafy greens, vegetables)
            "milk_dairy":          "#45B7D1",  # sky blue (milk, cheese, yogurt)
            "protein_foods":       "#96CEB4",  # sage green (meat, eggs, beans)
            "fats_oils":           "#FFEAA7",  # warm yellow (butter, oils)
            "mixed_dishes":        "#DDA0DD",  # plum purple (pizza, sandwiches, soups)
            "snacks_sweets":       "#FFB347",  # orange (candy, cookies, chips)
            "beverages":           "#87CEEB",  # light blue (coffee, tea, juice)
        }
        # convert to RGBA via Matplotlib for consistency
        import matplotlib.colors as mcolors
        return {k: mcolors.to_rgba(v) for k, v in palette.items()}


    def _cluster_food_category_rdi_deviation(self, meal_type: str, labels: "np.ndarray", food_category: str):
        """
        For each non-noise cluster, the relative deviation of the mean grams per meal of one
        food category from its target: (cluster_mean_grams - meal_target) / meal_target, where
        meal_target = daily target (_food_category_rdi_daily) * meal_fraction_by_type[meal_type].
        Grams come from the meal matrix, each food assigned with _categorize_foods_comprehensive.
        Returns (deviations, cluster_ids); returns 0.0 for a category without a target.
        """
        import numpy as np

        if food_category not in self._food_category_rdi_daily:
            return 0.0  # No RDI target for this category

        # Get meal-specific target for this food category
        daily_target = self._food_category_rdi_daily[food_category]
        frac = float(self.meal_fraction_by_type.get(meal_type, 1/3))
        meal_target = daily_target * frac

        # Get raw meal data
        meal_df = self.E.meal_data[meal_type]
        clus = [c for c in sorted(set(labels)) if c != -1]

        # Calculate mean consumption of this food category for each cluster using actual gram amounts
        deviations = []
        for cid in clus:
            m = (labels == cid)
            if not np.any(m):
                deviations.append(0.0)
                continue
            
            # Get meals in this cluster
            cluster_meals = meal_df.loc[m]
            
            # Calculate total grams for this food category across all meals in this cluster
            total_category_grams = 0.0
            num_meals = len(cluster_meals)
            
            for meal_idx, meal in cluster_meals.iterrows():
                # For each food code in this meal, check if it belongs to our target category
                for food_code, gram_amount in meal.items():
                    if gram_amount > 0:
                        # Check if this single food belongs to our target category
                        food_groups = self.E._categorize_foods_comprehensive([food_code])
                        if food_groups.get(food_category, 0) > 0:  # This food is in our target category
                            total_category_grams += gram_amount
            
            if num_meals > 0:
                mean_consumption = total_category_grams / num_meals
                
                if meal_target > 0:
                    rel_deviation = (mean_consumption - meal_target) / meal_target
                else:
                    rel_deviation = 0.0
            else:
                rel_deviation = 0.0
            
            deviations.append(float(rel_deviation))
        
        return np.array(deviations, dtype=float), clus



    def _scatter_tsne_all_with_gray(self, ax, xy, labels, meal_type, point_size=20):
        """
        Draw all points. Allowed clusters get palette colors; everything else (and noise) is gray.
        """
        allowed = self._allowed_cids(meal_type, labels)
        pal = self._build_cluster_palette(labels)

        m_allowed = np.isin(labels, list(allowed))
        m_noise   = (labels == self.filter.noise_label)
        # Base gray mask (non-allowed + noise)
        m_gray    = (~m_allowed) | m_noise

        # Optionally subsample noise points for background rendering
        if getattr(self.filter, "noise_fraction", 1.0) < 0.999:
            frac = max(0.0, float(self.filter.noise_fraction))
            if frac <= 0.0:
                # drop all noise points from background
                m_noise_sample = np.zeros_like(m_noise, dtype=bool)
            else:
                idx_noise = np.where(m_noise)[0]
                if idx_noise.size > 0:
                    rng = np.random.default_rng(int(getattr(self.filter, "noise_seed", 42)))
                    k = max(1, int(np.ceil(frac * idx_noise.size)))
                    choose = rng.choice(idx_noise, size=min(k, idx_noise.size), replace=False)
                    m_noise_sample = np.zeros_like(m_noise, dtype=bool)
                    m_noise_sample[choose] = True
                else:
                    m_noise_sample = np.zeros_like(m_noise, dtype=bool)
            # Rebuild gray mask with subsampled noise
            m_gray = (~m_allowed) | m_noise_sample

        # Optionally subsample all gray points (non-allowed + possibly subsampled noise)
        if getattr(self.filter, "gray_fraction", 1.0) < 0.999:
            gfrac = max(0.0, float(self.filter.gray_fraction))
            if gfrac <= 0.0:
                m_gray = np.zeros_like(m_gray, dtype=bool)
            else:
                idx_gray = np.where(m_gray)[0]
                if idx_gray.size > 0:
                    rng_g = np.random.default_rng(int(getattr(self.filter, "gray_seed", 43)))
                    k_g = max(1, int(np.ceil(gfrac * idx_gray.size)))
                    choose_g = rng_g.choice(idx_gray, size=min(k_g, idx_gray.size), replace=False)
                    m_gray_sample = np.zeros_like(m_gray, dtype=bool)
                    m_gray_sample[choose_g] = True
                    m_gray = m_gray_sample

        # background: gray points first (transparent with transparent outline)
        bg = self.filter.background_gray_rgba
        ax.scatter(
            xy[m_gray, 0], xy[m_gray, 1],
            s=point_size, color=bg, edgecolor=bg, linewidths=0.1, zorder=1
        )

        # overlay: selected clusters in color
        for cid in sorted(allowed):
            m = (labels == cid)
            ax.scatter(
                xy[m, 0], xy[m, 1],
                s=point_size, color=pal.get(cid, (0.7, 0.7, 0.7, 1.0)),
                edgecolor="black", linewidths=0.35, alpha=0.95, zorder=2
            )

    # ---- viz filtering helpers ----
    def _allowed_cids(self, meal_type: str, labels: np.ndarray) -> Set[int]:
        """Return the set of cluster IDs allowed for this meal (intersected with what's present)."""
        allowed = self.filter.allowed_clusters.get(meal_type, None)
        present = {int(c) for c in set(labels) if c != self.filter.noise_label}
        return present if (allowed is None) else (present & set(allowed))

    def _make_gridspec(self, fig):
        lc = self.layout
        if getattr(lc, "show_insets", False):
            ncols = 3
            width = [lc.width_left, lc.width_main, lc.width_right]
        else:
            ncols = 2
            # instead of splitting equally, shrink right column
            width = [lc.width_left + lc.width_main + 0.55,  lc.width_right * 0.45]  # narrower right col

        return fig.add_gridspec(
            nrows=2, ncols=ncols,
            height_ratios=[lc.height_top, lc.height_bottom],
            width_ratios=width,
            wspace=0.02,   # brings the pies close to the t-SNE
            hspace=lc.hspace
        )

    def _cluster_display_label(self, meal_type: str, cid: int, include_name: bool = True) -> str:
        # 1) Use explicit overrides if present (including dynamic short-name overrides)
        name = (self.cluster_label_overrides
                    .get(meal_type, {})
                    .get(cid, None))
        # 2) Otherwise fall back to engine-generated pretty name (if any)
        if name is None and include_name and hasattr(self.E, "_generate_meaningful_cluster_name"):
            pretty = self.E._generate_meaningful_cluster_name(
                meal_type, cid,
                self.E.significant_clusters.get(meal_type, {}).get(cid, {}),
                {}
            )
            name = pretty if pretty else None
        return f"{name}"

    def _annotate_tsne_clusters(self, ax, xy, labels, meal_type, cids: list[int]):
        label_offsets = {
            "lunch": {
                2: (12, 4),   # Snacks
                3: (8, -4),   # Bean & Veggie Bowls
                5: (8, -6),   # Yogurt Meals
            },
            "dinner": {
                4: (10, -4),  # Yogurt Meals
                5: (8, -4),   # Bean & Veggie Bowls
            },
        }
        for cid in cids:
            m = (labels == cid)
            if not np.any(m):
                continue
            # use median to avoid outliers
            x, y = np.median(xy[m, 0]), np.median(xy[m, 1])
            dx, dy = label_offsets.get(meal_type, {}).get(cid, (0, 0))
            ax.text(x + dx, y + dy, self._cluster_display_label(meal_type, cid, include_name=True),
                    fontsize=9, ha="left", va="center",
                    bbox=dict(fc="white", ec="0.8", lw=0.4, alpha=0.8))

    def _cluster_category_shares(self, meal_type: str, labels: np.ndarray, cats: list[str],
                                 allowed_cids: Optional[Set[int]] = None) -> pd.DataFrame:
        """
        Donut shares (clusters x cats): per meal, the cluster features' `grams_<cat>` values
        (z-scored food counts) clipped at 0 and normalised to sum to 1, averaged over the
        cluster's meals.
        """
        feats = self.E.meal_type_clusters[meal_type]["features"]
        cols = [f"grams_{c}" for c in cats if f"grams_{c}" in feats.columns]
        if not cols:
            cids = sorted(self._allowed_cids(meal_type, labels)) if allowed_cids is None else sorted(allowed_cids)
            return pd.DataFrame(index=cids)

        # choose cluster ids to compute
        cids = sorted(self._allowed_cids(meal_type, labels)) if allowed_cids is None else sorted(allowed_cids)

        shares = {}
        for cid in cids:
            m = (labels == cid)
            sub = feats.loc[m, cols].clip(lower=0).astype(float)
            tot = sub.sum(axis=1).replace(0, np.nan)
            frac = sub.div(tot, axis=0).fillna(0.0)
            shares[cid] = frac.mean(axis=0)

        df = pd.DataFrame(shares).T
        df.columns = [c.replace("grams_", "") for c in df.columns]
        for c in cats:
            if c not in df.columns:
                df[c] = 0.0
        return df[cats]



    def _draw_donut(self, ax, shares: pd.Series, palette: dict,
                    *, topk: int, inner_frac: float):
        """Draw a donut centered in [0,1]x[0,1] axes; shares is 1D Series over categories."""
        vals = shares.clip(lower=0).astype(float)
        lc = self.layout
        s = float(vals.sum())

        if s <= 1e-12:
            ax.add_patch(Wedge((0.5, 0.5), r=0.48, theta1=0, theta2=360,
                            width=max(0.01, 0.48*inner_frac),
                            facecolor=(1,1,1,0), edgecolor="0.7", linewidth=0.6))
            return

        # pick top-k, collapse remainder into "other"
        order = vals.sort_values(ascending=False)
        main = order.head(topk)
        other = order.iloc[topk:].sum()
        parts = main.tolist() + ([other] if other > 0 else [])
        labels = list(main.index) + (["other"] if other > 0 else [])

        # normalize to 1.0
        total = sum(parts)
        parts = [p/total for p in parts]

        start = 90.0
        R = min(max(lc.arcs_radius, 0.05), 0.5)  # cap at 0.5
        width = max(0.02, R * inner_frac)
        for lab, frac in zip(labels, parts):
            if frac <= 1e-6:
                continue
            theta = 360.0 * float(frac)
            col = palette.get(lab, (0.6, 0.6, 0.6, 1.0))
            ax.add_patch(Wedge((0.5, 0.5), r=R, theta1=start, theta2=start+theta,
                            width=width, facecolor=col, edgecolor="black", linewidth=0.35))
            start += theta

        ax.set_xlim(-0.02, 1.02)   # allow a hair of overspill so the donut appears larger
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.axis("off")

    def _add_category_legend_right_of_pies(self, fig, pies_ax, cats):
        pal = self._category_palette()
        import matplotlib.lines as mlines
        handles = []
        labels  = []
        for c in cats:
            if c in pal:
                handles.append(mlines.Line2D([], [], marker='o', linestyle='',
                                            markersize=7, markerfacecolor=pal[c],
                                            markeredgecolor='black', markeredgewidth=0.4))
                labels.append(c.replace('_', ' '))
        # Top-align the legend with the pies
        pies_ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.22, 0.98),
                    frameon=False, ncol=1, fontsize=9, title="Food categories", title_fontsize=10)


    def _render_cluster_arcs_panel(self, ax_area, meal_type: str, labels: np.ndarray):
        ax_area.set_anchor("W"); ax_area.margins(0); ax_area.axis("off")
        lc = self.layout
        lc.arcs_pad_y = 0.06     # more vertical space between pies
        lc.arcs_radius = 0.52    # a hair smaller to make room
        lc.arcs_label_font = 16
        lc.arcs_ncols = 2
        lc.arcs_even_rows = 6

        allowed = self._allowed_cids(meal_type, labels) if hasattr(self, "_allowed_cids") else \
                {c for c in np.unique(labels) if c != -1}
        pref_order = list((self.cluster_label_overrides.get(meal_type, {}) or {}).keys())
        if not allowed: return

        # the nine main food categories shown in the donuts
        cats = ["grains","fruits","vegetables","milk_dairy","protein_foods","fats_oils","mixed_dishes","snacks_sweets","beverages"]

        df = self._cluster_category_shares(meal_type, labels, cats, allowed_cids=allowed)
        if len(df) == 0: return

        sizes = {cid: int((labels == cid).sum()) for cid in df.index}
        by_size = sorted(df.index, key=lambda c: sizes.get(c, 0), reverse=True)
        ordered = [c for c in pref_order if c in df.index] + [c for c in by_size if c not in pref_order]
        df = df.loc[ordered]

        max_n = int(lc.arcs_max_clusters or 12)
        df = df.head(max_n)

        ncols = int(lc.arcs_ncols); nrows = int(lc.arcs_even_rows); nslots = ncols * nrows
        PAD_X, PAD_Y = lc.arcs_pad_x, lc.arcs_pad_y
        cell_w = (1 - (ncols + 1) * PAD_X) / ncols
        cell_h = (1 - (nrows + 1) * PAD_Y) / nrows
        pal = self._category_palette()

        items = list(df.iterrows())
        for i in range(nslots):
            r = i // ncols; c = i % ncols
            left   = PAD_X + c * (cell_w + PAD_X)
            bottom = 1 - PAD_Y - cell_h - r * (cell_h + PAD_Y)
            ax = ax_area.inset_axes([left, bottom, cell_w, cell_h])

            if i < len(items):
                cid, row = items[i]
                # _draw_donut renormalizes the slices to sum=1, so dropping categories is fine
                self._draw_donut(ax, row, pal, topk=lc.arcs_topk, inner_frac=lc.arcs_inner_frac)
                # Make text smaller and wrap to prevent overlap
                label_text = self._cluster_display_label(meal_type, cid, include_name=True)
                # Wrap text at 15 characters per line
                if len(label_text) > 15:
                    words = label_text.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        if len(current_line + " " + word) <= 15:
                            current_line += (" " + word) if current_line else word
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    if current_line:
                        lines.append(current_line)
                    label_text = "\n".join(lines)
                ax.text(0.5, -0.20,  # label below the donut
                        label_text,
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"))
            else:
                ax.axis("off")
        self._add_category_legend_right_of_pies(ax_area.figure, ax_area,
            cats=["grains","fruits","vegetables","milk_dairy","protein_foods","fats_oils","mixed_dishes","snacks_sweets","beverages"])



    def _add_shared_category_legend(self, fig, anchor_ax):
        """Draw one legend for food-category colors, anchored to the inset column."""
        pal = self._category_palette()
        # build handles
        import matplotlib.lines as mlines
        handles = []
        labels = []
        for name, color in pal.items():
            handles.append(mlines.Line2D([], [], marker='o', linestyle='',
                                        markersize=6, markerfacecolor=color,
                                        markeredgecolor='black', markeredgewidth=0.35))
            labels.append(name.replace('_', ' '))
        # put legend INSIDE the inset column, top-left
        anchor_ax.legend(handles, labels,
                            title="Food categories",
                            loc="upper left", bbox_to_anchor=(0.02, 0.98),
                            frameon=False, ncol=1, fontsize=8, title_fontsize=9)
        # ensure it draws even if anchor_ax has axis off
        fig.canvas.draw_idle()

    def _stack_inset_axes(self, parent_ax, n_insets):
        """
        Create evenly spaced inset axes inside parent_ax (normalized coords),
        enforcing a minimum height per inset so the bboxes get larger without
        changing the overall figure or gridspec.
        """
        import numpy as _np
        lc = self.layout

        V = float(lc.inset_vpad)
        min_h = float(lc.inset_min_cell_h)  # minimum inset height
        n_req = int(max(1, n_insets))

        # Compute how many insets can fit while respecting the min height
        # Total height used = n * min_h + (n + 1) * V  ≤ 1
        # => n ≤ floor( (1 - V) / (min_h + V) )
        max_fit = int(_np.floor((1.0 - V) / max(1e-9, (min_h + V))))
        n = max(1, min(n_req, max_fit))

        if n < n_req:
            print(f"[MealClusterViz] Reducing insets from {n_req} → {n} to honor min height={min_h:.3f}.")

        # With n fixed, finalize actual cell height (≥ min_h)
        cell_h = max(min_h, (1.0 - (n + 1) * V) / n)

        axes = []
        for i in range(n):
            bottom = 1.0 - (i + 1) * V - (i + 1) * cell_h
            ax = parent_ax.inset_axes([lc.inset_inner_left, bottom, lc.inset_inner_width, cell_h])
            ax.set_box_aspect(lc.inset_box_aspect)
            for sp in ax.spines.values():
                sp.set_linewidth(1.1)
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                fill=False, lw=1.1, ec="0.35"))
            axes.append(ax)
        return axes

    # For a meal (row in feature space), decide its dominant food category by grams_*
    def _dominant_category_from_features(self, row: pd.Series) -> str:
        gram_cols = [c for c in row.index if c.startswith("grams_")]
        if not gram_cols:
            return "other"
        grams = row[gram_cols].astype(float)
        if grams.max() <= 0:
            return "other"
        dom = grams.idxmax()            # e.g. "grams_vegetables"
        return dom.replace("grams_", "")  # -> "vegetables"

    # Top-k food *categories* inside a cluster (used to write "food names" into inset)
    def _cluster_top_food_categories(self, meal_type: str, labels: np.ndarray, cid: int, k: int = 8) -> list[str]:
        E = getattr(self, "E", self)
        feats = E.meal_type_clusters[meal_type]["features"]  # DataFrame
        mask = (labels == cid)
        if not mask.any(): 
            return []
        sub = feats.loc[mask]

        # Sum gram columns and pick top ones
        gram_cols = [c for c in sub.columns if c.startswith("grams_")]
        if not gram_cols:
            return []
        totals = sub[gram_cols].sum().sort_values(ascending=False)
        names = [c.replace("grams_", "").replace("_", " ") for c in totals.index[:k]]
        return names

    # Build a per-point color for insets (category coloring)
    def _category_colors_for_inset(self, meal_type: str, labels: np.ndarray, cid: int, palette: dict) -> np.ndarray:
        E = getattr(self, "E", self)
        feats = E.meal_type_clusters[meal_type]["features"]   # DataFrame
        mask = (labels == cid)
        sub = feats.loc[mask]
        cols = []
        for _, row in sub.iterrows():
            cat = self._dominant_category_from_features(row)
            cols.append(palette.get(cat, (0.5,0.5,0.5,1.0)))
        return np.array(cols)

    # ---------- t-SNE, insets and composite figures ----------

    def _tsne_cache_path(self, meal_type, perplexity, seed, X, n_iter):
        cache_dir = os.path.join(self.output_dir, "_tsne_cache")
        os.makedirs(cache_dir, exist_ok=True)
        # Key on the input matrix so changed features never reuse an old embedding.
        digest = hashlib.sha1(np.ascontiguousarray(np.asarray(X, dtype=np.float64)).tobytes()).hexdigest()[:12]
        return os.path.join(cache_dir, f"tsne_{meal_type}_perp{perplexity}_seed{seed}_iter{n_iter}_{digest}.pkl")

    def _tsne_embed_cached(self, X, meal_type, perplexity=64, seed=42, n_iter=1000, use_cache: Optional[bool] = None):
        """
        Compute or load cached t-SNE embeddings for a meal_type.
        """
        uc = self.use_tsne_cache if use_cache is None else use_cache
        if not uc:
            tsne = TSNE(
                n_components=2,
                random_state=seed,
                perplexity=min(perplexity, max(5, len(X)//3)),
                learning_rate="auto",
                init="pca",
                n_iter=n_iter,
                n_iter_without_progress=300,
                verbose=0,
            )
            return tsne.fit_transform(X)

        path = self._tsne_cache_path(meal_type, perplexity, seed, X, n_iter)
        if os.path.exists(path):
            try:
                emb = pd.read_pickle(path)
                if len(emb) == len(X):
                    return emb.values
            except Exception:
                pass

        tsne = TSNE(
            n_components=2,
            random_state=seed,
            perplexity=min(perplexity, max(5, len(X)//3)),  # guard tiny cohorts
            learning_rate="auto",
            init="pca",
            n_iter=n_iter,
            n_iter_without_progress=300,
            verbose=0,
        )
        Y = tsne.fit_transform(X)
        pd.to_pickle(pd.DataFrame(Y, columns=["x", "y"]), path)
        return Y

    def _build_cluster_palette(self, labels):
        """
        Returns: dict {label: rgba}
        """
        uniq = sorted(set(labels))
        # noise last for consistent color mapping
        if -1 in uniq:
            uniq = [u for u in uniq if u != -1] + [-1]
        cmap = plt.cm.get_cmap("tab20", max(1, len(uniq)))
        palette = {}
        for i, u in enumerate(uniq):
            if u == -1:
                palette[u] = (0, 0, 0, 0.55)  # semi-transparent black for noise
            else:
                palette[u] = cmap(i)
        return palette

    def _cluster_hulls(self, xy, labels, min_pts=10, allowed_cids: Optional[Set[int]] = None):
        try:
            from scipy.spatial import ConvexHull
        except Exception:
            ConvexHull = None

        allowed = sorted(self._allowed_cids("", labels)) if allowed_cids is None else sorted(allowed_cids)
        out = {}
        xy = np.asarray(xy)
        for cid in allowed:
            idx = np.where(labels == cid)[0]
            if len(idx) < min_pts:
                continue
            pts = xy[idx]
            if ConvexHull and len(idx) >= 3:
                hull = ConvexHull(pts)
                verts = pts[hull.vertices]
                path = mpath.Path(verts)
            else:
                verts = pts
                xmin, ymin = np.min(verts, axis=0); xmax, ymax = np.max(verts, axis=0)
                rect_vertices = np.array([[xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax],[xmin, ymin]])
                path = mpath.Path(rect_vertices)

            xmin, ymin = np.min(verts, axis=0); xmax, ymax = np.max(verts, axis=0)
            out[cid] = {"idx": idx, "path": path, "bbox": (xmin, ymin, xmax, ymax), "size": len(idx)}
        return out

    def _build_dot_matrix_data(self, meal_type, labels, include_only: set[int] | None = None):
        feats = self.E.meal_type_features[meal_type]
        cols = [c for c in [
            "protein_g","carbohydrate_g","fat_g","fiber_g",
            "energy_kcal","protein_ratio","carb_ratio","fat_ratio",
            "food_category_diversity","nutritional_balance","calorie_density"
        ] if c in feats.columns]

        clus = [c for c in sorted(set(labels)) if c != -1]
        if include_only is not None:
            clus = [c for c in clus if c in include_only]

        data = []
        for cid in clus:
            mask = (labels == cid)
            means = feats.loc[mask, cols].mean()
            data.append(pd.Series(means, name=cid))
        df = pd.DataFrame(data)
        df_z = (df - df.mean(axis=0)) / (df.std(axis=0).replace(0, 1))
        return df, df_z, clus


    def _draw_dot_matrix_panel_rdi(self, ax, meal_type, labels, feats_df, df_means, clus, orientation="cats_y"):
        """
        Dot matrix of the nine donut categories x clusters (orientation="cats_y": categories
        on the y axis). Dot colour = relative deviation of the cluster's mean grams of the
        category from its per-meal target (_cluster_food_category_rdi_deviation); the three
        categories without a target get 0. Dot size = 200 + 2 x mean grams per meal, but the
        grams are looked up with 'grams_<category>' keys that _categorize_foods_comprehensive
        never returns, so every dot gets the base size and the size legend falls back to
        10/50/100 g. feats_df and df_means are not used.
        """
        import numpy as np, pandas as pd

        # categories with a daily target (dot colour)
        cats_with_rdi = ["grains","fruits","vegetables","milk_dairy","protein_foods","fats_oils"]
        # Use the same 9 categories as the pie plots
        cats_all = ["grams_grains","grams_fruits","grams_vegetables","grams_milk_dairy","grams_protein_foods","grams_fats_oils","grams_mixed_dishes","grams_snacks_sweets","grams_beverages"]

        # --- build mean grams per cluster/category from raw meal data ---
        meal_df = self.E.meal_data[meal_type]  # Raw meal matrix (rows=meals, cols=food_codes, values=grams)
        
        rows = []
        for cid in clus:
            m = (labels == cid)
            cluster_meals = meal_df.loc[m]  # Get raw meal data for this cluster
            
            # Compute absolute grams per RDI category for this cluster
            row = {}
            for cat in cats_all:
                total_grams = 0.0
                num_meals = len(cluster_meals)
                
                for meal_idx, meal in cluster_meals.iterrows():
                    # For each food code in this meal, check if it belongs to our target category
                    for food_code, gram_amount in meal.items():
                        if gram_amount > 0:
                            # Check if this food belongs to our target category
                            food_groups = self.E._categorize_foods_comprehensive([food_code])
                            if food_groups.get(cat, 0) > 0:  # This food is in our target category
                                total_grams += gram_amount
                
                # Mean grams per meal for this category in this cluster
                row[cat] = total_grams / num_meals if num_meals > 0 else 0.0
            
            rows.append(row)
        
        G = pd.DataFrame(rows, index=[f"C{c}" for c in clus], columns=cats_all)  # grams matrix

        # --- color by individual food category RDI deviation ---
        # compute deviations for each category with RDI
        category_deviations = {}
        for cat in cats_with_rdi:
            dist, clus_order = self._cluster_food_category_rdi_deviation(meal_type, labels, cat)
            category_deviations[cat] = {cid: dist[i] for i, cid in enumerate(clus_order)}

        # global color scale over all categories
        all_devs = np.array([v for cat in cats_with_rdi for v in category_deviations.get(cat, {}).values()] or [0.0])
        vmax = np.percentile(np.abs(all_devs), 95) + 1e-9
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)
        cmap = plt.cm.coolwarm

        # --- marker sizes: actual grams with base size for visibility ---
        BASE_DOT_SIZE = 200.0  # Much larger base dot size for visibility
        
        def _size_from_grams(g):
            # Scale grams to make dots more visible, with base size
            scaled_grams = max(0.0, float(g)) * 2.0  # Aggressive scaling of grams
            return BASE_DOT_SIZE + scaled_grams

        Xpos, Ypos, Ccol, Ssz = [], [], [], []
        catsN, clusN = len(cats_all), len(clus)

        for r, cid in enumerate(clus):          # r indexes cluster
            for c, cat in enumerate(cats_all):  # c indexes category
                if orientation == "cats_y":
                    x, y = r, c         # x=cluster, y=category
                else:
                    x, y = c, r         # x=category, y=cluster

                Xpos.append(x); Ypos.append(y)

                # color from deviation (only defined for cats_with_rdi)
                # Map column name back to category name for lookup
                cat_name = cat.replace('grams_', '') if cat.startswith('grams_') else cat
                if cat_name in category_deviations:
                    color_val = category_deviations[cat_name].get(cid, 0.0)
                else:
                    color_val = 0.0  # No RDI deviation available for this category
                Ccol.append(cmap(norm(color_val)))

                # size from grams
                grams = G.iloc[r, c]
                Ssz.append(_size_from_grams(grams))

        ax.scatter(Xpos, Ypos, s=Ssz, c=Ccol, edgecolor="black", linewidths=0.35)

        # ticks/labels consistent with coordinates
        if orientation == "cats_y":
            ax.set_yticks(range(catsN)); ax.set_yticklabels([c.replace("_"," ") for c in cats_all], fontsize=12)
            ax.set_xticks(range(clusN)); ax.set_xticklabels(
                [self._cluster_display_label(meal_type, cid, include_name=True) for cid in clus],
                rotation=45, ha="right", fontsize=12
            )
            ax.set_xlim(-0.5, clusN-0.5); ax.set_ylim(-0.5, catsN-0.5)
        else:
            ax.set_yticks(range(clusN)); ax.set_yticklabels(
                [self._cluster_display_label(meal_type, cid, include_name=True) for cid in clus], fontsize=12
            )
            ax.set_xticks(range(catsN)); ax.set_xticklabels([c.replace("_"," ") for c in cats_all],
                rotation=45, ha="left", fontsize=12)
            ax.tick_params(axis="x", bottom=False, labelbottom=False, top=True, labeltop=True)
            ax.set_xlim(-0.5, catsN-0.5); ax.set_ylim(-0.5, clusN-0.5)

        # ---- colorbar ----
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Food category RDI deviation (per circle)", fontsize=12)
        cbar.ax.tick_params(labelsize=11)

        # ---- size legend *under* the colorbar ----
        # Use the SAME sizing function as the actual dots
        def _size_from_grams_legend(g):
            # Same function as above - base size + scaled grams
            scaled_grams = max(0.0, float(g)) * 2.0  # Aggressive scaling of grams
            return BASE_DOT_SIZE + scaled_grams

        # Get actual gram values to create legend
        grams_vals = G.values.flatten()
        grams_vals = grams_vals[grams_vals > 0]  # Only positive values
        
        if len(grams_vals) > 0:
            # Get min and max, round to nice values
            gmin = np.min(grams_vals)
            gmax = np.max(grams_vals)
            
            # Round min and max to nearest 5
            gmin_rounded = max(1.0, np.round(gmin / 5.0) * 5.0)
            gmax_rounded = max(1.0, np.round(gmax / 5.0) * 5.0)
            
            # Calculate midpoint between rounded min and max
            gmid_rounded = np.round((gmin_rounded + gmax_rounded) / 2.0 / 5.0) * 5.0
            
            vals = [gmin_rounded, gmid_rounded, gmax_rounded]
            vals = sorted(list(set(vals)))  # remove duplicates and sort
            
            # Ensure we have at least 3 distinct values
            if len(vals) < 3:
                if len(vals) == 1:
                    # If all values are the same, create a range around it
                    base_val = vals[0]
                    vals = [max(1.0, base_val - 10), base_val, base_val + 10]
                elif len(vals) == 2:
                    # Add a middle value
                    vals.append((vals[0] + vals[1]) / 2)
                    vals = [max(1.0, np.round(v / 5.0) * 5.0) for v in vals]
                    vals = sorted(list(set(vals)))
        else:
            # Fallback if no data
            vals = [10, 50, 100]
        import matplotlib.lines as mlines
        handles = [mlines.Line2D([], [], marker='o', linestyle='',
                    markersize=np.sqrt(_size_from_grams_legend(v)),  # legend expects diameter (points), not area
                    markerfacecolor='white', markeredgecolor='black', markeredgewidth=0.9,
                    label=f"{int(v)} g") for v in vals]

        leg_ax = cbar.ax.inset_axes([-0.12, -0.52, 1.30, 0.48])  # under the colorbar
        leg_ax.axis("off")
        leg_ax.legend(handles=handles, title="Dot size = mean grams",
                    loc="center", frameon=False, fontsize=9, title_fontsize=10, ncol=1)  # Vertical legend


    def _rank_clusters_by_nutritional_difference(self, meal_type, labels, topk=6,
                                                w_geo=1.0, w_breadth=0.5, w_maxeff=1.0, w_strength=0.25,
                                                allowed_cids: Optional[Set[int]] = None):
        feats = self.E.meal_type_clusters[meal_type]['features']
        sigs  = self.E.significant_clusters.get(meal_type, {})

        candidates = sorted(self._allowed_cids(meal_type, labels)) if allowed_cids is None else sorted(allowed_cids)
        if not candidates:
            return []

        centroids = {cid: feats[labels == cid].mean().values for cid in candidates}

        def _nearest_dist(cid):
            others = [o for o in candidates if o != cid]
            if not others: return 0.0
            this = centroids[cid]
            return min(np.linalg.norm(this - centroids[o]) for o in others)

        ranked = []
        for cid in candidates:
            geo = _nearest_dist(cid)
            sdict = sigs.get(cid, {})
            n_sig = len(sdict)
            if n_sig:
                effs = [abs(v.get('cohen_d', 0.0)) for v in sdict.values()]
                max_eff = max(effs) if effs else 0.0
                import math
                combo = [abs(v.get('cohen_d', 0.0)) * max(0.0, -math.log10(max(v.get('q_value', 1.0), 1e-300)))
                         for v in sdict.values()]
                mean_strength = float(np.mean(combo)) if combo else 0.0
            else:
                max_eff = 0.0; mean_strength = 0.0

            score = (w_geo * geo) + (w_breadth * n_sig) + (w_maxeff * max_eff) + (w_strength * mean_strength)
            ranked.append((cid, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:topk]]
        

    def compose_cluster_landscape(self, meal_type, outdir, *, dpi=300, layout=None):
        """
        Write <meal>_cluster_landscape.png/.svg (t-SNE of the cluster features, perplexity 64,
        seed 42, with the archetype donuts) and <meal>_dot_matrix.png/.svg to `outdir`.
        `layout` is not used.
        """
        E = self.E
        clusters = E.meal_type_clusters[meal_type]
        labels   = clusters["labels"]
        feats_df = clusters["features"]
        X        = feats_df.values

        ts = self._tsne_embed_cached(X, meal_type, perplexity=64, seed=42)
        pal = self._build_cluster_palette(labels)
        cat_pal = self._category_palette()

        fig = plt.figure(figsize=(19.2, 12.4), constrained_layout=False)
        gs  = self._make_gridspec(fig)

        if getattr(self.layout, "show_insets", False):
            ax_inset_col = fig.add_subplot(gs[0, 0]); ax_inset_col.axis("off")
            ax_main      = fig.add_subplot(gs[0, 1])
            ax_pies      = fig.add_subplot(gs[0, 2])     # RIGHT = pies
        else:
            ax_main = fig.add_subplot(gs[0, 0])
            ax_pies = fig.add_subplot(gs[0, 1])          # RIGHT = pies

        # --- MAIN t-SNE (gray out non-selected and noise) ---
        self._scatter_tsne_all_with_gray(ax_main, ts, labels, meal_type, point_size=20)
        allowed = sorted(self._allowed_cids(meal_type, labels))
        self._annotate_tsne_clusters(ax_main, ts, labels, meal_type, allowed)

        ax_main.set_title(f"{meal_type.title()} significant clusters", pad=6, fontsize=15)
        ax_main.set_xlabel("t-SNE 1", fontsize=12)
        ax_main.set_ylabel("t-SNE 2", fontsize=12)
        ax_main.tick_params(labelsize=10)

        # --- Insets (only if enabled) ---
        if getattr(self.layout, "show_insets", False):
            hulls  = self._cluster_hulls(ts, labels, min_pts=15)  # optionally pass allowed_cids
            chosen = self._rank_clusters_by_nutritional_difference(meal_type, labels, topk=self.layout.max_insets)
            inset_axes = self._stack_inset_axes(ax_inset_col, len(chosen))
            for ax_in, cid in zip(inset_axes, chosen):
                m = (labels == cid)
                cols_arr = self._category_colors_for_inset(meal_type, labels, cid, cat_pal)
                ax_in.scatter(ts[m,0], ts[m,1], s=16, c=cols_arr, alpha=0.95,
                            edgecolor="black", linewidths=0.35)
                if cid in hulls:
                    xmin, ymin, xmax, ymax = hulls[cid]["bbox"]
                else:
                    xmin, ymin = ts[m,:].min(axis=0); xmax, ymax = ts[m,:].max(axis=0)
                dx, dy = xmax-xmin, ymax-ymin
                ax_in.set_xlim(xmin-0.12*dx, xmax+0.12*dx)
                ax_in.set_ylim(ymin-0.12*dy, ymax+0.12*dy)
                ax_in.set_xticks([]); ax_in.set_yticks([])
                ax_in.add_patch(Rectangle((0,0), 1, 1, transform=ax_in.transAxes,
                                        fill=False, lw=1.1, ec="0.35"))
                title_txt = self._cluster_display_label(meal_type, cid)
                ax_in.set_title(title_txt, fontsize=9, pad=3)
                names = self._cluster_top_food_categories(meal_type, labels, cid, k=7)
                if names:
                    ax_in.text(0.01, 0.99, "\n".join(names), transform=ax_in.transAxes,
                            va="top", ha="left", fontsize=8,
                            bbox=dict(fc="white", ec="0.8", lw=0.5, alpha=0.85))

                rect = Rectangle((xmin, ymin), dx, dy, fill=False, lw=1.0, ec="black", linestyle="--")
                ax_main.add_patch(rect)
                x0, y0 = xmin + dx/2, ymin + dy/2
                fig.add_artist(ConnectionPatch(
                    xyA=(1.0, 0.5), coordsA=ax_in.transAxes,
                    xyB=(x0, y0),   coordsB=ax_main.transData,
                    arrowstyle="-", lw=1.0, color="black"
                ))
            self._add_shared_category_legend(fig, ax_inset_col)

        # --- Right pies ---
        self.layout.arcs_ncols = 2
        self.layout.arcs_nrows = 6

        self.layout.arcs_max_clusters = 12
        self._render_cluster_arcs_panel(ax_pies, meal_type, labels)

        fig.suptitle(f"{meal_type.title()} – Cluster Landscape", y=0.995, fontsize=16)
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, f"{meal_type}_cluster_landscape.png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(os.path.join(outdir, f"{meal_type}_cluster_landscape.svg"), bbox_inches="tight")
        plt.close(fig)
        
        # --- Create separate dot matrix plot ---
        self._create_separate_dot_matrix_plot(meal_type, outdir, labels, feats_df, dpi=dpi)

    def create_combined_tsne_plot(self, outdir, *, dpi=300):
        """
        Figure 2: one row per meal type (panels A-C) with the t-SNE landscape and the archetype
        donuts; writes combined_tsne_landscape.png/.svg to `outdir`.
        """
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(13.0, 12.6), constrained_layout=False)
        
        # Use the same width ratios as individual plots
        lc = self.layout
        width_ratios = [lc.width_left + lc.width_main + 0.55, lc.width_right * 0.95]
        
        # Create gridspec with 3 rows and 2 columns
        gs = fig.add_gridspec(
            nrows=3, ncols=2,
            height_ratios=[1, 1, 1],
            width_ratios=width_ratios,
            wspace=0.05,
            hspace=0.18
        )
        
        meal_types = ['breakfast', 'lunch', 'dinner']
        panel_labels = ['A', 'B', 'C']
        
        for i, meal_type in enumerate(meal_types):
            if meal_type not in self.E.meal_type_clusters:
                continue
                
            # Left column: t-SNE plot
            ax_tsne = fig.add_subplot(gs[i, 0])
            # Right column: pie plots
            ax_pies = fig.add_subplot(gs[i, 1])
            
            # Get data for this meal type
            clusters = self.E.meal_type_clusters[meal_type]
            labels = clusters["labels"]
            feats_df = clusters["features"]
            X = feats_df.values
            
            # Get t-SNE embedding
            ts = self._tsne_embed_cached(X, meal_type, perplexity=64, seed=42)
            
            # Draw the t-SNE plot
            self._scatter_tsne_all_with_gray(ax_tsne, ts, labels, meal_type, point_size=20)
            allowed = sorted(self._allowed_cids(meal_type, labels))
            self._annotate_tsne_clusters(ax_tsne, ts, labels, meal_type, allowed)
            xmin, xmax = ax_tsne.get_xlim()
            ax_tsne.set_xlim(xmin, xmax + (xmax - xmin) * 0.12)
            
            ax_tsne.set_title(f"{meal_type.title()} significant clusters", pad=4, fontsize=10)
            ax_tsne.set_xlabel("t-SNE 1", fontsize=9)
            ax_tsne.set_ylabel("t-SNE 2", fontsize=9)
            ax_tsne.tick_params(labelsize=8)
            ax_tsne.text(
                -0.055, 1.03, panel_labels[i],
                transform=ax_tsne.transAxes,
                fontsize=18,
                fontweight="bold",
                va="top",
                ha="left",
                clip_on=False,
            )
            
            # Draw the pie plots
            ax_pies.set_anchor("W"); ax_pies.margins(0); ax_pies.axis("off")
            
            # Configure pie plot layout
            lc.arcs_pad_y = 0.06
            lc.arcs_radius = 0.52
            lc.arcs_label_font = 8.5
            lc.arcs_ncols = 2
            lc.arcs_nrows = 6
            lc.arcs_max_clusters = 12
            
            # Render pie plots
            self._render_cluster_arcs_panel(ax_pies, meal_type, labels)
            
            ax_pies.set_title("Meal clusters", pad=4, fontsize=10)
        
        # Save the combined plot
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, "combined_tsne_landscape.png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(os.path.join(outdir, "combined_tsne_landscape.svg"), bbox_inches="tight")
        plt.close(fig)

    def _create_separate_dot_matrix_plot(self, meal_type, outdir, labels, feats_df, *, dpi=300):
        """Create a separate plot for the dot matrix visualization."""
        allowed = self._allowed_cids(meal_type, labels)
        df_means, df_z, clus = self._build_dot_matrix_data(meal_type, labels, include_only=allowed)
        
        # Create a new figure for the dot matrix
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Draw the dot matrix
        self._draw_dot_matrix_panel_rdi(ax, meal_type, labels, feats_df, df_means, clus, orientation="cats_y")
        
        fig.suptitle(f"{meal_type.title()} – Food Category Dot Matrix", y=0.95, fontsize=16)
        
        # Save the separate dot matrix plot
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, f"{meal_type}_dot_matrix.png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(os.path.join(outdir, f"{meal_type}_dot_matrix.svg"), bbox_inches="tight")
        plt.close(fig)
