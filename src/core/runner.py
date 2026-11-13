#!/usr/bin/env python3
"""
Evaluation helpers shared by cli/run_multi_meal_cvae.py (`make cvae_train`).

- compute_metrics_by_cluster(): one row per cluster of presence metrics (AUROC, AUPRC, Brier,
  ECE), set-level F1, item-count error and amount error.
- CVAEPanelVisualizer.render_dashboard(): the 4x3 evaluation dashboard that
  cli/run_multi_meal_cvae.py saves per meal type as <meal>_multi_meal_panel.png.
- _print_pretty_table() and set_global_seed(); setup_logging() is used by
  analyses/condition_ablation.py.
"""

import os
import logging
import random
from datetime import datetime

import numpy as np
import pandas as pd
from typing import Optional, Tuple

from sklearn.metrics import precision_recall_curve, roc_curve, auc

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt

# --- helpers ------------------------------------------------------------
def _f1_at_k(y_true_bin, y_scores, k):
    import numpy as np
    N, D = y_true_bin.shape
    if np.isscalar(k):
        k = np.full(N, int(k))
    k = np.clip(k, 0, D)
    idx = np.argsort(-y_scores, axis=1)
    y_pred = np.zeros_like(y_true_bin, dtype=np.int32)
    for i in range(N):
        if k[i] > 0:
            y_pred[i, idx[i, :k[i]]] = 1
    tp = (y_pred & y_true_bin).sum(axis=1)
    fp = (y_pred & (1 - y_true_bin)).sum(axis=1)
    fn = ((1 - y_pred) & y_true_bin).sum(axis=1)
    denom = (2 * tp + fp + fn)
    f1 = np.where(denom > 0, 2 * tp / denom, 1.0)  # define empty-meal F1 = 1 when both are empty
    return float(f1.mean())

def _soft_fbeta_meal(y_true_bin, y_scores, beta=1.0, eps=1e-8):
    import numpy as np
    beta2 = beta * beta
    tp = (y_scores * y_true_bin).sum(axis=1)
    pred_sum = y_scores.sum(axis=1) + eps
    true_sum = y_true_bin.sum(axis=1) + eps
    prec = tp / pred_sum
    rec  = tp / true_sum
    sf = (1 + beta2) * (prec * rec) / (beta2 * prec + rec + eps)
    # if a meal truly has 0 items, count perfect score when predictions also sum to ~0
    sf = np.where(true_sum <= 1e-12, (pred_sum <= 1e-12).astype(float), sf)
    return float(sf.mean())

def _ece(y_true_flat, y_prob_flat, n_bins=10):
    import numpy as np
    bins = np.linspace(0., 1., n_bins + 1)
    m = y_prob_flat.size
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob_flat >= bins[i]) & (y_prob_flat < bins[i+1])
        if mask.any():
            conf = y_prob_flat[mask].mean()
            acc  = y_true_flat[mask].mean()
            ece += (mask.sum() / m) * abs(acc - conf)
    return float(ece)

# --- main cluster report ------------------------------------------------
def compute_metrics_by_cluster(meal_type, y_true_grams, p_presence, e_present, exp_grams, C, eps=1.0):
    """Per-cluster metrics, one row per value of C (largest cluster first); a food counts as
    present when its true grams exceed eps."""
    import numpy as np, pandas as pd
    from sklearn.metrics import r2_score, roc_auc_score, average_precision_score

    # The model predicts presence only; presence probabilities stand in for amounts
    if e_present is None:
        e_present = p_presence
    if exp_grams is None:
        exp_grams = p_presence

    y_bin = (y_true_grams > eps).astype(np.int32)
    clusters = np.unique(C)
    rows = []

    for c in clusters:
        mask = (C == c)
        yg  = y_true_grams[mask]
        yb  = y_bin[mask]
        pp  = p_presence[mask]
        ep  = e_present[mask]
        ex  = exp_grams[mask]

        # presence (flattened micro)
        yb_flat = yb.ravel()
        pp_flat = pp.ravel()
        has_pos_and_neg = (yb_flat.min() == 0) and (yb_flat.max() == 1)

        try:
            auroc = roc_auc_score(yb_flat, pp_flat) if has_pos_and_neg else np.nan
        except Exception:
            auroc = np.nan
        
        # Handle NaN values in predictions for average precision score
        if (yb_flat.sum() > 0) and not np.any(np.isnan(pp_flat)):
            try:
                auprc = average_precision_score(yb_flat, pp_flat)
            except Exception:
                auprc = np.nan
        else:
            auprc = np.nan
        # Handle NaN values in Brier score and ECE computation
        if not np.any(np.isnan(pp_flat)):
            brier = float(np.mean((pp_flat - yb_flat)**2))
            ece   = _ece(yb_flat, pp_flat, n_bins=10)
        else:
            brier = np.nan
            ece   = np.nan

        # set-level F1 variants
        trueK = yb.sum(axis=1).astype(int)
        hatK  = np.rint(pp.sum(axis=1)).astype(int)
        
        # Handle NaN values in F1 computations
        if not np.any(np.isnan(pp)):
            f1_at_trueK = _f1_at_k(yb, pp, trueK)
            f1_at_hatK  = _f1_at_k(yb, pp, hatK)
            soft_f1     = _soft_fbeta_meal(yb, pp, beta=1.0)
            soft_f05    = _soft_fbeta_meal(yb, pp, beta=0.5)
        else:
            f1_at_trueK = f1_at_hatK = soft_f1 = soft_f05 = np.nan

        # counts (per meal) - handle NaN in predictions
        if not np.any(np.isnan(pp)):
            count_mae  = float(np.mean(np.abs(hatK - trueK)))
            count_bias = float(np.mean(hatK - trueK))
        else:
            count_mae = count_bias = np.nan

        # amounts
        present_mask = (yb == 1)
        if present_mask.any():
            yg_present = yg[present_mask]
            ep_present = ep[present_mask]
            
            # Check for NaN values in predictions
            if not np.any(np.isnan(ep_present)) and not np.any(np.isnan(yg_present)):
                try:
                    r2_present = float(r2_score(yg_present, ep_present))
                except Exception:
                    r2_present = np.nan
                mae_present = float(np.mean(np.abs(yg_present - ep_present)))
            else:
                r2_present, mae_present = np.nan, np.nan
        else:
            r2_present, mae_present = np.nan, np.nan

        # Handle NaN values in total grams computation
        if not np.any(np.isnan(ex)) and not np.any(np.isnan(yg)):
            mae_total = float(np.mean(np.abs(yg.sum(axis=1) - ex.sum(axis=1))))
        else:
            mae_total = np.nan

        rows.append(dict(
            meal_type=meal_type, cluster=int(c), n_meals=int(mask.sum()),
            auroc_presence=auroc, auprc_presence=auprc, brier_presence=brier, ece_presence=ece,
            f1_set_at_trueK=f1_at_trueK, f1_set_at_hatK=f1_at_hatK,
            soft_f1_meal=soft_f1, soft_f05_meal=soft_f05,
            count_mae=count_mae, count_bias=count_bias,
            r2_amount_present=r2_present, mae_amount_present=mae_present,
            mae_total_grams=mae_total
        ))
    df = pd.DataFrame(rows).sort_values(["n_meals"], ascending=False)
    return df


class CVAEPanelVisualizer:
    """
    Evaluation dashboard for the CVAE, one PNG per meal type via render_dashboard().

    Panels:
      - Presence: reliability, PR (micro), ROC with operating points.
      - Counts: true vs predicted foods per meal.
      - Clusters: caterpillar (count bias with CIs), beeswarm (top-K), per-cluster metric heatmap.
    """

    def __init__(self, meal_name: str, out_dir: str):
        self.meal = str(meal_name)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)


    @staticmethod
    def _bootstrap_ci(x: np.ndarray, stat="median", alpha: float = 0.05, n_boot: int = 500,
                      rng: Optional[np.random.Generator] = None) -> Tuple[float, Tuple[float,float]]:
        rng = rng or np.random.default_rng(123)
        x = np.asarray(x).astype(float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float("nan"), (float("nan"), float("nan"))
        if stat == "median":
            s0 = float(np.median(x))
            boot = [np.median(rng.choice(x, size=x.size, replace=True)) for _ in range(n_boot)]
        elif stat == "mean":
            s0 = float(np.mean(x))
            boot = [np.mean(rng.choice(x, size=x.size, replace=True)) for _ in range(n_boot)]
        else:
            raise ValueError("stat must be 'median' or 'mean'")
        lo, hi = np.quantile(boot, [alpha/2, 1.0 - alpha/2])
        return s0, (float(lo), float(hi))

    @staticmethod
    def _op_point(y_true_bin: np.ndarray, p: np.ndarray, thr: float) -> Tuple[float, float]:
        y = y_true_bin.ravel().astype(bool)
        yhat = (p.ravel() >= float(thr))
        tp = np.logical_and(yhat, y).sum()
        tn = np.logical_and(~yhat, ~y).sum()
        fp = np.logical_and(yhat, ~y).sum()
        fn = np.logical_and(~yhat, y).sum()
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        return float(fpr), float(tpr)

    @staticmethod
    def _compute_presence_counts(p_presence: np.ndarray, threshold: float) -> np.ndarray:
        """Compute thresholded counts for presence-only models"""
        return (p_presence >= threshold).sum(axis=1).astype(np.float32)

    @staticmethod
    def _savefig(fig, path: str, dpi=180, bbox_inches="tight"):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
            plt.close(fig)
            print(f"✓ Plot saved to: {path}")
        except Exception as e:
            print(f"✗ Failed to save plot to {path}: {e}")
            plt.close(fig)
            raise

    # ----------------------- presence calibration -----------------------
    def plot_presence_reliability(self, p_presence: np.ndarray, y_true_bin: np.ndarray, ax=None, n_bins: int = 12):
        p = np.asarray(p_presence).ravel()
        y = np.asarray(y_true_bin).ravel().astype(int)
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.digitize(p, bins[1:-1], right=False)
        xs, ys = [], []
        for b in range(n_bins):
            m = (idx == b)
            if not np.any(m): 
                continue
            xs.append(float(np.mean(p[m])))
            ys.append(float(np.mean(y[m])))
        if ax is None:
            fig, ax = plt.subplots(figsize=(4,3))
        ax.plot([0,1], [0,1], ls="--", c="0.5")
        if xs:
            ax.plot(xs, ys, marker="o")
        ax.set_title("Presence reliability")
        ax.set_xlabel("Mean predicted p")
        ax.set_ylabel("Empirical freq")
        return ax

    def plot_presence_pr_micro(self, p_presence: np.ndarray, y_true_bin: np.ndarray, ax=None):
        y = np.asarray(y_true_bin).ravel().astype(int)
        p = np.asarray(p_presence).ravel()
        pr, rc, _ = precision_recall_curve(y, p)
        if ax is None:
            fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(rc, pr)
        ax.set_title("Presence PR (micro)")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        return ax

    def plot_presence_roc(self, p_presence: np.ndarray, y_true_bin: np.ndarray,
                          thresholds: Optional[np.ndarray] = None,
                          sample_items: int = 200,
                          ax=None):
        y = np.asarray(y_true_bin).ravel().astype(int)
        p = np.asarray(p_presence).ravel()
        fpr, tpr, _ = roc_curve(y, p)
        au = auc(fpr, tpr)

        if ax is None:
            fig, ax = plt.subplots(figsize=(4,3))
        ax.plot([0,1],[0,1], ls="--", c="0.8")
        ax.plot(fpr, tpr, lw=2, label=f"Micro ROC (AUROC={au:.3f})")
        ax.set_title("Presence ROC")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")

        # operating point(s)
        if thresholds is not None:
            thr_arr = np.asarray(thresholds)
            if thr_arr.ndim == 0:
                fpr0, tpr0 = self._op_point(y_true_bin, p_presence, float(thr_arr))
                ax.scatter([fpr0], [tpr0], marker="o", s=40, label=f"op @ t={float(thr_arr):.2f}")
            else:
                D = thr_arr.size
                rng = np.random.default_rng(0)
                pick = rng.choice(np.arange(D), size=min(D, sample_items), replace=False)
                for j in pick:
                    fprj, tprj = self._op_point(y_true_bin[:, j], p_presence[:, j], float(thr_arr[j]))
                    ax.scatter([fprj], [tprj], s=10, alpha=0.25, c="C1")
                # mean per-item op-point
                fprs, tprs = [], []
                for j in range(D):
                    fj, tj = self._op_point(y_true_bin[:, j], p_presence[:, j], float(thr_arr[j]))
                    fprs.append(fj); tprs.append(tj)
                ax.scatter([np.mean(fprs)], [np.mean(tprs)], marker="D", s=55, c="C1", label="per-item mean op")
        ax.legend(loc="lower right")
        return ax

    def plot_threshold_hist(self, thresholds: np.ndarray, ax=None, title="Presence thresholds"):
        th = np.asarray(thresholds).ravel()
        if ax is None:
            fig, ax = plt.subplots(figsize=(4,3))
        ax.hist(th, bins=15)
        ax.set_title(title)
        ax.set_xlabel("threshold")
        ax.set_ylabel("count")
        return ax

    # ----------------------- clusters -----------------------
    def plot_cluster_caterpillar_count_bias(self, C: np.ndarray,
                                            y_true_bin: np.ndarray,
                                            y_pred_bin: np.ndarray,
                                            min_n: int = 30,
                                            ax=None):
        C = np.asarray(C).astype(int).ravel()
        ytb = y_true_bin.astype(int)
        ypb = y_pred_bin.astype(int)
        ids = np.unique(C)
        rows = []
        for k in ids:
            m = (C == k)
            n = int(m.sum())
            if n < min_n: 
                continue
            count_true = ytb[m].sum(axis=1)
            count_pred = ypb[m].sum(axis=1)
            mt, (lt, ht) = self._bootstrap_ci(count_true, stat="mean")
            mp, (lp, hp) = self._bootstrap_ci(count_pred, stat="mean")
            rows.append((k, n, mt, lt, ht, mp, lp, hp, mp - mt))
        if not rows:
            return ax
        rows.sort(key=lambda r: abs(r[-1]), reverse=True)
        lbl = [f"C{int(k)} (n={n})" for (k,n,*_) in rows]
        idx = np.arange(len(rows))

        true_means = [r[2] for r in rows]; true_lo = [r[3] for r in rows]; true_hi = [r[4] for r in rows]
        pred_means = [r[5] for r in rows]; pred_lo = [r[6] for r in rows]; pred_hi = [r[7] for r in rows]

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, max(3, 0.25*len(rows))))
        for i in range(len(rows)):
            ax.plot([true_means[i], pred_means[i]], [i, i], c="0.7", zorder=1)
        ax.errorbar(true_means, idx, xerr=[np.array(true_means)-np.array(true_lo),
                                           np.array(true_hi)-np.array(true_means)],
                    fmt="o", label="True", capsize=2, zorder=2)
        ax.errorbar(pred_means, idx, xerr=[np.array(pred_means)-np.array(pred_lo),
                                           np.array(pred_hi)-np.array(pred_means)],
                    fmt="o", label="Pred", capsize=2, zorder=3)
        ax.set_yticks(idx)
        ax.set_yticklabels(lbl, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("# foods per meal (mean ± CI)")
        ax.set_title("Cluster-wise food count bias (sorted by |bias|)")
        ax.legend()
        return ax


    def plot_cluster_beeswarm_count_error(self, C: np.ndarray, y_true_bin: np.ndarray,
                                          y_pred_bin: np.ndarray, top_k: int = 12, ax=None):
        C = np.asarray(C).astype(int).ravel()
        ytb = y_true_bin.astype(int); ypb = y_pred_bin.astype(int)
        ids, counts = np.unique(C, return_counts=True)
        order = ids[np.argsort(-counts)][:top_k]

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(6, 0.9*top_k), 3.5))
        xs, xticks = [], []
        for i, k in enumerate(order):
            m = (C == k)
            err = (ypb[m].sum(axis=1) - ytb[m].sum(axis=1)).astype(float)
            rng = np.random.default_rng(i)
            jitter = (rng.random(err.size) - 0.5) * 0.6
            ax.scatter(np.full(err.size, i) + jitter, err, s=10, alpha=0.35)
            ax.scatter([i], [np.median(err)], c="k", marker="|", s=120, lw=1.5)
            xs.append(i); xticks.append(f"C{k}")
        ax.axhline(0, ls="--", c="0.7")
        ax.set_xticks(xs); ax.set_xticklabels(xticks, rotation=0)
        ax.set_ylabel("#pred - #true (foods/meal)")
        ax.set_title("Per-meal count error by cluster (top-K by support)")
        return ax

    def plot_cluster_heatmap_in_dashboard(self, cluster_df, ax=None):
        """Cluster heatmap adapted for dashboard subplot"""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        
        # Select key metrics for heatmap
        metrics = ['auroc_presence','auprc_presence','brier_presence','ece_presence',
                   'f1_set_at_trueK','soft_f1_meal','count_mae','count_bias',
                   'r2_amount_present','mae_amount_present','mae_total_grams']
        
        # Filter to available metrics
        available_metrics = [m for m in metrics if m in cluster_df.columns]
        M = cluster_df[available_metrics].to_numpy()
        
        # Create heatmap
        im = ax.imshow(M, aspect='auto', interpolation='nearest', cmap='RdYlBu_r')
        
        # Set labels
        ax.set_yticks(range(len(cluster_df)))
        ax.set_yticklabels(cluster_df['cluster'].astype(str))
        ax.set_xticks(range(len(available_metrics)))
        ax.set_xticklabels(available_metrics, rotation=45, ha='right', fontsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title('Per-cluster metrics')
        
        return ax

    def render_dashboard(self,
                         y_true_grams: np.ndarray,
                         y_true_bin: np.ndarray,
                         p_presence: np.ndarray,
                         e_present: Optional[np.ndarray] = None,
                         exp_grams: Optional[np.ndarray] = None,
                         thresholds: Optional[np.ndarray] = None,
                         C: Optional[np.ndarray] = None,
                         save_as: Optional[str] = None,
                         metrics: dict | None = None):
        """
        4x3 dashboard of the presence model:
         [0,0] Reliability          [0,1] PR (micro)       [0,2] ROC (+ops)
         [1,0] Counts scatter       [1,1] K histogram      [1,2] Threshold hist
         [2,0] Cluster caterpillar  [2,1] Cluster heatmap  [2,2] Cluster metrics
         [3,0] Cluster beeswarm     [3,1] Count bias       [3,2] Calibration

        Saved to save_as (default <out_dir>/<meal>_comprehensive_panel.png).
        """
        import matplotlib.gridspec as gridspec

        print(f"Creating dashboard for {self.meal}...")
        print(f"Data shapes: y_true_grams={y_true_grams.shape}, p_presence={p_presence.shape}")
        if e_present is not None:
            print(f"e_present shape: {e_present.shape}")
        if exp_grams is not None:
            print(f"exp_grams shape: {exp_grams.shape}")

        # Create larger figure for comprehensive dashboard
        fig = plt.figure(figsize=(24, 18))
        gs = gridspec.GridSpec(4, 3, figure=fig, wspace=0.3, hspace=0.4)

        # Row 1: Presence metrics
        print("Creating presence metrics plots...")
        ax00 = fig.add_subplot(gs[0,0])
        self.plot_presence_reliability(p_presence, y_true_bin, ax=ax00)

        ax01 = fig.add_subplot(gs[0,1])
        self.plot_presence_pr_micro(p_presence, y_true_bin, ax=ax01)

        ax02 = fig.add_subplot(gs[0,2])
        self.plot_presence_roc(p_presence, y_true_bin, thresholds=thresholds, ax=ax02)

        # Row 2: counts
        print("Creating count metrics plots for presence-only model...")
        # Counts scatter plot (true vs predicted counts)
        ax10 = fig.add_subplot(gs[1,0])
        true_ct = y_true_bin.sum(axis=1).astype(np.float32)
        # Use thresholded counts for predicted
        if thresholds is not None:
            if hasattr(thresholds, '__len__') and len(thresholds) > 1:
                # Multiple thresholds - use mean for binary predictions
                threshold = float(np.mean(thresholds))
            else:
                # Single threshold
                threshold = float(thresholds)
        else:
            threshold = 0.5
        pred_ct = self._compute_presence_counts(p_presence, threshold)
        lo = float(min(true_ct.min(), pred_ct.min()))
        hi = float(max(true_ct.max(), pred_ct.max()))
        ax10.scatter(true_ct, pred_ct, s=8, alpha=0.6)
        ax10.plot([lo, hi], [lo, hi], linestyle="--", c="0.7")
        ax10.set_xlabel("True foods/meal")
        ax10.set_ylabel("Predicted foods/meal (thresholded)")
        ax10.set_title(f"{self.meal} – Counts: true vs predicted")
        
        # K histogram
        ax11 = fig.add_subplot(gs[1,1])
        ax11.hist(true_ct, bins=30, alpha=0.5, label="true", density=True)
        ax11.hist(pred_ct, bins=30, alpha=0.5, label="pred", density=True)
        ax11.legend()
        ax11.set_xlabel("Items per meal")
        ax11.set_ylabel("Density")
        ax11.set_title(f"{self.meal} – K distributions")
        ax12 = fig.add_subplot(gs[1,2])
        if thresholds is not None and np.ndim(thresholds) > 0:
            self.plot_threshold_hist(thresholds, ax=ax12)
        else:
            ax12.axis("off")
            ax12.text(0.5, 0.5, "Scalar threshold", ha="center", va="center")
            ax12.set_title("Presence thresholds")

        # Row 3: cluster analysis (filled below)
        ax20 = fig.add_subplot(gs[2,0])
        ax21 = fig.add_subplot(gs[2,1])
        ax22 = fig.add_subplot(gs[2,2])

        # Row 4: Cluster analysis (if available)
        if C is not None:
            # derive predicted bin using thresholds for count views
            if thresholds is not None:
                if hasattr(thresholds, '__len__') and len(thresholds) > 1:
                    # Multiple thresholds - use mean for binary predictions
                    threshold = float(np.mean(thresholds))
                else:
                    # Single threshold
                    threshold = float(thresholds)
            else:
                threshold = 0.5
            y_pred_bin = (p_presence >= threshold).astype(int)
            
            # For presence-only, use row 3 for cluster analysis
            self.plot_cluster_caterpillar_count_bias(C, y_true_bin, y_pred_bin, ax=ax20)
            
            # Cluster heatmap
            cluster_df = compute_metrics_by_cluster(
                self.meal, y_true_grams=y_true_grams,
                p_presence=p_presence, e_present=None, exp_grams=None, C=C,
                eps=10 # presence threshold
            )
            self.plot_cluster_heatmap_in_dashboard(cluster_df, ax=ax21)
            
            # Cluster metrics summary
            ax22.text(0.1, 0.9, "Cluster Count Bias Summary", transform=ax22.transAxes,
                     fontsize=12, fontweight='bold')
            if cluster_df is not None and len(cluster_df) > 0:
                # Show key metrics
                metrics_text = f"Clusters: {len(cluster_df)}\n"
                if 'count_mae' in cluster_df.columns:
                    metrics_text += f"Avg Count MAE: {cluster_df['count_mae'].mean():.2f}\n"
                if 'count_bias' in cluster_df.columns:
                    metrics_text += f"Avg Count Bias: {cluster_df['count_bias'].mean():.2f}\n"
                if 'f1_micro' in cluster_df.columns:
                    metrics_text += f"Avg F1 Micro: {cluster_df['f1_micro'].mean():.3f}\n"
                if 'auroc_presence' in cluster_df.columns:
                    metrics_text += f"Avg AUROC: {cluster_df['auroc_presence'].mean():.3f}\n"
                ax22.text(0.1, 0.7, metrics_text, transform=ax22.transAxes, 
                         fontsize=10, verticalalignment='top')
            else:
                ax22.text(0.1, 0.7, "No cluster data available", transform=ax22.transAxes, 
                         fontsize=10, verticalalignment='top')
            ax22.set_xlim(0, 1)
            ax22.set_ylim(0, 1)
            ax22.axis('off')
            
            # Row 4: Additional presence-only plots
            ax30 = fig.add_subplot(gs[3,0])
            self.plot_cluster_beeswarm_count_error(C, y_true_bin, y_pred_bin, top_k=12, ax=ax30)
            
            # Count bias plot
            ax31 = fig.add_subplot(gs[3,1])
            true_counts = y_true_bin.sum(axis=1)
            pred_counts = self._compute_presence_counts(p_presence, threshold)
            bias = pred_counts - true_counts
            ax31.hist(bias, bins=30, alpha=0.7, edgecolor='black')
            ax31.axvline(0, color='red', linestyle='--', label='Perfect')
            ax31.set_xlabel('Count Bias (Pred - True)')
            ax31.set_ylabel('Frequency')
            ax31.set_title('Count Bias Distribution')
            ax31.legend()
            
            # Calibration plot
            ax32 = fig.add_subplot(gs[3,2])
            # Show calibration quality
            from sklearn.metrics import brier_score_loss
            brier = brier_score_loss(y_true_bin.ravel(), p_presence.ravel())
            ax32.text(0.1, 0.9, "Calibration Quality", transform=ax32.transAxes,
                     fontsize=12, fontweight='bold')
            ax32.text(0.1, 0.7, f"Brier Score: {brier:.4f}\n(Lower = Better)", 
                     transform=ax32.transAxes, fontsize=10, verticalalignment='top')
            if metrics and 'ece' in metrics:
                ax32.text(0.1, 0.5, f"ECE: {metrics['ece']:.4f}\n(Lower = Better)", 
                         transform=ax32.transAxes, fontsize=10, verticalalignment='top')
            ax32.set_xlim(0, 1)
            ax32.set_ylim(0, 1)
            ax32.axis('off')

        else:
            # No cluster data - fill with empty plots
            ax22 = fig.add_subplot(gs[2,2])
            ax22.axis("off")
            ax22.text(0.5, 0.5, "No cluster data", ha="center", va="center")
            
            ax30 = fig.add_subplot(gs[3,0])
            ax30.axis("off")
            ax30.text(0.5, 0.5, "No cluster data", ha="center", va="center")
            
            ax31 = fig.add_subplot(gs[3,1])
            ax31.axis("off")
            ax31.text(0.5, 0.5, "No cluster data", ha="center", va="center")
            ax31.set_title("Cluster Heatmap")
            
            ax32 = fig.add_subplot(gs[3,2])
            ax32.axis("off")
            ax32.text(0.5, 0.5, "No cluster data", ha="center", va="center")
            ax32.set_title("Calibration")
        fig.suptitle(f"{self.meal} — Presence-Only CVAE Evaluation Dashboard", y=0.98, fontsize=14, fontweight='bold')
        if save_as is None:
            save_as = os.path.join(self.out_dir, f"{self.meal}_comprehensive_panel.png")
        print(f"Saving dashboard to: {save_as}")
        self._savefig(fig, save_as, dpi=180)
        print(f"Dashboard creation completed for {self.meal}")


def _print_pretty_table(df):
    """Print DataFrame as a formatted PrettyTable"""
    try:
        from prettytable import PrettyTable
        
        # Create table
        table = PrettyTable()
        
        # Add columns with proper formatting
        for col in df.columns:
            if col == "Meal":
                table.add_column(col, df[col].tolist())
            else:
                # Format numeric columns to 3 decimal places
                formatted_values = []
                for val in df[col].tolist():
                    if pd.isna(val):
                        formatted_values.append("N/A")
                    elif isinstance(val, (int, float)):
                        formatted_values.append(f"{val:.3f}")
                    else:
                        formatted_values.append(str(val))
                table.add_column(col, formatted_values)
        
        # Set table style
        table.align = "r"  # Right align for numbers
        table.align["Meal"] = "l"  # Left align for meal names
        
        # Print the table
        print(table)
        
    except ImportError:
        # Fallback to pandas display if PrettyTable not available
        print("PrettyTable not available, using pandas display...")
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(df.to_string(index=False))
    except Exception as e:
        print(f"Error creating PrettyTable: {e}")
        # Fallback to pandas display
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(df.to_string(index=False))

def setup_logging(stage: str = "cvae"):
    """Root logger writing to logs/<stage>/<stage>_<timestamp>.log and to stderr."""
    log_dir = os.path.join("logs", stage)
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{stage}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def set_global_seed(seed: int):
    """Seed Python's random, NumPy's global RNG and TensorFlow; PYTHONHASHSEED only affects
    subprocesses."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # noqa: F401
        tf.random.set_seed(seed)  # type: ignore
    except Exception:
        pass
