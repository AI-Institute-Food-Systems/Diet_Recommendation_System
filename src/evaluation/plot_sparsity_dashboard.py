"""
Diagnostic plots for the sparsity cut-off chosen by evaluation/select_sparsity.py.

Called by src/preprocessing/data_preprocessing.py (`make preprocess`); writes
`<meal>_sparsity_dashboard_utility.png` and `<meal>_utility_vs_K_dashboard.png` to the
diagnostics folder (data/processed/real_meals/diagnostics/).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def utility_vs_topK_curve(df):
    """
    Utility retained vs number of foods kept (K), keeping the foods with the highest
    utility 2*p*(1-p) first. Returns (K, fraction of the total utility).
    """
    pres = (df > 0).astype('uint8')
    p = pres.mean(0).values
    u = 2*p*(1-p)  # Bernoulli utility
    order = u.argsort()[::-1]              # keep most-informative first
    cumU = u[order].cumsum()
    U0 = cumU[-1]
    K = np.arange(1, len(u)+1)
    return K, (cumU / U0)                  # utility retained vs #foods kept

def plot_utility_vs_K_dashboard(metrics_df: pd.DataFrame, chosen_pct: int, meal_type: str, outdir: str, bci_df: pd.DataFrame):
    """
    Utility retained (%) vs number of foods kept (K), computed on `bci_df`, with the chosen
    operating point (foods and utility retained at `chosen_pct`) as a red star.
    The 90/92/95 % lines and the shading above 90 % are fixed reference marks, not the
    guard values passed to choose_sparsity_threshold. Returns the PNG path.
    """
    _ensure_dir(outdir)
    M = metrics_df.sort_values('percentile').reset_index(drop=True).copy()

    # Get the smooth utility vs K curve
    K, utility_retained = utility_vs_topK_curve(bci_df)
    utility_retained_pct = 100.0 * utility_retained  # Convert to percentage

    # Locate chosen operating point
    if chosen_pct in M['percentile'].values:
        row = M.index[M['percentile'] == chosen_pct][0]
    else:
        # fallback: pick nearest
        row = int(np.argmin(np.abs(M['percentile'].values - chosen_pct)))
        chosen_pct = int(M.loc[row, 'percentile'])

    chosen_foods = int(M.loc[row, 'foods_retained'])
    chosen_util_pct = 100.0 * M.loc[row, 'utility_retain']

    # ---- Figure layout ----
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(f'{meal_type.capitalize()} Utility vs Foods Kept Dashboard', fontsize=16, y=0.95)

    # Main plot: Utility retained vs K
    ax_main = fig.add_axes([0.1, 0.15, 0.85, 0.75])
    
    # Plot the smooth utility curve
    ax_main.plot(K, utility_retained_pct, 'b-', linewidth=2, label='Utility Retained (%)')
    
    # Add guard lines (horizontal)
    guard_lines = [90, 92, 95]
    for guard in guard_lines:
        ax_main.axhline(y=guard, color='orange', linestyle=':', alpha=0.7, linewidth=1)
        ax_main.text(K[-1], guard, f'{guard}%', ha='right', va='bottom', fontsize=10, color='orange')
    
    # Shaded region for guard-passing K values (where utility >= 90%)
    passing_mask = utility_retained_pct >= 90
    if passing_mask.any():
        ax_main.fill_between(K, 90, 100, where=passing_mask, alpha=0.1, color='green', label='Guard Passing')
    
    # Mark chosen operating point
    ax_main.axvline(x=chosen_foods, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax_main.plot([chosen_foods], [chosen_util_pct], marker='*', markersize=20, color='red', 
                label=f'Chosen Point (K={chosen_foods}, {chosen_util_pct:.1f}%)')
    
    # Add annotation for chosen point
    ax_main.annotate(f'Chosen: {chosen_pct}% percentile\nK={chosen_foods} foods\n{chosen_util_pct:.1f}% utility',
                    xy=(chosen_foods, chosen_util_pct),
                    xytext=(20, 20), textcoords='offset points',
                    fontsize=10, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='red'))
    
    ax_main.set_xlabel('Number of Foods Kept (K)')
    ax_main.set_ylabel('Utility Retained (%)')
    ax_main.set_ylim(0, 100)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='lower right')
    ax_main.set_title('Utility Retained vs Number of Foods Kept (Smooth Curve)')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outpath = os.path.join(outdir, f'{meal_type}_utility_vs_K_dashboard.png')
    fig.savefig(outpath, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return outpath

def plot_sparsity_dashboard(metrics_df: pd.DataFrame, chosen_pct: int, meal_type: str, outdir: str):
    """
    Sparsity dashboard over the percentile grid:
      - Top panel: utility retained (%) and meal coverage (%) vs percentile
      - Bottom panel: foods kept vs percentile (step plot) to show plateaus
      - Vertical line and star mark the chosen operating point
      - The 90/92/95 % lines and the shading above 90 % are fixed reference marks,
        not the guard values passed to choose_sparsity_threshold

    Expected columns in metrics_df:
      ['percentile','foods_retained','meals_retained','median_nonzeros','utility','utility_retain']
    Returns the PNG path.
    """
    _ensure_dir(outdir)
    M = metrics_df.sort_values('percentile').reset_index(drop=True).copy()

    # Series we need
    percentiles = M['percentile'].astype(float).values
    foods = M['foods_retained'].astype(float).values
    util = 100.0 * M['utility_retain'].astype(float).values  # Convert to percentage
    meals = M['meals_retained'].astype(float).values
    meals0 = float(M.loc[0, 'meals_retained'])
    covg = 100.0 * (meals / max(1.0, meals0))  # Meal coverage percentage

    # Locate chosen operating point
    if chosen_pct in M['percentile'].values:
        row = M.index[M['percentile'] == chosen_pct][0]
    else:
        # fallback: pick nearest
        row = int(np.argmin(np.abs(M['percentile'].values - chosen_pct)))
        chosen_pct = int(M.loc[row, 'percentile'])

    chosen_util = util[row]
    chosen_foods = foods[row]
    chosen_covg = covg[row]

    # ---- Figure layout ----
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f'{meal_type.capitalize()} Sparsity Performance Dashboard', fontsize=16, y=0.95)

    # Top panel: Utility retained vs Percentile
    ax_top = fig.add_axes([0.1, 0.55, 0.85, 0.35])
    
    # Plot utility retained vs percentile
    ax_top.plot(percentiles, util, 'b-o', linewidth=2, markersize=6, label='Utility Retained (%)')
    ax_top.plot(percentiles, covg, 'g-o', linewidth=2, markersize=6, label='Meal Coverage (%)')
    
    # Add guard lines (horizontal)
    guard_lines = [90, 92, 95]
    for guard in guard_lines:
        ax_top.axhline(y=guard, color='orange', linestyle=':', alpha=0.7, linewidth=1)
        ax_top.text(percentiles[-1], guard, f'{guard}%', ha='left', va='bottom', fontsize=8, color='orange')
    
    # Shaded region for guard-passing percentiles (where utility >= 90%)
    passing_mask = util >= 90
    if passing_mask.any():
        ax_top.fill_between(percentiles, 90, 100, where=passing_mask, alpha=0.1, color='green', label='Guard Passing')
    
    # Mark chosen operating point
    ax_top.axvline(x=chosen_pct, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax_top.plot([chosen_pct], [chosen_util], marker='*', markersize=16, color='red', label=f'Chosen Point ({chosen_pct}%)')
    ax_top.plot([chosen_pct], [chosen_covg], marker='*', markersize=16, color='red')
    
    ax_top.set_xlabel('Sparsity Percentile')
    ax_top.set_ylabel('Percentage (%)')
    ax_top.set_ylim(0, 100)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc='lower left')
    ax_top.set_title('Utility Retained and Meal Coverage vs Sparsity Percentile')

    # Bottom panel: Foods kept vs Percentile (step plot)
    ax_bottom = fig.add_axes([0.1, 0.1, 0.85, 0.35])
    
    # Create step plot for foods retained
    ax_bottom.step(percentiles, foods, 'b-', linewidth=2, where='post', label='Foods Retained')
    ax_bottom.plot(percentiles, foods, 'bo', markersize=4)
    
    # Mark chosen operating point
    ax_bottom.axvline(x=chosen_pct, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax_bottom.plot([chosen_pct], [chosen_foods], marker='*', markersize=16, color='red', label=f'Chosen Point ({chosen_pct}%)')
    
    # Annotate breakpoints (where foods_retained changes)
    for i in range(1, len(foods)):
        if foods[i] != foods[i-1]:
            ax_bottom.annotate(f'{int(foods[i])}', 
                             xy=(percentiles[i], foods[i]), 
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8, ha='left', va='bottom',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax_bottom.set_xlabel('Sparsity Percentile')
    ax_bottom.set_ylabel('Number of Foods')
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc='upper right')
    ax_bottom.set_title('Foods Retained vs Sparsity Percentile (Step Plot)')
    
    # Add rug plot on top axis showing all tested percentiles
    ax_rug = ax_top.twiny()
    ax_rug.set_xlim(ax_top.get_xlim())
    ax_rug.set_xticks(percentiles)
    ax_rug.set_xticklabels([f'{int(p)}' for p in percentiles], rotation=45, fontsize=8)
    ax_rug.set_xlabel('All Tested Percentiles', fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outpath = os.path.join(outdir, f'{meal_type}_sparsity_dashboard_utility.png')
    fig.savefig(outpath, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return outpath
