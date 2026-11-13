# Supplementary analyses

Scripts behind the sensitivity, robustness, cost-model and ablation results in the article's
Supporting Information (S1). They read the outputs of the main pipeline, so run `make reproduce`
first. Results go to `outputs/runs/<RUN_NAME>/supplementary/<analysis>/`.

```bash
make supplementary        # all analyses below except the condition ablation
make condition-ablation   # retrains two models
```

`make supplementary` runs three groups, which can also be run on their own:

| Target | Analyses | Needs |
|---|---|---|
| `sup-clustering` | cluster stability, leakage-safe clustering check | `make clustering` |
| `sup-evaluation` | LOF sensitivity, meal energy split, portion ablation, nutrient-level summary, primary-outcome CIs | `make eval`, `make theta_sweep` |
| `sup-substitutions` | cost-model sensitivity, substitution coverage and outcomes | `make eval` |

| Result | Script |
|---|---|
| S1 Table 11, meal energy-split sensitivity | [`meal_split_sensitivity.py`](meal_split_sensitivity.py) |
| S1 Table 12 and §1.2.4, cluster stability | [`cluster_stability.py`](cluster_stability.py) |
| S1 §1.2.4, leakage-safe clustering check | [`leakage_safe_clustering_cv.py`](leakage_safe_clustering_cv.py) |
| S1 Table 13, nutrient-level summary | [`nutrient_level_summary.py`](nutrient_level_summary.py) |
| S1 Tables 14–15, restaurant overhead and grocery pricing | [`cost_sensitivity.py`](cost_sensitivity.py) |
| S1 Table 17, LOF contamination sensitivity | [`lof_outlier_sensitivity.py`](lof_outlier_sensitivity.py) |
| S1 Table 18, primary outcomes with 95% CIs | [`primary_outcome_cis.py`](primary_outcome_cis.py) |
| S1 Table 19, portion-optimizer ablation | [`portion_ablation.py`](portion_ablation.py) |
| Results and S1 §3.5, substitution coverage and outcomes | [`substitution_outcomes.py`](substitution_outcomes.py) |
| Condition ablation | [`condition_ablation.py`](condition_ablation.py) |

Every script has `--help`, takes its input and output folders as arguments and fixes its random
seeds, so a rerun gives the same files. Shared helpers (loading a run's exports, per-cluster
medians, the portion-only generator, seeding) are in [`_common.py`](_common.py).

## Clustering robustness

**Cluster stability** (`cluster_stability.py`). For each meal type, draws random subsamples of the
meals, re-runs HDBSCAN with the fitted baseline's settings (minimum cluster size, minimum samples,
selection ε), and compares the new labels with the full-data labels by the adjusted Rand index
(ARI), over all meals and over meals that are not noise in either labelling. `make sup-clustering`
uses 10 subsamples of 80% of meals for breakfast and 50% for lunch and dinner.
Output: `cluster_stability/<meal>/stability_replicates.csv` (one row per subsample) and
`stability_summary.csv` (median, mean and 2.5–97.5% range of the ARI).

**Fold-wise clustering check** (`leakage_safe_clustering_cv.py`). Re-fits the clustering within
cross-validation folds: in each fold, a z-score scaling and HDBSCAN are fitted on the training meals
only, the held-out meals are assigned with `hdbscan.approximate_predict`, and the result is compared
with the full-data labels by ARI. The input is the exported feature matrix
(`outputs/clustering/<meal>_feature_matrix_unscaled.csv`), which is already z-scored over all meals.
`make sup-clustering` uses all meals, in 3 folds for breakfast and 2 for lunch and dinner.
Output: `leakage_safe_clustering/<meal>/`.

## Evaluation robustness

These read `outputs/runs/<RUN_NAME>/data_exports/`, written by `make eval`: the real and generated
meals of each archetype (grams per food), their archetypes, the nutrient table and the RDI targets.
RDI deviation is the per-meal log2 deviation from the meal's share of daily targets
(`per_meal_deviation_from_rdi` in `cli/evaluate.py`), summarised as the median per archetype and
then as the median over archetypes of the percent reduction (generated vs real).

**LOF contamination** (`lof_outlier_sensitivity.py`). Fits a Local Outlier Factor (35 neighbours) on
the real meals' food presence, removes the most outlying 0.1%, 0.3%, 0.5% and 1% of meals, and
recomputes the median reduction in RDI deviation. Output: `outlier_sensitivity/`.

**Meal energy split** (`meal_split_sensitivity.py`). Re-portions the generated food sets with the
RDI-per-kcal optimiser for four breakfast/lunch/dinner energy splits (25/35/40, 30/35/35, 20/40/40,
33/33/34) and recomputes the median reduction. Up to 200 meals per archetype (random, fixed seed).
Output: `meal_split_sensitivity/`.

**Portion ablation** (`portion_ablation.py`). Separates the portion optimiser's contribution from the
model's food choices by comparing three versions of each archetype's meals: real meals as eaten, real
food sets re-portioned by the optimiser, and the generated meals. Reports the median per-archetype
reduction of the last two against the first, with a bootstrap CI over archetypes (1,000 resamples).
Up to 200 meals per archetype. Output: `ablation_portions/ablation_summary.csv` and
`ablation_by_cluster.csv`.

**Nutrient-level summary** (`nutrient_level_summary.py`). For each nutrient and meal type, the percent
reduction in the median deviation from the per-meal target, generated vs real, with a bootstrap CI
(300 resamples in `make sup-evaluation`). Sodium, total sugars and saturated fat are treated as
moderation nutrients: only intake above the target counts as deviation. Output:
`nutrient_level_summary/nutrient_level_summary.csv` and a compact `nutrient_spotlight_table.csv`.

**Primary outcomes with CIs** (`primary_outcome_cis.py`). Collects the 95% intervals reported for the
primary outcomes: the RDI-alignment reduction per meal type (from the portion ablation; the "Overall"
interval is the range of the three meal-type intervals) and, per number of substituted foods k, the
median nutrition gain and cost saving of the selected substitutions with 1,000-resample bootstrap
CIs. Substitutions are taken at θ = 1 for every k by default; `--theta knee` uses each k's selected θ
from the sweep. Output: `primary_outcomes_cis/`.

## Cost model and substitution coverage

`make sup-substitutions` first runs extra substitution analyses with the substitution script's
default search (nearest real meals of similar energy and item count, across archetypes, θ = 0.5),
then summarises them:

- `cost_sensitivity/restaurant_overhead0/`: one-food substitutions priced with the portion-based
  restaurant model and no overhead;
- `cost_sensitivity/grocery_k{1,2,3}/`: substitutions of 1, 2 and 3 foods priced from the per-100 g
  price table with the substitution script's food-category multipliers (no portion model, no
  overhead);
- `substitution_outcomes/matches/`: substitutions of up to three foods for every generated meal.

**Cost sensitivity** (`cost_sensitivity.py`). Overhead table: the median cost change when a flat
overhead of 0–3 USD is added to every meal, (generated − real) / (real + overhead), with a bootstrap
CI over meals (500 resamples). Grocery table: the median saving of the selected substitutions for
k = 1, 2, 3 under per-100 g pricing (1,000 resamples). Output: `cost_sensitivity/table_S_COST1_*.csv`
and `table_S_COST2_*.csv`.

**Substitution coverage and outcomes** (`substitution_outcomes.py`). Assigns each generated meal to the
lowest k at which it has a selected substitution and reports how many meals have one at each k, the
share of the evaluation cohort with none, and per k and meal type the median and interquartile range
of the nutrition gain and the cost change, the share of substitutions that cost more, and the share
that are worse on nutrition or cost. Output: `substitution_outcomes/substitution_counts.csv` and
`substitution_outcomes.csv`.

## Condition ablation

**`condition_ablation.py`** retrains the presence-only multi-meal model without the archetype
condition, in two variants (meal type only; no condition at all), with a 10% hold-out and a 10%
validation split stratified by cluster, and reports presence and count metrics (F1, precision, recall,
AUROC, AUPRC, Brier score, calibration, count errors) and latent-space diagnostics per meal type.
It trains on every cluster in `outputs/clustering/`, for 20 epochs with patience 5 (the defaults).
Output: `condition_ablation/condition_ablation_summary.csv`.
