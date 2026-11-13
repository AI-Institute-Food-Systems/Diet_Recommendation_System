# Pipeline entry points

The three main stages after preprocessing. The [`Makefile`](../Makefile) calls them with the
settings used for the paper; run them from the repository root, because they read `data/`,
`config/` and `outputs/` relative to it. Each prints its progress and writes a log to
`logs/<stage>/<stage>.log`. `--help` lists every option.

## `run_clustering.py` (`make clustering`)

Builds meal-level features, clusters each meal type with HDBSCAN, tests which clusters are
distinctive, and draws Figure 2.

1. Reads `data/processed/real_meals/dataframes/<meal>_bci_filtered.pkl` (from `make preprocess`) and
   the nutrient table `data/mappings/food_to_nutrient.pkl`.
2. Computes 84 "hybrid" features per meal (nutrient amounts and ratios, counts of foods per WWEIA
   food group, composition and diversity measures, log transforms) and z-scores them.
3. Runs HDBSCAN per meal type (minimum cluster size 200, minimum samples 100, cluster-selection ε of
   0.10, 0.08 and 0.06 for breakfast, lunch and dinner) and tests each cluster's features against
   the other meals (effect sizes with FDR control).
4. Draws the t-SNE landscapes (perplexity 64, seed 42), the per-cluster insets and dot matrices, and
   the combined Figure 2.

Main options: `--out-dir` (outputs, default `outputs/clustering`), `--feature-type` (`hybrid`),
`--normalize` (`zscore`), `--method` (`enhanced_hdbscan`), `--min-cluster-size`/`--min-samples`,
`--clear-cache` (recompute features), `--gray-fraction` (share of non-archetype meals drawn in grey;
0.2 for the paper), `--no-visuals`.

Writes to `outputs/clustering/`: the meals of each cluster in `<meal>/cluster_<id>_training_data.pkl`
(read by training), cluster labels and fitted models (`cached_clustering.pkl`), the z-scored
features (`cached_features_hybrid_zscore.pkl`, and as CSV in `<meal>_feature_matrix_unscaled.csv`
despite its name), per-cluster statistics (`feature_contribution_report.csv`,
`<meal>_cluster_diagnosis_report.txt`) and the figures.

## `run_multi_meal_cvae.py` (`make cvae_train`)

Trains one presence-only conditional VAE across breakfast, lunch and dinner, conditioned on meal type
and archetype through FiLM layers.

1. Loads the cluster files from `outputs/clustering/<meal>/` (sorted by cluster id) and keeps the 34
   archetypes listed in `config/cluster_filter_default.json`.
2. Holds out 10% of meals (stratified by archetype), runs 5-fold cross-validation on the rest
   (S1 Table 7), then trains the final model.
3. Draws sanity samples for every archetype and compares them with real meals: per-archetype
   similarity, diversity and coverage go to `sanity_samples/analysis/overview_real_vs_generated.csv`,
   the table behind Figure 3A. The Makefile samples without presence calibration (`--no_cal`).

Main options: `--config` (`config/cvae_multi_meal.json`), `--cluster_filter`, `--epochs`,
`--batch_size`, `--cv_folds`, `--outdir`, `--sanity_n` (samples per archetype),
`--no_cal` (sample without calibration), `--invoke_only` (load a saved model and only sample).

Writes to `outputs/runs/<RUN_NAME>/`: `presence_only_model/` (the model read by evaluation),
`model/` (an intermediate export), `cv/`, `sanity_samples/`, `pair_table.json` and
`diversity_monitor.csv`.

## `evaluate.py` (`make eval`)

Generates one meal for each real meal of the 34 archetypes, portions it, and compares generated and
real meals (Figure 3).

1. Loads the model from `presence_only_model/` and the real meals of each archetype.
2. Decodes food-presence probabilities for the meal's archetype and meal type (deterministic with
   `--deterministic`), selects the foods, and assigns grams with the RDI-per-kcal portion optimiser
   (`src/core/generator.py`), so that each meal approaches its share of the daily targets (25%,
   35% and 40% of daily energy for breakfast, lunch and dinner).
3. Computes nutrient metrics (RDI deviation, MAR, AMDR, energy density, coverage of single nutrients)
   for real and generated meals, bootstraps per-archetype comparisons, and draws Figure 3B–C.
4. Draws Figure 3A from the training-stage table, colouring each archetype by its change in RDI
   deviation, generated vs real (`src/utils/cluster_overview.py`).

Main options: `--model_dir`, `--out_dir`, `--seed`, `--deterministic`, `--top_k`, `--min_prob`,
`--max_beverages`, `--presence_to_grams` / `--portion_strategy` (`rdi_per_kcal`), `--cluster_filter`,
`--price_map` (food prices for the cost tables), `--force_regen` (ignore saved exports; the Makefile
passes it unless `FORCE_REGEN=0`).

Writes to `outputs/runs/<RUN_NAME>/`: `data_exports/` (real and generated meals, archetypes, nutrient
matrix, RDI targets, food names and groups; read by the substitution analysis and the supplementary
analyses), `figures/` and `tables/`.
