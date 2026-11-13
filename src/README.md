# Library code

The modules behind the pipeline stages. The entry points in [`cli/`](../cli/),
[`experiments/`](../experiments/), [`analyses/`](../analyses/) and `src/preprocessing/` import them;
run everything from the repository root (the CLI scripts put `src/` on the import path, so some
modules import each other as `utils.*` and others as `src.*`).

| Package | Stage | Contents |
|---|---|---|
| [`preprocessing/`](preprocessing/) | `make preprocess` | NHANES recalls to meal × food matrices |
| [`evaluation/`](evaluation/) | `make preprocess` | Sparsity cut-off selection and its diagnostic plots |
| [`utils/`](utils/) | `make clustering`, `make cvae_train`, all | Clustering engine and figures, the CVAE model, shared helpers |
| [`core/`](core/) | `make cvae_train`, `make eval` | Meal generator and portion optimiser, training helpers, nutrient harmonisation |
| [`pipeline_eval/`](pipeline_eval/) | `make eval` | Evaluation steps and the Figure 3 plots |
| [`analysis_rdi_eval/`](analysis_rdi_eval/) | `make eval` | Nutrition metrics and bootstrap statistics |

## `preprocessing/`

**`data_preprocessing.py`** (run directly by `make preprocess`). Turns the NHANES individual-foods
recall files (`data/raw/meals/*.xpt`) into one meal × food gram matrix per meal type:

1. `standardize_food_codes` reads the recalls, replaces discontinued FNDDS food codes with their
   successors and splits each respondent's day into breakfast, lunch and dinner by eating occasion.
2. `build_systematic_prototype_mapping` maps similar foods to nutrient-based prototype foods,
   chosen per food subcategory so that most of the consumed mass is covered with a small
   nutrient error (`data/mappings/food_to_prototype.pkl`).
3. Per meal type: Local Outlier Factor removes 0.3% of meals, a bootstrap test on food frequency
   removes foods, and a sparsity cut-off is computed for diagnostics.

A step whose output file exists is loaded instead of recomputed. The counts after every step are
written to `logs/preprocessing/preprocessing.log`; the outputs are described in
[data/README.md](../data/README.md#preprocessing-outputs).

## `evaluation/`

Called by `data_preprocessing.py` only.

- **`select_sparsity.py`**: `choose_sparsity_threshold()` scores a grid of food-prevalence cut-offs
  (foods and meals kept, median foods per meal, retained Gini utility Σ 2p(1 − p)) and picks the knee
  of the utility-loss curve among the cut-offs that pass its guards.
- **`plot_sparsity_dashboard.py`**: the diagnostic plots of that choice, written to
  `data/processed/real_meals/diagnostics/`.

## `utils/`

- **`clustering_engine.py`**: `MealClusterEngine` builds 84 features per meal (nutrient amounts and
  ratios, counts of foods per WWEIA category, composition and diversity scores, log transforms),
  z-scores them, runs HDBSCAN per meal type, tests each cluster's features against the other meals
  (with Benjamini–Hochberg FDR control), and exports one gram matrix per cluster for training
  (`outputs/clustering/<meal>/cluster_<id>_training_data.pkl`). Used by `cli/run_clustering.py`.
- **`clustering_viz.py`**: `MealClusterViz` draws the per-meal-type cluster landscapes and dot
  matrices and Figure 2 (`create_combined_tsne_plot`): t-SNE maps (perplexity 64, seed 42; cached in
  `outputs/clustering/_tsne_cache/`) with the 34 archetypes in colour, a sample of the other meals
  in grey, and a food-category donut per archetype.
- **`multi_meal_presence_cvae.py`**: `MultiMealPresenceOnlyCVAE`, the generative model. An encoder
  and a decoder conditioned on meal type and archetype through FiLM layers, with condition dropout
  (`CondDrop`), a fixed per-pair prior and a mask of allowed foods. `fit()` trains it, a temperature
  and logit offsets are calibrated on the validation split, `save()`/`load()` write and read
  `presence_only_model/`, and `diversify_top_n` selects the items of sample meals.
- **`multi_meal_cvae.py`**: `MultiMealCVAEConfig` (reads the `cvae_model` section of
  `config/cvae_multi_meal.json`) and `Log1pZScore`, the encoder's input layer.
- **`diversity_monitor.py`**: `DiversityFidelityMonitor`, a Keras callback that every few epochs
  samples meals for each (meal type, archetype) pair and compares them with the validation meals
  (divergence, coverage, self-similarity), writing `diversity_monitor.csv`.
- **`cluster_overview.py`**: Figure 3A. `plot_cluster_overview()` draws one point per archetype
  (similarity of generated to real meals, share of generated meals with two or more foods, number
  of real meals); training plots it by meal type. `draw_figure3a()`, called by `make eval`, colours
  each archetype by its change in median RDI deviation, generated vs real, from the Figure 3C
  statistics.
- **`meal_nutrition.py`**: `calculate_meal_nutrition_from_df()`, a meal's nutrients from its grams
  (used for the clustering features).
- **`helpers.py`**: the `Meal` record, pickle helpers, file loggers (every stage logs through
  `setup_logger`), cached downloads, and the loaders of the processed meal tables and the nutrient
  table.
- **`simple_portion_assigner.py`**: `wweia_group_and_subgroup()`, a rule-based food-group classifier
  used to recognise beverages when sample meals are written.

## `core/`

- **`generator.py`**: `Generator` turns the trained model into portioned meals. `generate()` draws
  latent vectors for each requested (meal type, archetype) pair from one seeded random generator,
  decodes calibrated, masked presence probabilities, selects the foods and portions them with
  `_portion_rdi_per_kcal()`. The portion optimiser starts from typical serving sizes, moves each
  food's grams toward the per-kcal RDI targets, and rescales the meal to its energy target (25%,
  35% or 40% of daily energy) under per-item, food-group, beverage and total-gram caps.
  `Generator(model=None)` is used by `analyses/` to re-portion existing food sets.
- **`runner.py`**: helpers for training: per-cluster metrics (`compute_metrics_by_cluster`), the
  per-meal dashboard figure (`CVAEPanelVisualizer`), seeding and logging.
- **`nutrition_harmonizer.py`**: `load_harmonized_nutrition()` reads the FNDDS nutrient table and
  the Healthy US-Style pattern (2,000 kcal), harmonises nutrient names and units, and returns the
  food × nutrient matrix and the daily RDI vector used by evaluation.

## `pipeline_eval/`

Used by `cli/evaluate.py` (`make eval`).

- **`evaluation.py`**: `EvaluationManager` bootstraps per-archetype nutrient comparisons of real and
  generated meals (printed to the evaluation log) and runs `scripts/print_nutrient_profiles.py` as a
  sanity check.
- **`visualization.py`**: `VisualizationManager` draws Figure 3C (per-archetype RDI deviation of
  real and generated meals, with Mann–Whitney U tests and FDR-corrected significance marks; values
  in `boxplots_rdi_deviation_real_vs_generated_stats.csv`) and Figure 3B (spider plots of median
  nutrition scores per meal type), plus a combined preview figure.

## `analysis_rdi_eval/`

Import the submodules directly; the package `__init__` is kept empty.

- **`nutrition_metrics.py`**: nutrients from grams, energy, macronutrient shares and AMDR
  compliance, NAR and MAR, mean excess ratio, food-group (Hill) and Shannon diversity, per-meal
  coverage; `compute_nutrition_metrics_comprehensive` bundles them for Figure 3B.
- **`stats_bootstrap.py`**: Cliff's delta, bootstrap medians, Benjamini–Hochberg, and the
  per-archetype bootstrap comparison that `pipeline_eval/evaluation.py` prints.
- **`io_utils.py`**: `set_random_seeds` and `validate_data_contracts` (shape checks on the meals,
  nutrient matrix, nutrient names and RDI).
