# Configuration

Small JSON files that, together with the Makefile variables, fix every setting of a run.

| File | Used by | Purpose |
|---|---|---|
| [`cvae_multi_meal.json`](cvae_multi_meal.json) | `make cvae_train`, `make condition-ablation` | Model architecture, loss weights and training settings of the multi-meal CVAE |
| [`cluster_filter_default.json`](cluster_filter_default.json) | `make cvae_train`, `make eval` | The 34 meal archetypes the model is trained and evaluated on |
| [`cluster_names.json`](cluster_names.json) | `make clustering`, `make cvae_train`, `make eval` | Display names of the clusters in figures and tables |
| [`manuscript_figures.json`](manuscript_figures.json) | `make manuscript-figures` | Which generated files become the panels of Figures 2–4 |

## `cvae_multi_meal.json`

The pipeline reads two sections:

- **`cvae_model`**: the model and its training.
  - Architecture: `encoder_sizes`, `latent_dim`, `cond_dim` and `film_blocks` (FiLM conditioning on
    meal type and archetype), `cluster_embedding_dim`, `meal_embedding_dim`, `dropout_rate`.
  - Losses: `presence_eps` (grams above which a food counts as present), `presence_pos_weight`
    (and `presence_pos_weight_max`), `focal_gamma`, the KL schedule (`kl_weight`, `kl_warmup_epochs`,
    `kl_annealing_epochs`, `kl_free_bits`) and the auxiliary classifier weight `lambda_ac`.
  - Training: `epochs`, `batch_size`, `learning_rate`, `patience`, and the condition-dropout
    probabilities `guidance_drop_prob` and `meal_guidance_drop_prob`.
- **`monitor`**: the diversity monitor that tracks sample coverage during training
  (`runs/<RUN_NAME>/diversity_monitor.csv`).

The Makefile passes `--epochs 50 --batch_size 128`, and these command-line values take precedence
over the file. Some settings are also fixed in the training code itself;
`cli/run_multi_meal_cvae.py` and `src/utils/multi_meal_presence_cvae.py` show how each value is
used, and the trained model saves the configuration it ran with in
`runs/<RUN_NAME>/presence_only_model/multi_meal_presence_cvae_config.json`.

## `cluster_filter_default.json`

```json
{"Breakfast": [0, 6, 9, ...], "Lunch": [2, 3, 4, ...], "Dinner": [3, 4, 5, ...]}
```

The HDBSCAN cluster ids of the 34 archetypes (S1 Table 6): 12 breakfast, 12 lunch and 10 dinner.
The ids refer to the labels that `make clustering` assigns to the bundled data, which are the same
on every run. Clusters not listed (and noise, id −1) are excluded from training and evaluation.

## `cluster_names.json`

```json
{"breakfast": {"0": "Pizza Meals", "1": "Hearty Mixed Plates", ...}, "lunch": {...}, "dinner": {...}}
```

A short name for every cluster (not only the archetypes), keyed by lower-case meal type and cluster
id. Figures and tables fall back to "Cluster <id>" for ids without a name.

## `manuscript_figures.json`

`figures` lists, for each manuscript figure, its panels and the generated file each panel is copied
from (`{RUN_NAME}` is replaced by the run name). `supplement_globs` selects the other generated
figures copied to `outputs/manuscript/supplement/figures/`. `make verify-manuscript-figures` checks
that every listed panel exists.

## Makefile variables

Run settings live in the Makefile; `make status` prints the current values and any of them can be
overridden on the command line (`make eval RUN_NAME=test`). The defaults are the settings used for
the paper:

| Variable | Default | Role |
|---|---|---|
| `RUN_NAME` | `eval_run` | Names the run folder `outputs/runs/<RUN_NAME>/` |
| `SEED` | `42` | Seed for meal generation in `make eval` (the Figure 2 t-SNE uses a fixed seed of 42) |
| `DETERMINISTIC` | `1` | Deterministic decoding during evaluation |
| `FORCE_REGEN` | `1` | Generate meals on every `make eval`; `0` reuses `data_exports/` if present (to redraw figures from saved meals) |
| `GRAY_FRACTION` | `0.2` | Share of non-archetype meals drawn in grey in Figure 2 |
| `EPOCHS`, `BATCH_SIZE`, `CV_FOLDS` | `50`, `128`, `5` | Training length, batch size and cross-validation folds |
| `TOP_K`, `MIN_PROB`, `MAX_BEVERAGES` | `10`, `0.10`, `1` | Decoding and portioning settings of the evaluation |
| `K_NEAREST`, `SUB_THETA`, `SWEEP_THETAS` | `50`, `0.5`, `0 0.25 0.5 1 2 1000` | Substitution search and the θ values of the sweep |
| `REST_OVERHEAD_USD` | `2.0` | Flat overhead added to restaurant meal prices |
