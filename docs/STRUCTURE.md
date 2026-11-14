# Repository and output layout

## Repository

| Path | Contents |
|---|---|
| [`cli/`](../cli/) | Entry points for clustering, CVAE training and evaluation, called by the Makefile |
| [`src/preprocessing/`](../src/preprocessing/) | `data_preprocessing.py`: builds meals from the NHANES recall files (`make preprocess`) |
| [`src/evaluation/`](../src/evaluation/) | Sparsity-selection helpers used by preprocessing |
| [`src/utils/`](../src/utils/) | Clustering engine and figures, CVAE models, meal nutrition, shared helpers |
| [`src/core/`](../src/core/) | Meal generator (decoding and RDI-per-kcal portioning), training helpers, nutrient harmonisation |
| [`src/pipeline_eval/`](../src/pipeline_eval/) | Evaluation and visualisation steps used by `cli/evaluate.py` |
| [`src/analysis_rdi_eval/`](../src/analysis_rdi_eval/) | Nutrition metrics, bootstrap statistics and Figure 3 plotting |
| [`experiments/`](../experiments/) | Food-substitution analysis and Figure 4; CVAE vs GPT comparison (`gpt_comparison/`) |
| [`analyses/`](../analyses/) | Supplementary analyses (S1 Table 6, Tables 11–19) |
| [`scripts/`](../scripts/) | Headline-numbers log, figure export, environment and status checks |
| [`config/`](../config/) | Model configuration, the 34 archetypes, cluster names, figure registry |
| [`data/`](../data/) | Input data: NHANES recall files, FNDDS tables, food prices |
| [`tests/`](../tests/) | Checks run by `make ci` and GitHub Actions |
| [`docs/`](.) | This documentation |

Each folder has a README describing its files. See [src/README.md](../src/README.md) for the library code.

## Outputs

Everything the pipeline writes is git-ignored. Run outputs are grouped by `RUN_NAME` (default
`eval_run`), so several runs can live side by side; preprocessing and clustering outputs are
shared between runs.

```text
data/processed/real_meals/dataframes/       make preprocess: meal × food matrices after each step
data/mappings/food_to_prototype.pkl         make preprocess: food-to-prototype map

outputs/
├── clustering/                             make clustering
│   ├── <meal>/cluster_<id>_training_data.pkl   meals of each cluster (gram amounts), read by training
│   ├── cached_clustering.pkl               cluster labels and fitted HDBSCAN models per meal type
│   ├── cached_features_hybrid_zscore.pkl   clustering features (z-scored)
│   ├── <meal>_feature_matrix_unscaled.csv  the same z-scored features as CSV (despite the name)
│   ├── feature_contribution_report.csv     per-cluster feature effect sizes (S1 Table 6)
│   ├── <meal>_cluster_diagnosis_report.txt
│   ├── combined_tsne_landscape.svg/.png    Figure 2; per-meal landscapes and dot matrices next to it
│   └── _tsne_cache/                        t-SNE coordinates, keyed by a hash of the input
├── runs/<RUN_NAME>/
│   ├── presence_only_model/                trained CVAE (weights, config, calibration), read by make eval
│   ├── model/                              intermediate export written during training
│   ├── cv/                                 5-fold cross-validation; summary_mean_std.csv is S1 Table 7
│   ├── multi_meal_results.csv              final model's metrics on the validation split and the 10% hold-out
│   ├── eval_panels/                        per-meal-type panels of those metrics (val_tiny/, holdout/)
│   ├── diversity_monitor.csv               diversity of sampled meals during training (plot: *_trajectories.png)
│   ├── pair_table.json                     (meal type, archetype) pairs the model is conditioned on
│   ├── sanity_samples/                     samples drawn after training; analysis/overview_real_vs_generated.csv is the table behind Figure 3A
│   ├── figures/                            Figure 3A (overview_real_vs_generated_rdi), 3B (spider_plots_comprehensive) and 3C (boxplots_rdi_deviation_*, with a stats CSV)
│   ├── tables/                             evaluated meals per archetype, cost and health tables
│   ├── data_exports/                       real and generated meals, nutrient matrix, RDI, names; input to the later stages
│   ├── substitutions_k3/
│   │   ├── tradeoff_theta_<θ>_k<k>/        substitution matches and statistics for one θ and k
│   │   ├── figures_theta<θ>_k<k>/          per-k category heatmaps (make substitutions)
│   │   ├── figures_theta_sweep/            knee θ per k, Figure 4B–D, per-θ summary CSVs (make theta_sweep)
│   │   └── figure4a/                       Figure 4A (make figure4a)
│   ├── gpt_comparison/<model>/             make gpt-comparison: CVAE and GPT meals (experiment_results.json), evaluation/
│   └── supplementary/                      make supplementary, make condition-ablation
├── manuscript/
│   ├── figures/figure_<n>_*/panels/        Figure 2–4 panels (make manuscript-figures)
│   └── supplement/figures/                 all other generated figures
└── final_results_log.txt                   headline numbers with their sources (make manuscript-numbers)

logs/<stage>/<stage>.log                    full log of each stage
```

`presence_only_model/` is the packaged model that `cli/evaluate.py` loads (`MODEL_DIR` in the
Makefile). `model/` is an intermediate Keras export made at the end of training, kept for debugging.

## Determinism

Every stage is seeded (`SEED=42` and fixed seeds inside the analyses), and inputs that come from
directory listings are sorted, so two runs with the same environment on the same machine write
identical files. A different CPU or different package versions can change floating-point results.
Preprocessing and clustering are the most robust stages; the trained model, and everything that
depends on it, is the most sensitive.

Figures are the exception: PNG files differ at the pixel level between matplotlib builds that
bundle different FreeType versions (for example a conda build and the pip wheel), and SVG files
carry a creation date and generated element ids, even when the data behind them are identical.
Compare the CSV, pickle and NumPy outputs to check that two runs agree.

## Troubleshooting

**`CUDA_ERROR_NO_DEVICE` / "no CUDA-capable device is detected".** TensorFlow probes for a GPU when
it starts. On CPU-only machines the message is harmless and the work continues on the CPU;
`cli/evaluate.py` hides GPUs on purpose (`CUDA_VISIBLE_DEVICES=-1`).

**A stage fails or is interrupted.** Read `logs/<stage>/<stage>.log`. Each stage reads the outputs of
the stages before it, so fix the cause and rerun from that stage on, for example
`make eval substitutions theta_sweep figure4a manuscript-figures manuscript-numbers`.

**Results differ from another machine.** Run `make verify-env` on both machines and compare
package versions. Different versions, BLAS libraries or CPUs can change floating-point results; the
trained model is where such differences are amplified.

**`make clustering` is slow.** Most of its time goes into HDBSCAN on about 40,000 meals per meal
type and into the t-SNE embeddings for Figure 2. The embeddings are cached in
`outputs/clustering/_tsne_cache/`; `CLEAR_CACHE=1` (the default) only clears the feature cache.
