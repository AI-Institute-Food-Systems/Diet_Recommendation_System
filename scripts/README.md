# Scripts

Helpers around the pipeline. Run from the repository root.

| Script | Make target | Purpose |
|---|---|---|
| [`generate_manuscript_numbers_log.py`](generate_manuscript_numbers_log.py) | `manuscript-numbers` | Writes `outputs/final_results_log.txt`: the headline numbers of the Results (cluster counts, RDI-deviation reductions, substitution knee points, energy density, MAR, vitamin C), each with the files it was computed from. |
| [`export_manuscript_figures.py`](export_manuscript_figures.py) | `manuscript-figures`, `verify-manuscript-figures` | Copies the Figure 2–4 panels into `outputs/manuscript/figures/` as listed in `config/manuscript_figures.json`; other generated figures go to `outputs/manuscript/supplement/figures/`. |
| [`check_env.py`](check_env.py) | `verify-env` | Compares installed package versions with `requirements.txt`. |
| [`check_pipeline_status.py`](check_pipeline_status.py) | `pipeline-status` | Lists which stage outputs exist for `RUN_NAME`. |
| [`print_nutrient_profiles.py`](print_nutrient_profiles.py) | (called by `cli/evaluate.py`) | Prints nutrient profiles of real and generated meals as a sanity check during evaluation. |
