SHELL := /bin/bash

PY ?= python3
VENV_DIR ?= .venv
VENV_PY := $(VENV_DIR)/bin/python
PROJECT_ROOT ?= $(CURDIR)
RUN_NAME ?= eval_run
OUTPUT_ROOT ?= $(PROJECT_ROOT)/outputs
RUNS_ROOT ?= $(OUTPUT_ROOT)/runs
OUT ?= $(RUNS_ROOT)/$(RUN_NAME)
CVAE_OUT ?= $(OUT)
MODEL_DIR ?= $(CVAE_OUT)/presence_only_model
SEED ?= 42
CLUSTER_OUT ?= $(OUTPUT_ROOT)/clustering
MANUSCRIPT_OUT ?= $(OUTPUT_ROOT)/manuscript

# Clustering
FEATURE_TYPE ?= hybrid
CLUSTER_METHOD ?= enhanced_hdbscan
# Share of non-archetype meals drawn in grey in Figure 2.
GRAY_FRACTION ?= 0.2
CLEAR_CACHE ?= 1

# CVAE training (the presence-only model, sampled without calibration)
CONFIG ?= $(PROJECT_ROOT)/config/cvae_multi_meal.json
CLUSTER_FILTER ?= $(PROJECT_ROOT)/config/cluster_filter_default.json
EPOCHS ?= 50
BATCH_SIZE ?= 128
SANITY_N ?= 1000
REP_THRESHOLD ?= 0.5
ANALYSIS_EPS ?= auto
SANITY_TEMP ?= 1.0
SANITY_TOPP ?= 1.0
DIVERGENCE ?= kl
INVOKE_ONLY ?= 0
CV_FOLDS ?= 5

# Evaluation
PRESENCE_TO_GRAMS ?= rdi_per_kcal
TOP_K ?= 10
MIN_PROB ?= 0.10
DETERMINISTIC ?= 1
PRICE_MAP ?= $(PROJECT_ROOT)/data/prices/price_map.csv
MAX_BEVERAGES ?= 1
PORTION_STRATEGY ?= rdi_per_kcal
# 1: generate meals on every `make eval`. 0: reuse <run>/data_exports if present (e.g. to redraw
# figures from saved meals). Keep 1 after retraining, so that the new model's meals are evaluated.
FORCE_REGEN ?= 1

# Substitutions
SUB_OUT ?= $(OUT)/substitutions_k3
SUB_THETA ?= 0.5
K_NEAREST ?= 50
REST_OVERHEAD_USD ?= 2.0
SWEEP_THETAS ?= 0 0.25 0.5 1 2 1000

# GPT comparison (optional; needs OPENAI_API_KEY and requirements-gpt.txt)
GPT_MODEL ?= gpt-4o
GPT_N ?= 100
GPT_BATCH ?= 10
GPT_OUT ?= $(OUT)/gpt_comparison/$(GPT_MODEL)
GPT_RESULTS ?= $(GPT_OUT)/experiment_results.json

ifeq ($(strip $(DETERMINISTIC)),1)
  EVAL_DET_FLAG := --deterministic
else
  EVAL_DET_FLAG :=
endif

ifeq ($(strip $(FORCE_REGEN)),1)
  EVAL_REGEN_FLAG := --force_regen
else
  EVAL_REGEN_FLAG :=
endif

ifeq ($(strip $(INVOKE_ONLY)),1)
  INVOKE_ONLY_FLAG := --invoke_only
else
  INVOKE_ONLY_FLAG :=
endif

ifeq ($(strip $(CLEAR_CACHE)),1)
  CLEAR_CACHE_FLAG := --clear-cache
else
  CLEAR_CACHE_FLAG :=
endif

# Substitution settings shared by `substitutions` and `theta_sweep`.
SUB_ARGS = --k_nearest $(K_NEAREST) --selection_mode tradeoff --tie_band_pct 0.10 \
  --omit_add_categories beverages,other --omit_remove_categories beverages,other \
  --penalize_add_categories mixed_dishes --gamma_penalize_add 4.0 \
  --use_restaurant_portion_pricing --restaurant_overhead_usd $(REST_OVERHEAD_USD) \
  --no_cost_increase
# Cross-category penalty for mixed dishes, set per k (shell snippet; expects $$k).
SUB_ALPHA_BETA = alpha=0.10; beta=0.5; \
  [ "$$k" = "2" ] && { alpha=0.20; beta=1.0; }; \
  [ "$$k" = "3" ] && { alpha=0.35; beta=3.0; };


.PHONY: help setup verify-env ci reproduce preprocess clustering cvae_train eval substitutions \
        theta_sweep figure4a manuscript-figures verify-manuscript-figures manuscript-numbers smoke \
        supplementary sup-clustering sup-evaluation sup-substitutions condition-ablation \
        gpt-comparison gpt-generate gpt-evaluate status pipeline-status clean clean-scratch clean-generated

help:
	@echo "make setup               create .venv and install the pinned requirements"
	@echo "make reproduce           run the full pipeline (about 3 h on 16 CPU cores)"
	@echo "   = preprocess clustering cvae_train eval substitutions theta_sweep figure4a"
	@echo "     manuscript-figures manuscript-numbers"
	@echo "make supplementary       S1 sensitivity, robustness and cost analyses (after reproduce)"
	@echo "make condition-ablation  retrain without cluster conditioning (a few minutes)"
	@echo "make smoke               quick train/eval/substitution check (needs clustering outputs)"
	@echo "make gpt-comparison      CVAE vs GPT meals (after eval; needs OPENAI_API_KEY, see .env.example)"
	@echo "make ci                  environment check and unit tests"
	@echo "make status              show the configured paths"

setup:
	$(PY) -m venv $(VENV_DIR)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt

verify-env:
	$(PY) scripts/check_env.py

ci: verify-env
	$(PY) -m unittest discover -v tests

# Raw NHANES files to manuscript figures and the headline-numbers log.
reproduce: preprocess clustering cvae_train eval substitutions theta_sweep figure4a manuscript-figures manuscript-numbers

preprocess:
	$(PY) $(PROJECT_ROOT)/src/preprocessing/data_preprocessing.py

clustering:
	$(PY) $(PROJECT_ROOT)/cli/run_clustering.py \
	  --out-dir $(CLUSTER_OUT) \
	  --feature-type $(FEATURE_TYPE) \
	  --normalize zscore \
	  --method $(CLUSTER_METHOD) \
	  $(CLEAR_CACHE_FLAG) \
	  --viz-out $(CLUSTER_OUT) \
	  --gray-fraction $(GRAY_FRACTION)

cvae_train:
	$(PY) $(PROJECT_ROOT)/cli/run_multi_meal_cvae.py \
	  --config $(CONFIG) \
	  --cluster_filter $(CLUSTER_FILTER) \
	  --epochs $(EPOCHS) \
	  --batch_size $(BATCH_SIZE) \
	  --outdir $(CVAE_OUT) \
	  --sanity_n $(SANITY_N) \
	  --rep_threshold $(REP_THRESHOLD) \
	  --analysis_presence_eps $(ANALYSIS_EPS) \
	  --sanity_temp $(SANITY_TEMP) \
	  --sanity_nucleus_p $(SANITY_TOPP) \
	  --divergence $(DIVERGENCE) \
	  --cv_folds $(CV_FOLDS) \
	  --no_cal $(INVOKE_ONLY_FLAG)

eval:
	$(PY) $(PROJECT_ROOT)/cli/evaluate.py \
	  --project_root $(PROJECT_ROOT) \
	  --model_dir $(MODEL_DIR) \
	  --out_dir $(OUT) \
	  --portion_strategy $(PORTION_STRATEGY) \
	  --max_beverages $(MAX_BEVERAGES) \
	  --presence_to_grams $(PRESENCE_TO_GRAMS) \
	  --top_k $(TOP_K) \
	  --min_prob $(MIN_PROB) \
	  $(EVAL_DET_FLAG) $(EVAL_REGEN_FLAG) \
	  --price_map $(PRICE_MAP) \
	  --seed $(SEED) \
	  --cluster_filter $(CLUSTER_FILTER)

# Substitutions at one theta (SUB_THETA) for k = 1, 2, 3, with per-k figures.
substitutions:
	mkdir -p $(SUB_OUT)
	set -e; tname=$$(echo $(SUB_THETA) | sed 's/\./p/g'); \
	for k in 1 2 3; do \
	  $(SUB_ALPHA_BETA) \
	  $(PY) -m experiments.analyze_meal_substitutions \
	    --data_dir $(OUT) \
	    --output_dir $(SUB_OUT)/tradeoff_theta_$${tname}_k$$k \
	    --require_k_substitution $$k --theta $(SUB_THETA) \
	    --mixed_cross_alpha $$alpha --mixed_cross_beta $$beta \
	    $(SUB_ARGS); \
	  $(PY) -m experiments.plot_substitution_figures \
	    --input_dir $(SUB_OUT)/tradeoff_theta_$${tname}_k$$k \
	    --out_dir $(SUB_OUT)/figures_theta$${tname}_k$$k \
	    --mode k_from_input \
	    --with_category_heatmap; \
	done

# Theta sweep, knee-point theta per k, and the Figure 4 panels.
theta_sweep:
	mkdir -p $(SUB_OUT)/figures_theta_sweep
	set -e; for t in $(SWEEP_THETAS); do \
	  tname=$$(echo $$t | sed 's/\./p/g'); \
	  for k in 1 2 3; do \
	    $(SUB_ALPHA_BETA) \
	    $(PY) -m experiments.analyze_meal_substitutions \
	      --data_dir $(OUT) \
	      --output_dir $(SUB_OUT)/tradeoff_theta_$${tname}_k$$k \
	      --require_k_substitution $$k --theta $$t \
	      --mixed_cross_alpha $$alpha --mixed_cross_beta $$beta \
	      $(SUB_ARGS); \
	  done; \
	done
	$(PY) -m experiments.plot_substitution_figures \
	  --theta_runs_base $(SUB_OUT) \
	  --input_dir $(SUB_OUT) \
	  --out_dir $(SUB_OUT)/figures_theta_sweep \
	  --mode optimal_theta --dump_debug

# Figure 4A: a worked 1-hop / 2-hop example from the k=1 and k=2 matches at their knee theta.
figure4a:
	$(PY) -m experiments.substitution_foldchange.plot_nutrient_fold_changes \
	  --sweep_dir $(SUB_OUT) \
	  --data_dir $(OUT) \
	  --outdir $(SUB_OUT)/figure4a

manuscript-figures:
	$(PY) $(PROJECT_ROOT)/scripts/export_manuscript_figures.py \
	  --repo-root $(PROJECT_ROOT) \
	  --run-name $(RUN_NAME) \
	  --manuscript-dir $(MANUSCRIPT_OUT) \
	  --clean

verify-manuscript-figures:
	$(PY) $(PROJECT_ROOT)/scripts/export_manuscript_figures.py \
	  --repo-root $(PROJECT_ROOT) \
	  --run-name $(RUN_NAME) \
	  --manuscript-dir $(MANUSCRIPT_OUT) \
	  --verify-only

# Writes outputs/final_results_log.txt from the run's tables, exports and theta sweep.
manuscript-numbers:
	$(PY) scripts/generate_manuscript_numbers_log.py --out-dir $(OUT)

# ---------------------------------------------------------------------------------------
# Supplementary analyses (see analyses/README.md). They read the main pipeline's outputs.
SUP_OUT ?= $(OUT)/supplementary
# Extra substitution runs for S1 Tables 14-15 and S1 section 3.5: the substitution script's
# defaults (nearest real meals across clusters, theta 0.5).
SUP_SUBST = $(PY) -m experiments.analyze_meal_substitutions --data_dir $(OUT) --bootstraps 1000

supplementary: sup-clustering sup-evaluation sup-substitutions

# S1 Table 6 (archetype table), S1 Table 12 and section 1.2.4 (subsample stability, leakage-safe check).
sup-clustering:
	$(PY) -m analyses.archetype_table --clustering_dir $(CLUSTER_OUT) \
	  --meals_dir $(PROJECT_ROOT)/data/processed/real_meals/dataframes \
	  --archetypes $(CLUSTER_FILTER) --names $(PROJECT_ROOT)/config/cluster_names.json \
	  --out_dir $(SUP_OUT)/archetype_table
	set -e; for m in breakfast lunch dinner; do \
	  frac=0.5; folds=2; [ "$$m" = "breakfast" ] && { frac=0.8; folds=3; }; \
	  $(PY) -m analyses.cluster_stability --meal_type $$m --B 10 --subsample_frac $$frac \
	    --features_cache $(CLUSTER_OUT)/cached_features_hybrid_zscore.pkl \
	    --clustering_cache $(CLUSTER_OUT)/cached_clustering.pkl \
	    --out_dir $(SUP_OUT)/cluster_stability/$$m; \
	  $(PY) -m analyses.leakage_safe_clustering_cv --meal_type $$m --folds $$folds --max_samples 0 \
	    --clustering_cache $(CLUSTER_OUT)/cached_clustering.pkl \
	    --unscaled_feature_csv $(CLUSTER_OUT)/$${m}_feature_matrix_unscaled.csv \
	    --out_dir $(SUP_OUT)/leakage_safe_clustering/$$m; \
	done

# S1 Tables 11, 13, 17, 18 and 19.
sup-evaluation:
	$(PY) -m analyses.lof_outlier_sensitivity --exports_dir $(OUT)/data_exports --out_dir $(SUP_OUT)/outlier_sensitivity
	$(PY) -m analyses.meal_split_sensitivity --exports_dir $(OUT)/data_exports --out_dir $(SUP_OUT)/meal_split_sensitivity
	$(PY) -m analyses.portion_ablation --exports_dir $(OUT)/data_exports --out_dir $(SUP_OUT)/ablation_portions
	$(PY) -m analyses.nutrient_level_summary --exports_dir $(OUT)/data_exports \
	  --out_dir $(SUP_OUT)/nutrient_level_summary --bootstraps 300
	$(PY) -m analyses.primary_outcome_cis --ablation_summary $(SUP_OUT)/ablation_portions/ablation_summary.csv \
	  --sweep_dir $(SUB_OUT) --out_dir $(SUP_OUT)/primary_outcomes_cis

# S1 Tables 14-15 (cost model) and the substitution coverage and outcomes (Results, S1 section 3.5).
sup-substitutions:
	mkdir -p $(SUP_OUT)/cost_sensitivity $(SUP_OUT)/substitution_outcomes
	$(SUP_SUBST) --output_dir $(SUP_OUT)/cost_sensitivity/restaurant_overhead0 --require_k_substitution 1 \
	  --use_restaurant_portion_pricing --restaurant_overhead_usd 0
	set -e; for k in 1 2 3; do \
	  $(SUP_SUBST) --output_dir $(SUP_OUT)/cost_sensitivity/grocery_k$$k --require_k_substitution $$k; \
	done
	$(PY) -m analyses.cost_sensitivity --runs_dir $(SUP_OUT)/cost_sensitivity --bootstraps 500
	$(SUP_SUBST) --output_dir $(SUP_OUT)/substitution_outcomes/matches --max_food_diff 3
	$(PY) -m analyses.substitution_outcomes \
	  --matches $(SUP_OUT)/substitution_outcomes/matches/substitution_matches.csv \
	  --cohort_counts $(OUT)/tables/cluster_counts_totals.csv --out_dir $(SUP_OUT)/substitution_outcomes

# Retrains the model without cluster conditioning (two variants, a few minutes on CPU).
condition-ablation:
	$(PY) -m analyses.condition_ablation --config $(CONFIG) --out_dir $(SUP_OUT)/condition_ablation

# CVAE vs GPT meals for the 34 archetypes (after `make eval`). gpt-generate calls the OpenAI API
# (OPENAI_API_KEY from the environment or .env; pip install -r requirements-gpt.txt);
# gpt-evaluate needs no API access and also works on a results file from elsewhere
# (make gpt-evaluate GPT_RESULTS=path/to/experiment_results.json).
gpt-comparison: gpt-generate gpt-evaluate

gpt-generate:
	$(PY) $(PROJECT_ROOT)/experiments/gpt_comparison/generate_meals.py \
	  --run_dir $(OUT) --model_dir $(MODEL_DIR) --out_dir $(GPT_OUT) \
	  --model $(GPT_MODEL) --n_per_archetype $(GPT_N) --batch_size $(GPT_BATCH) --seed $(SEED)

gpt-evaluate:
	$(PY) $(PROJECT_ROOT)/experiments/gpt_comparison/evaluate.py \
	  --results $(GPT_RESULTS) --run_dir $(OUT)

# Quick check of training, evaluation and substitutions on a separate run
# (outputs/runs/smoke). Needs the outputs of `make preprocess clustering`.
smoke:
	$(MAKE) cvae_train RUN_NAME=smoke EPOCHS=1 CV_FOLDS=1 SANITY_N=10
	$(MAKE) eval RUN_NAME=smoke
	$(MAKE) substitutions RUN_NAME=smoke K_NEAREST=10

status:
	@echo PROJECT_ROOT=$(PROJECT_ROOT)
	@echo RUN_NAME=$(RUN_NAME)
	@echo OUT=$(OUT)
	@echo CLUSTER_OUT=$(CLUSTER_OUT)
	@echo MANUSCRIPT_OUT=$(MANUSCRIPT_OUT)
	@echo MODEL_DIR=$(MODEL_DIR)
	@echo SEED=$(SEED)
	@echo FEATURE_TYPE=$(FEATURE_TYPE) CLUSTER_METHOD=$(CLUSTER_METHOD) GRAY_FRACTION=$(GRAY_FRACTION)
	@echo PRESENCE_TO_GRAMS=$(PRESENCE_TO_GRAMS) DETERMINISTIC=$(DETERMINISTIC) FORCE_REGEN=$(FORCE_REGEN)
	@echo CONFIG=$(CONFIG) CLUSTER_FILTER=$(CLUSTER_FILTER)

pipeline-status:
	RUN_NAME=$(RUN_NAME) $(PY) $(PROJECT_ROOT)/scripts/check_pipeline_status.py

# Remove logs, caches and the smoke run; keeps full-run outputs.
clean-scratch:
	rm -rf $(PROJECT_ROOT)/logs $(PROJECT_ROOT)/cache $(RUNS_ROOT)/smoke

# Remove every generated output under outputs/.
clean-generated: clean-scratch
	rm -rf $(OUTPUT_ROOT)

clean:
	rm -rf $(OUT)
