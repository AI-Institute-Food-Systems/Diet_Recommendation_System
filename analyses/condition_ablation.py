#!/usr/bin/env python3
"""Condition ablation: retrain the presence-only multi-meal model without the cluster
condition, in two variants, and report presence and count metrics on a validation split
and a 10% hold-out:
  - meal_type_only: conditioned on meal type only (cluster input zeroed)
  - unconditional:  no cluster and no meal-type input

Data are the meals of all clusters in outputs/clustering/<meal>/ (no archetype filter).
Training uses 20 epochs with patience 5 (the defaults). A few minutes on a CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import tensorflow as tf  # noqa: E402

from cli.run_multi_meal_cvae import (_can_stratify, latent_diagnostics, load_all_meal_data,  # noqa: E402
                                     make_encoder_preproc_stats)
from src.core.runner import setup_logging  # noqa: E402
from src.utils.multi_meal_cvae import MultiMealCVAEConfig  # noqa: E402
from src.utils.multi_meal_presence_cvae import MultiMealPresenceOnlyCVAE  # noqa: E402

MEAL_TYPES = ["Breakfast", "Lunch", "Dinner"]
WEIGHTED_COLUMNS = [
    "f1_micro", "f1_macro", "precision_micro", "precision_macro", "recall_micro", "recall_macro",
    "auroc_presence", "auprc_presence", "brier_score", "ece", "r2_count", "mae_count", "rmse_count",
    "count_bias", "presence_threshold", "true_count_mean", "pred_count_mean", "active_units",
    "kl_ex", "kl_agg", "mi",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_raw_config(base_cfg: dict, outdir: Path) -> dict:
    cm = base_cfg.get("cvae_model", {})
    return {
        "cvae_model": {
            "presence_pos_weight": cm.get("presence_pos_weight", 10.0),
            "presence_eps": cm.get("presence_eps", 10.0),
            "kl_free_bits": 0.0,
            "kl_warmup_epochs": cm.get("kl_warmup_epochs", 30),
            "kl_annealing_epochs": cm.get("kl_annealing_epochs", cm.get("kl_warmup_epochs", 30)),
            "kl_weight": 0.01,
            "cluster_embedding_dim": 8,
            "meal_embedding_dim": 8,
            "focal_gamma": cm.get("focal_gamma", 0.0),
            "learning_rate": cm.get("learning_rate", 5e-4),
            "cond_dim": 256,
            "film_blocks": 3,
            "lambda_ac": 0.0,
            "guidance_drop_prob": 0.0,
            "meal_guidance_drop_prob": 0.0,
            "encoder_guidance_drop_prob": 0.0,
            "presence_pos_weight_max": 15,
            "ac_detach": False,
            "use_pair_prior": False,
            "use_pair_allowed_mask": False,
            "mask_strength": 0.0,
        },
        "monitor": {"enabled": False},
        "outdir": str(outdir),
    }


def variant_inputs(variant: str, M_real: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cluster ids, meal-type ids) fed to the model for a variant."""
    zeros = np.zeros(len(M_real), dtype=np.int32)
    if variant == "meal_type_only":
        return zeros, M_real.astype(np.int32)
    if variant == "unconditional":
        return zeros, zeros.copy()
    raise ValueError(f"Unknown variant: {variant}")


def evaluate_split(model, *, split_name: str, variant: str, X: np.ndarray, M_real: np.ndarray,
                   C_eval: np.ndarray, M_eval: np.ndarray, meal_type_mapping: dict[str, int]) -> list[dict]:
    rows: list[dict] = []
    for meal_name, meal_idx in meal_type_mapping.items():
        mask = M_real == meal_idx
        if not mask.any():
            continue
        X_meal, C_meal, M_meal = X[mask], C_eval[mask], M_eval[mask]
        row = {
            "variant": variant,
            "split": split_name,
            "Meal": meal_name,
            "n_samples": int(X_meal.shape[0]),
            "n_clusters_present_in_eval": int(np.unique(C_meal).size),
        }
        row.update(model.evaluate(X_meal, X_meal, C_meal, M_meal, presence_threshold="optimize"))
        row.update(latent_diagnostics(model, X_meal, C_meal, M_meal,
                                      name=f"{variant}_{split_name}_{meal_name.lower()}"))
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        weights = df["n_samples"].to_numpy(dtype=float)
        overall = {
            "variant": variant,
            "split": split_name,
            "Meal": "ALL_weighted",
            "n_samples": int(df["n_samples"].sum()),
            "n_clusters_present_in_eval": int(df["n_clusters_present_in_eval"].max()),
        }
        for col in WEIGHTED_COLUMNS:
            overall[col] = float(np.average(df[col].to_numpy(dtype=float), weights=weights))
        rows.append(overall)
    return rows


def run_variant(variant: str, *, X_tr, X_val, X_ho, M_tr_real, M_val_real, M_ho_real, base_cfg: dict,
                epochs: int, batch_size: int, patience: int, output_dir: Path,
                meal_type_mapping: dict[str, int], logger) -> list[dict]:
    print(f"\n=== Variant: {variant} ===")
    var_dir = output_dir / variant
    var_dir.mkdir(parents=True, exist_ok=True)

    C_tr, M_tr = variant_inputs(variant, M_tr_real)
    C_val, M_val = variant_inputs(variant, M_val_real)
    C_ho, M_ho = variant_inputs(variant, M_ho_real)

    config = MultiMealCVAEConfig({"cvae_model": base_cfg.get("cvae_model", {})})
    config.input_dim = X_tr.shape[1]
    config.num_clusters = 1
    config.meal_types = MEAL_TYPES

    raw_config = build_raw_config(base_cfg, var_dir)
    raw_config["encoder_preproc"] = make_encoder_preproc_stats(X_tr, method="zscore_log1p")

    model = MultiMealPresenceOnlyCVAE(config, raw_config)
    model.fit(X_tr, C_tr, M_tr, X_val, C_val, M_val, epochs=epochs, batch_size=batch_size,
              patience=patience, logger=logger)

    rows = evaluate_split(model, split_name="val_tiny", variant=variant, X=X_val, M_real=M_val_real,
                          C_eval=C_val, M_eval=M_val, meal_type_mapping=meal_type_mapping)
    rows += evaluate_split(model, split_name="holdout", variant=variant, X=X_ho, M_real=M_ho_real,
                           C_eval=C_ho, M_eval=M_ho, meal_type_mapping=meal_type_mapping)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="config/cvae_multi_meal.json")
    ap.add_argument("--out_dir", default="outputs/runs/eval_run/supplementary/condition_ablation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    set_seed(args.seed)
    logger = setup_logging("condition_ablation")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_json = json.loads(Path(args.config).read_text(encoding="utf-8"))

    X_all, C_all, M_all, _feature_names, meal_type_mapping, _pair_table, _orig_ids = load_all_meal_data(MEAL_TYPES)
    if X_all is None:
        raise RuntimeError("No cluster files found under outputs/clustering/ (run `make clustering`).")

    # Same splits as training: 10% hold-out, then 10% of the rest for validation.
    strat = C_all if _can_stratify(C_all, 0.10) else None
    X_dev, X_ho, C_dev, _C_ho, M_dev, M_ho = train_test_split(
        X_all, C_all, M_all, test_size=0.10, random_state=args.seed, stratify=strat)
    strat = C_dev if _can_stratify(C_dev, 0.10) else None
    X_tr, X_val, _C_tr, _C_val, M_tr_real, M_val_real = train_test_split(
        X_dev, C_dev, M_dev, test_size=0.10, random_state=args.seed, stratify=strat)

    rows: list[dict] = []
    for variant in ("unconditional", "meal_type_only"):
        rows += run_variant(variant, X_tr=X_tr, X_val=X_val, X_ho=X_ho, M_tr_real=M_tr_real,
                            M_val_real=M_val_real, M_ho_real=M_ho, base_cfg=cfg_json, epochs=args.epochs,
                            batch_size=args.batch_size, patience=args.patience, output_dir=out_dir,
                            meal_type_mapping=meal_type_mapping, logger=logger)

    pd.DataFrame(rows).to_csv(out_dir / "condition_ablation_summary.csv", index=False)
    meta = {
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "patience": args.patience,
        "meal_types": MEAL_TYPES, "total_samples": int(X_all.shape[0]), "total_features": int(X_all.shape[1]),
        "total_pairs_baseline": int(np.unique(C_all).size), "holdout_samples": int(X_ho.shape[0]),
        "val_tiny_samples": int(X_val.shape[0]), "train_samples": int(X_tr.shape[0]),
    }
    (out_dir / "condition_ablation_run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'condition_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
