#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Train the multi-meal presence-only CVAE (Makefile target `cvae_train`).

Loads the per-cluster training meals written by the clustering stage
(outputs/clustering/<meal>/cluster_<id>_training_data.pkl, relative to the working directory),
keeps the archetype clusters listed in --cluster_filter and encodes each (meal type, cluster)
pair as one condition. A 10% stratified holdout is set aside; with --cv_folds > 1 a stratified
K-fold cross-validation runs on the remaining development set. The final model is trained on the
development set minus a 10% tiny validation split, and evaluated on that split and on the
holdout. It then writes --sanity_n sample meals per pair and compares them with the real meals.

Writes under --outdir: pair_table.json, cv/, model/, presence_only_model/ (read by
cli/evaluate.py), eval_panels/ (evaluation panels), sanity_samples/ (samples and analysis/),
diversity_monitor.csv (and its _trajectories.png plot) and multi_meal_results.csv; the log goes
to logs/cvae_train/cvae_train.log.

Makefile call (defaults shown):
    python cli/run_multi_meal_cvae.py --config config/cvae_multi_meal.json \
        --cluster_filter config/cluster_filter_default.json --epochs 50 --batch_size 128 \
        --outdir outputs/runs/eval_run --sanity_n 1000 --rep_threshold 0.5 \
        --analysis_presence_eps auto --sanity_temp 1.0 --sanity_nucleus_p 1.0 \
        --divergence kl --cv_folds 5 --no_cal

With --invoke_only it skips training and samples from a saved model instead
(--model_dir, default <outdir>/presence_only_model).
"""

import os
import sys
import time
import pickle
import logging
import argparse
from datetime import datetime

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warnings, 3=errors
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '')  # Disable GPU warnings if no GPU
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', message='.*CUDA.*')
from pathlib import Path

# Ensure repository root is importable when running as a script
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List

from sklearn.model_selection import train_test_split

from src.utils.multi_meal_cvae import MultiMealCVAEConfig
from src.utils.multi_meal_presence_cvae import MultiMealPresenceOnlyCVAE
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker as mticker

# Shared evaluation and plotting helpers
from src.core.runner import (
    CVAEPanelVisualizer, compute_metrics_by_cluster,
    _print_pretty_table, set_global_seed
)
from src.utils.helpers import setup_logger as _setup_file_logger


# Environment defaults read by MultiMealPresenceOnlyCVAE.convert_probabilities_to_binary
PRESENCE_THRESH=0.5
SAMPLE_BERNOULLI=1

os.environ.setdefault("PRESENCE_THRESH", str(PRESENCE_THRESH))  # e.g., "0.5"
os.environ.setdefault("SAMPLE_BERNOULLI", str(SAMPLE_BERNOULLI))# "1" to sample, "0" to threshold

# Additional TensorFlow logging suppression
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# redirect stdout/stderr to a file logger similar to preprocessing
class _StreamToLogger:
    def __init__(self, logger: logging.Logger, level: int):
        self._logger = logger
        self._level = level

    def write(self, buf: str):
        for line in buf.rstrip().splitlines():
            self._logger.log(self._level, line.rstrip())

    def flush(self):
        pass

def eval_leakage(model, X, C, M):
    """
    Fraction of predicted probability mass that goes to disallowed foods,
    averaged over (meal,cluster) pairs present in the data.
    """
    import numpy as np

    # 1) Get per-sample probabilities
    P = model.predict_heads(X, C, M)  # shape [N,D]

    # 2) Pull the non-trainable gate (0 = allowed, 1 = disallowed)
    try:
        gate_tbl = model.decoder.get_layer("pair_allowed_gate").get_weights()[0]  # [(K+1), D]
    except Exception:
        return {"leakage_mean": np.nan, "note": "no gate layer present"}

    # 3) Aggregate by pair
    pairs = np.unique(C)
    leak_vals, pair_ns = [], []
    for k in pairs:
        idx = (C == k)
        if not np.any(idx): 
            continue
        G = gate_tbl[k]            # [D], 0/1
        Pk = P[idx].mean(axis=0)   # [D], mean probs for the pair
        total = Pk.sum() + 1e-9
        leakage = float((Pk * G).sum() / total)
        leak_vals.append(leakage); pair_ns.append(idx.sum())

    wmean = float(np.average(leak_vals, weights=pair_ns))
    return {"leakage_mean": wmean, "per_pair": leak_vals}


def eval_topk_alignment(model, X, C, M, K=10, eps=1e-6):
    """
    How close are the model's top foods per (meal,cluster) to the real frequent foods?
    """
    import numpy as np

    # Empirical prevalence per pair
    Xb = (X > eps).astype(np.float32)
    pairs = np.unique(C)
    prec_at_k, ndcg_at_k, pearson_r = [], [], []

    # Model mean probs per pair
    P = model.predict_heads(X, C, M)

    def dcg(rel):
        return np.sum((2**rel - 1) / np.log2(np.arange(2, len(rel)+2)))

    for k in pairs:
        idx = (C == k)
        if idx.sum() < 5: 
            continue

        prev = Xb[idx].mean(axis=0)        # empirical P(food|pair)
        pred = P[idx].mean(axis=0)         # model mean prob

        # Rankings
        top_emp = np.argsort(prev)[-K:]
        top_pred = np.argsort(pred)[-K:]

        # Precision@K (set overlap)
        prec = len(set(top_emp) & set(top_pred)) / float(K)
        prec_at_k.append(prec)

        # nDCG@K using empirical prevalence as relevance
        rel_at_pred = prev[top_pred]
        ideal_rel   = np.sort(prev)[-K:][::-1]
        ndcg = dcg(rel_at_pred) / (dcg(ideal_rel) + 1e-9)
        ndcg_at_k.append(ndcg)

        # Pearson across all items (linear correlation)
        if prev.std() > 0 and pred.std() > 0:
            p = np.corrcoef(prev, pred)[0,1]
        else:
            p = np.nan
        pearson_r.append(p)

    return {
        "P@{}".format(K): float(np.mean(prec_at_k)),
        "nDCG@{}".format(K): float(np.mean(ndcg_at_k)),
        "Pearson_r": float(np.nanmean(pearson_r)),
        "n_pairs": int(len(prec_at_k)),
    }


def eval_count_mae_per_pair(model, X, C, M, eps=1e-6):
    """
    Check per-(meal,cluster) count MAE for calibration within pairs.
    """
    import numpy as np
    Xb = (X > eps).astype(np.float32)
    P = model.predict_heads(X, C, M)
    pairs = np.unique(C)
    maes = []
    for k in pairs:
        idx = (C == k)
        if idx.sum() < 5: continue
        true_cnt = Xb[idx].sum(axis=1)
        pred_cnt = P[idx].sum(axis=1)
        maes.append(np.mean(np.abs(true_cnt - pred_cnt)))
    return {"count_mae_per_pair_mean": float(np.mean(maes)), "n_pairs": len(maes)}


def print_pair_evaluation_summary(results, config_name="Current"):
    """
    Print a summary of pair-based evaluations for easy comparison across configurations.
    """
    print(f"\n📋 Pair Evaluation Summary - {config_name}")
    print("=" * 60)
    
    for meal_type, result in results.items():
        pair_evals = result.get('pair_evaluations', {})
        print(f"\n{meal_type}:")
        
        # Leakage
        leakage = pair_evals.get('leakage', {})
        if 'leakage_mean' in leakage and not np.isnan(leakage['leakage_mean']):
            print(f"  Leakage: {leakage['leakage_mean']:.4f}")
        else:
            print(f"  Leakage: {leakage.get('note', 'N/A')}")
        
        # Alignment
        alignment = pair_evals.get('alignment', {})
        if 'P@10' in alignment:
            print(f"  P@10: {alignment['P@10']:.3f}  •  nDCG@10: {alignment['nDCG@10']:.3f}  •  Pearson: {alignment['Pearson_r']:.3f}")
        else:
            print(f"  Alignment: {alignment.get('error', 'N/A')}")
        
        # Count MAE per pair
        count_mae = pair_evals.get('count_mae_per_pair', {})
        if 'count_mae_per_pair_mean' in count_mae:
            print(f"  Count MAE per pair: {count_mae['count_mae_per_pair_mean']:.3f} (n_pairs={count_mae['n_pairs']})")
        else:
            print(f"  Count MAE per pair: {count_mae.get('error', 'N/A')}")


def latent_diagnostics(model, X, C, M=None, name="diag"):
    """Diagnose latent collapse by checking active units and mutual information."""
    import numpy as np

    # --- 1) get posterior stats robustly ---
    # Try different ways to get encoder outputs
    mu, logvar = None, None
    
    # Method 1: Direct encoder access (for presence-only CVAE)
    if hasattr(model, 'encoder') and callable(model.encoder):
        try:
            outputs = model.encoder.predict([X, C, M], batch_size=256, verbose=0)
            mu, logvar = outputs[0], outputs[1]
        except Exception as e:
            print(f"  Encoder direct access failed: {e}")
    
    # Method 2: Try encode methods
    if mu is None:
        for attr in ["encode", "encode_stats", "encode_return_stats"]:
            enc = getattr(model, attr, None)
            if callable(enc): 
                try:
                    if M is not None:
                        outputs = enc(X, C, M)
                    else:
                        outputs = enc(X, C)
                    mu, logvar = outputs[0], outputs[1]
                    break
                except Exception:
                    continue
    
    if mu is None or logvar is None:
        print(f"  [WARNING] Could not extract latent representations for {name}")
        return {"active_units": 0, "kl_ex": 0.0, "kl_agg": 0.0, "mi": 0.0}

    mu = np.asarray(mu); logvar = np.asarray(logvar)
    zdim = mu.shape[1]

    # --- 2) active units ---
    au = (mu.var(axis=0) > 1e-2).sum()

    # --- 3) I(x;z) approx ---
    Esig2 = np.exp(logvar).mean(axis=0)
    VarMu = mu.var(axis=0)
    mbar  = mu.mean(axis=0)
    # KL(q(z|x)||p(z))
    kl_per = 0.5 * (mu**2 + np.exp(logvar) - 1.0 - logvar)
    kl_ex  = kl_per.sum(axis=1).mean()
    # KL(q(z)||p(z)) with Gaussian approx
    Sig = Esig2 + VarMu
    kl_agg = 0.5 * np.sum(mbar**2 + Sig - 1.0 - np.log(Sig + 1e-12))
    mi = float(max(0.0, kl_ex - kl_agg))

    print(f"[{name}] z_dim={zdim}  active_units(τ=0.01)={int(au)}")
    print(f"[{name}] E KL(q(z|x)||p(z))={kl_ex:.4f}  KL(q(z)||p(z))={kl_agg:.4f}  I(x;z)≈{mi:.4f}")

    return {"active_units": int(au), "kl_ex": float(kl_ex), "kl_agg": float(kl_agg), "mi": mi}


def z_ablation_delta(model, X, C, M, batch=512):
    import numpy as np

    mu, _, _ = model.encoder.predict([X, C, M], batch_size=batch, verbose=0)

    # Prefer the model helper if available (handles both variants cleanly)
    if hasattr(model, "predict_heads_from_z"):
        p_mu, _ = model.predict_heads_from_z(mu, C, M, batch_size=batch)
        p_0,  _ = model.predict_heads_from_z(np.zeros_like(mu), C, M, batch_size=batch)

        return float(np.mean(np.abs(p_mu - p_0)))

    # Fallback: call decoder and extract probabilities/logits carefully
    out_mu = model.decoder.predict([mu, C, M], batch_size=batch, verbose=0)
    out_0  = model.decoder.predict([np.zeros_like(mu), C, M], batch_size=batch, verbose=0)

    def to_prob(out):
        # presence-only: [p, logits, ac]; old decoders may return logits directly
        if isinstance(out, (list, tuple)):
            p = out[0]
            if p is not None:
                return p
            logits = out[1]
            return 1.0 / (1.0 + np.exp(-logits))
        # single tensor -> assume logits
        return 1.0 / (1.0 + np.exp(-out))

    p_mu = to_prob(out_mu)
    p_0  = to_prob(out_0)
    return float(np.mean(np.abs(p_mu - p_0)))


def make_encoder_preproc_stats(X_train: np.ndarray, method: str = "zscore_log1p"):
    """
    Returns a dict you can stash in the model config under 'encoder_preproc'.
    Computed **only on training** grams. Handles zero-variance safely.
    """
    if method == "zscore_log1p":
        Z = np.log1p(np.clip(X_train, 0.0, None).astype(np.float32))
        mu = Z.mean(axis=0)
        sd = Z.std(axis=0)
        sd[sd < 1e-8] = 1.0  # avoid div-by-zero; keeps column as 0 after centering
        return {"type": "zscore_log1p",
                "mean": mu.astype(np.float32).tolist(),
                "std":  sd.astype(np.float32).tolist()}
    elif method == "zscore":
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0); sd[sd < 1e-8] = 1.0
        return {"type": "zscore",
                "mean": mu.astype(np.float32).tolist(),
                "std":  sd.astype(np.float32).tolist()}
    elif method == "minmax":
        lo = X_train.min(axis=0)
        hi = X_train.max(axis=0)
        return {"type": "minmax",
                "min":  lo.astype(np.float32).tolist(),
                "max":  hi.astype(np.float32).tolist()}
    else:
        return {"type": "identity"}


def load_all_meal_data(meal_types, include_noise=False):
    """
    Load every meal type and align them on the union of food codes (sorted).
    Returns X_all, C_all (pair codes), M_all (meal index), feature names, meal_type_mapping,
    pair_table {pair code: (meal index, cluster_id)} and the original cluster ids per row.
    """
    print(f"Loading data for meal types: {meal_types}")
    per_meal = []  # list of dicts with X, C, M, cols, meal_type
    meal_type_mapping = {meal: idx for idx, meal in enumerate(meal_types)}

    # --- load each meal type and keep its real column names ---
    for meal_type in meal_types:
        print(f"\nLoading {meal_type}...")
        out = load_cluster_data(meal_type, include_noise)
        if out is None or out[0] is None:  # load_cluster_data returns a tuple of Nones when empty
            print(f"⚠ No data found for {meal_type}, skipping...")
            continue
        X_tr, y_tr, X_te, y_te, cols = out
        X_meal = np.vstack([X_tr, X_te])
        C_meal = np.hstack([y_tr, y_te])
        M_meal = np.full(len(X_meal), meal_type_mapping[meal_type], dtype=np.int32)
        per_meal.append({"X": X_meal, "C": C_meal, "M": M_meal, "cols": list(cols), "meal_type": meal_type})
        print(f"  Loaded {len(X_meal)} samples from {meal_type}")

    if not per_meal:
        print("❌ No data loaded from any meal type!")
        return None, None, None, None, None, None, None

    # --- build global union and align deterministically ---
    all_features = sorted({c for blk in per_meal for c in blk["cols"]})
    feat_to_idx = {f: i for i, f in enumerate(all_features)}

    aligned_blocks = []
    for blk in per_meal:
        X = blk["X"]; cols = blk["cols"]
        A = np.zeros((X.shape[0], len(all_features)), dtype=np.float32)
        for j, name in enumerate(cols):
            A[:, feat_to_idx[name]] = X[:, j]
        aligned_blocks.append({"X": A, "C": blk["C"], "M": blk["M"], "meal_type": blk["meal_type"]})
    
    X_all = np.vstack([b["X"] for b in aligned_blocks])
    C_all = np.hstack([b["C"] for b in aligned_blocks])
    M_all = np.hstack([b["M"] for b in aligned_blocks])

    print("\n✅ Combined dataset:")
    print(f"  Total samples: {len(X_all)}")
    print(f"  Features: {X_all.shape[1]}")
    print(f"  Unique clusters: {len(np.unique(C_all))}")
    print(f"  Meal types: {len(np.unique(M_all))}")

    # Recode each (meal, original cluster_id) pair as one condition id; keep the original ids
    orig_cluster_ids = np.hstack([b["C"] for b in aligned_blocks])  # original cluster_id from filenames

    pairs = np.stack([M_all, C_all], axis=1)
    uniq_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    C_all = inverse.astype(np.int32)
    pair_table = {int(i): (int(uniq_pairs[i,0]), int(uniq_pairs[i,1])) for i in range(len(uniq_pairs))}

    for b in aligned_blocks:
        idx = meal_type_mapping[b["meal_type"]]
        n = (M_all == idx).sum()
        print(f"  {b['meal_type']}: {n} samples")

    return X_all, C_all, M_all, all_features, meal_type_mapping, pair_table, orig_cluster_ids


def _parse_cluster_filter(meal_type_mapping: Dict[str,int], spec: Optional[str]):
    """
    Returns dict[int -> set[int]] mapping meal_idx -> allowed original cluster IDs.
    Accepts a file path to JSON or an inline JSON string.
    JSON format: {"Breakfast":[1,2], "Lunch":[3], "Dinner":[5,6]}
    """
    if not spec or spec.strip() == "":
        return None
    import json, os
    if os.path.exists(spec):
        with open(spec, "r") as f:
            data = json.load(f)
    else:
        data = json.loads(spec)
    filt = {}
    for meal_name, ids in data.items():
        if meal_name not in meal_type_mapping:
            raise ValueError(f"Unknown meal name in cluster_filter: {meal_name}")
        filt[meal_type_mapping[meal_name]] = set(int(x) for x in ids)
    return filt


def _apply_cluster_filter(X, C_pair, M, orig_cluster_ids, meal_filter: Dict[int,set]):
    """
    Keep only rows where (meal_idx -> original cluster_id) is allowed.
    Note: C_pair is the remapped pair code; we don't use it for filtering except to keep aligned shapes.
    """
    if not meal_filter:
        return X, C_pair, M, orig_cluster_ids
    mask = np.zeros(len(X), dtype=bool)
    # original cluster id is in orig_cluster_ids; M is meal_idx aligned to X row-wise
    for meal_idx, allowed in meal_filter.items():
        mask |= ((M == meal_idx) & np.isin(orig_cluster_ids, list(allowed)))
    # apply mask
    return X[mask], C_pair[mask], M[mask], orig_cluster_ids[mask]



def load_cluster_data(meal_type, include_noise=False):
    """Load one meal type's cluster pickles; returns a stratified 90/10 split (X_tr, y_tr, X_te, y_te, cols)."""
    cluster_dir = f"outputs/clustering/{meal_type.lower()}/"
    if not os.path.exists(cluster_dir):
        print(f"⚠ No cluster directory found for {meal_type}")
        return None, None, None, None, None

    # Sort by cluster id: os.listdir order is filesystem-dependent, and row order
    # feeds the stratified holdout split and CV folds, so it must be fixed.
    cluster_files = sorted(
        (f for f in os.listdir(cluster_dir)
         if f.startswith("cluster_") and f.endswith("_training_data.pkl")),
        key=lambda f: int(f.split("_")[1]),
    )
    if not cluster_files:
        print(f"No cluster data found for {meal_type}")
        return None, None, None, None, None

    print(f"Found {len(cluster_files)} clusters for {meal_type}")

    # Load each cluster frame
    frames, cids = [], []
    for fn in cluster_files:
        cid = int(fn.split("_")[1])
        if (not include_noise) and cid == -1:
            continue
        path = os.path.join(cluster_dir, fn)
        try:
            with open(path, "rb") as f:
                df = pickle.load(f)
            if not hasattr(df, "shape"):
                print(f"Unexpected data format in {fn}")
                continue

            # Canonicalize labels
            df = df.copy()
            df.columns = [_canon_item_label(c) for c in df.columns]

            # Handle duplicates
            if len(set(df.columns)) != len(df.columns):
                df = df.T.groupby(level=0).sum().T

            df["cluster_id"] = cid
            frames.append(df)
            cids.append(cid)
            print(f"  Loaded {fn}: {df.shape} (Cluster {cid})")

        except Exception as e:
            print(f"Error loading {fn}: {e}")

    if not frames:
        print(f"No valid data for {meal_type}")
        return None, None, None, None, None

    # Combine frames
    combined = pd.concat(frames, ignore_index=True)
    cols_final = [c for c in combined.columns if c != "cluster_id"]

    # Clean data
    feats = combined[cols_final].values
    neg_ct = int((feats < 0).sum())
    if neg_ct:
        print(f"⚠ Found {neg_ct:,} negative gram entries → clamping to 0.")
        feats = np.clip(feats, 0.0, None)

    # Remove all-zero columns
    nonzero_cols = (feats.sum(axis=0) > 0)
    zero_cols = ~nonzero_cols
    zero_col_count = int(zero_cols.sum())
    
    if zero_col_count > 0:
        print(f"⚠ Found {zero_col_count:,} all-zero columns → removing unused features.")
        feats = feats[:, nonzero_cols]
        cols_final = [cols_final[i] for i in range(len(cols_final)) if nonzero_cols[i]]

    # Build X, y
    X = feats.astype(np.float32)
    y = combined["cluster_id"].values.astype(np.int32)

    # Split train/test
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    return X_tr, y_tr, X_te, y_te, cols_final


def _canon_item_label(c) -> str:
    """Canonicalize item code as a string"""
    import re
    s = str(c).strip()
    if re.fullmatch(r"\d+(?:\.0+)?", s):
        return s.split(".")[0]
    return s


def train_multi_meal_model(
    X_tr, C_tr, M_tr,
    X_val, C_val, M_val,
    config, raw_config, logger, epochs, batch_size, patience, outdir=None,
):
    print("\n🚀 Training presence-only CVAE model...")
    print(f"  Train: {X_tr.shape[0]}  •  Val: {X_val.shape[0]}")

    # Train-only preproc
    preproc_stats = make_encoder_preproc_stats(X_tr, method="zscore_log1p")
    raw_config["encoder_preproc"] = preproc_stats

    # CRITICAL: Assert embedding size matches pair space
    n_pairs = int(np.max(C_tr)) + 1
    assert n_pairs == config.num_clusters, \
        f"Mismatch: n_pairs={n_pairs} vs config.num_clusters={config.num_clusters}"

    # Per-pair log-odds prior and allowed-food mask (built from the training meals in fit)
    raw_config.setdefault("cvae_model", {})
    raw_config["cvae_model"].update({
        "use_pair_prior": True,
        "pair_prior_scale": 1.0,
        "use_pair_allowed_mask": True,
        "mask_strength": 12.0,            # ~ pushes p≈6e-6 on disallowed foods
        "allowed_topk": "median",         # or an int (e.g., 8) or "quantile:0.75"
        "min_prev": 0.005                 # 0.5% minimum prevalence within the pair
    })

    # Output directory, for the training monitor
    raw_config["outdir"] = outdir
    model = MultiMealPresenceOnlyCVAE(config, raw_config)

    t0 = time.time()
    model.fit(X_tr, C_tr, M_tr, X_val, C_val, M_val,
              epochs=epochs, batch_size=batch_size, patience=patience, logger=logger)
    train_time = time.time() - t0
    print(f"✅ Training completed in {train_time:.1f}s")
    return model, train_time


def evaluate_by_meal_type(model, X_all, C_all, M_all, meal_type_mapping, config, logger):
    """Evaluate the model separately for each meal type"""
    print("\n📊 Evaluating by meal type...")
    
    results = {}
    
    for meal_type, meal_idx in meal_type_mapping.items():
        print(f"\nEvaluating {meal_type}...")
        
        # Get data for this meal type
        mask = (M_all == meal_idx)
        X_meal = X_all[mask]
        C_meal = C_all[mask]
        M_meal = M_all[mask]
        
        if len(X_meal) == 0:
            print(f"  No data for {meal_type}")
            continue
        
        print(f"  {len(X_meal)} samples, {len(np.unique(C_meal))} clusters")
        
        # Get predictions
        p_presence = model.predict_heads(X_meal, C_meal, M_meal)
        
        # Run latent diagnostics to check for collapse
        try:
            latent_diagnostics(model, X_meal, C_meal, M_meal, name=f"{meal_type}_latent")
        except Exception as e:
            print(f"  Latent diagnostics failed: {e}")
        
        # Run z-ablation check
        try:
            dz = z_ablation_delta(model, X_meal[:2000], C_meal[:2000], M_meal[:2000])
            print(f"  Δ(p|z=μ vs z=0): {dz:.4f}")
        except Exception as e:
            print(f"  z-ablation failed: {e}")
        
        # The model predicts presence only: presence probabilities stand in for expected grams
        exp_grams = p_presence

        # Evaluate
        metrics = model.evaluate(
            X_meal, X_meal, C_meal, M_meal,
            logger=logger,
            presence_threshold="optimize",  # Use optimized threshold
            split_name=f"{meal_type}",
        )
        
        # Compute cluster-level metrics
        cluster_df = compute_metrics_by_cluster(
            meal_type, y_true_grams=np.asarray(X_meal, np.float32),
            p_presence=p_presence, e_present=p_presence, exp_grams=exp_grams, C=C_meal,
            eps=config.presence_eps
        )
        
        # Pair-level checks: leakage, top-K alignment, count MAE
        pair_eval_results = {}
        try:
            # A) Leakage check
            leakage_results = eval_leakage(model, X_meal, C_meal, M_meal)
            pair_eval_results['leakage'] = leakage_results
            if 'leakage_mean' in leakage_results and not np.isnan(leakage_results['leakage_mean']):
                print(f"  Leakage (disallowed foods): {leakage_results['leakage_mean']:.4f}")
            else:
                print(f"  Leakage: {leakage_results.get('note', 'N/A')}")
        except Exception as e:
            print(f"  Leakage eval failed: {e}")
            pair_eval_results['leakage'] = {"error": str(e)}

        try:
            # B) Top-K alignment
            alignment_results = eval_topk_alignment(model, X_meal, C_meal, M_meal, K=10)
            pair_eval_results['alignment'] = alignment_results
            print(f"  P@10: {alignment_results['P@10']:.3f}  •  nDCG@10: {alignment_results['nDCG@10']:.3f}  •  Pearson: {alignment_results['Pearson_r']:.3f}")
        except Exception as e:
            print(f"  Alignment eval failed: {e}")
            pair_eval_results['alignment'] = {"error": str(e)}

        try:
            # C) Count MAE per pair
            count_mae_results = eval_count_mae_per_pair(model, X_meal, C_meal, M_meal)
            pair_eval_results['count_mae_per_pair'] = count_mae_results
            print(f"  Count MAE per pair: {count_mae_results['count_mae_per_pair_mean']:.3f} (n_pairs={count_mae_results['n_pairs']})")
        except Exception as e:
            print(f"  Count MAE per pair eval failed: {e}")
            pair_eval_results['count_mae_per_pair'] = {"error": str(e)}

        results[meal_type] = {
            'metrics': metrics,
            'cluster_df': cluster_df,
            'n_samples': len(X_meal),
            'n_clusters': len(np.unique(C_meal)),
            'pair_evaluations': pair_eval_results
        }
        
        print(f"  F1 micro: {metrics['f1_micro']:.3f}  •  AUROC: {metrics['auroc_presence']:.3f}  •  Brier: {metrics.get('brier_score', 0):.3f}  •  ECE: {metrics.get('ece', 0):.3f}")
        print(f"  Count MAE: {metrics['mae_count']:.2f}  •  Bias: {metrics['count_bias']:.2f}")
    
    return results


def create_comprehensive_visualizations(model, results, X_all, C_all, M_all, 
                                      meal_type_mapping, config, output_dir):
    """Write one evaluation dashboard per meal type to output_dir/<meal>_multi_meal_panel.png"""
    print("\n🎨 Creating comprehensive visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for meal_type, result in results.items():
        print(f"  Creating dashboard for {meal_type}...")
        
        meal_idx = meal_type_mapping[meal_type]
        mask = (M_all == meal_idx)
        
        X_meal = X_all[mask]
        C_meal = C_all[mask]
        M_meal = M_all[mask]
        
        # Get predictions
        p_presence = model.predict_heads(X_meal, C_meal, M_meal)
        
        # Presence probabilities stand in for the amount tensors the panel expects
        exp_grams = np.asarray(p_presence, dtype=np.float32)
        e_present = np.asarray(p_presence, dtype=np.float32)

        # Create visualizations
        y_true_grams = np.asarray(X_meal, dtype=np.float32)
        y_true_bin = (y_true_grams > config.presence_eps).astype(int)
        
        try:
            viz = CVAEPanelVisualizer(meal_type, out_dir=output_dir)
            panel_path = os.path.join(output_dir, f"{meal_type}_multi_meal_panel.png")
            
            # Helper function to safely extract scalar threshold
            def _scalar_threshold(thr):
                # Accept {…}, list/array, numpy scalar, python scalar, or None
                if thr is None:
                    return 0.5
                if isinstance(thr, dict):
                    # prefer key 'scalar', fall back to first value
                    if 'scalar' in thr and thr['scalar'] is not None:
                        return float(thr['scalar'])
                    # next(iter(...)) works even if it's an OrderedDict-like
                    return float(next(iter(thr.values())))
                arr = np.asarray(thr).ravel()
                return float(arr[0]) if arr.size else 0.5
            
            # Extract clean scalar threshold
            thr = _scalar_threshold(result['metrics'].get('presence_threshold', 0.5))
            
            viz.render_dashboard(
                y_true_grams=y_true_grams,
                y_true_bin=y_true_bin,
                p_presence=p_presence,
                e_present=e_present,
                exp_grams=np.clip(exp_grams, 0.0, 1e12),
                thresholds=thr,  # scalar float
                C=C_meal,
                save_as=panel_path,
                metrics=result['metrics'],
            )
            print(f"    ✅ Dashboard saved: {panel_path}")
        except Exception as e:
            print(f"    ❌ Failed to create dashboard for {meal_type}: {e}")


def run_cross_validation(
    X_dev, C_dev, M_dev,
    *, folds: int,
    config, raw_config, logger,
    epochs, batch_size, patience,
    meal_type_mapping, outdir: str
):
    """
    Stratified K-fold on (X_dev, C_dev, M_dev) using C_dev (pair code) for stratification.
    Produces:
      - {outdir}/cv/fold_{i}/results.csv  (per-meal metrics per fold)
      - {outdir}/cv/summary_mean_std.csv (mean/std across folds by meal)
      - Prints a full table with mean & std for ALL metrics.
    """
    from sklearn.model_selection import StratifiedKFold
    import os
    import numpy as np
    import json
    import pandas as pd
    os.makedirs(os.path.join(outdir, "cv"), exist_ok=True)

    # Cap folds by the rarest class BEFORE constructing the splitter
    counts = np.bincount(C_dev)
    max_k = int(counts.min()) if len(counts) else 0
    if folds > max_k:
        print(f"⚠️  Capping cv_folds from {folds} → {max_k} due to rarest class count.")
        folds = max_k
    if folds < 2:
        print("⚠️  Skipping CV: not enough samples per class for at least 2 folds.")
        return None, []

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    fold_tables = []  # list[pd.DataFrame] per fold (rows = meals)
    calib_dump = []   # per-fold calibration blobs

    for fi, (tr_idx, val_idx) in enumerate(skf.split(X_dev, C_dev), start=1):
        # per-fold determinism
        try:
            set_global_seed(42 + fi)
        except Exception:
            pass

        print(f"\n🔁 CV fold {fi}/{folds}: train={len(tr_idx)}, val={len(val_idx)}")

        X_tr, C_tr, M_tr = X_dev[tr_idx], C_dev[tr_idx], M_dev[tr_idx]
        X_val, C_val, M_val = X_dev[val_idx], C_dev[val_idx], M_dev[val_idx]

        model, train_time = train_multi_meal_model(
            X_tr, C_tr, M_tr, X_val, C_val, M_val,
            config, raw_config, logger, epochs, batch_size, patience, outdir=outdir
        )
        fold_results = evaluate_by_meal_type(model, X_val, C_val, M_val, meal_type_mapping, config, logger)
        # keep this fold's calibration
        calib_dump.append(model.export_calibration())
        
        # Print pair evaluation summary for this fold
        print_pair_evaluation_summary(fold_results, f"CV Fold {fi}")

        fold_dir = os.path.join(outdir, "cv", f"fold_{fi}")
        os.makedirs(fold_dir, exist_ok=True)

        # Save fold model (optional) and its calibration (required)
        model.save(os.path.join(fold_dir, "model"))  # writes calibration.json inside
        with open(os.path.join(fold_dir, "calibration.json"), "w") as f:
            json.dump(model.export_calibration(), f, indent=2)
        print(f"[calibration] Fold {fi} calibration → {os.path.join(fold_dir,'calibration.json')}")

        # Flatten to a tidy table for this fold
        rows = []
        for meal_type, res in fold_results.items():
            row = {"fold": fi, "Meal": meal_type,
                   "n_samples": res['n_samples'], "n_clusters": res['n_clusters']}
            row.update(res['metrics'])
            rows.append(row)
        df_fold = pd.DataFrame(rows)

        fold_dir = os.path.join(outdir, "cv", f"fold_{fi}")
        os.makedirs(fold_dir, exist_ok=True)
        df_fold.to_csv(os.path.join(fold_dir, "results.csv"), index=False)
        fold_tables.append(df_fold)

    # Aggregate mean/std over folds for each meal
    all_folds = pd.concat(fold_tables, ignore_index=True)

    # choose numeric columns for aggregations
    num_cols = all_folds.select_dtypes(include=[np.number]).columns.tolist()
    # exclude 'fold' from agg
    num_cols = [c for c in num_cols if c != "fold"]

    grouped = all_folds.groupby("Meal")
    df_mean = grouped[num_cols].mean().add_suffix("_mean").reset_index()
    df_std  = grouped[num_cols].std(ddof=1).add_suffix("_std").reset_index()
    df_summary = pd.merge(df_mean, df_std, on="Meal", how="inner")

    summary_path = os.path.join(outdir, "cv", "summary_mean_std.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"✅ CV summary saved to: {summary_path}")

    # ----- Print a FULL summary table with ALL metrics (mean & std) -----
    # Order columns: Meal, then <metric>_mean for all metrics, then <metric>_std
    metric_bases = sorted(set(
        c[:-5] for c in df_summary.columns if c.endswith("_mean")
    ))
    ordered_cols = ["Meal"] + [f"{m}_mean" for m in metric_bases] + [f"{m}_std" for m in metric_bases]
    ordered_cols = [c for c in ordered_cols if c in df_summary.columns]  # guard

    print("\n📊 Cross-Validation Summary (mean & std across folds for ALL metrics):")
    _print_pretty_table(df_summary[ordered_cols])

    # Optional: also print a compact "mean ± std" view
    try:
        compact = pd.DataFrame({"Meal": df_summary["Meal"]})
        for m in metric_bases:
            mu = df_summary[f"{m}_mean"]
            sd = df_summary[f"{m}_std"]
            compact[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(mu, sd)]
        print("\n📊 Cross-Validation Summary (compact mean ± std):")
        _print_pretty_table(compact)
    except Exception:
        pass

    # ---------- Aggregate & freeze calibration across folds ----------
    def _agg_list(key, reducer="median"):
        vals = [c.get(key) for c in calib_dump if c.get(key) is not None]
        if not vals:
            return None
        # dict aggregation (e.g., per-meal)
        if isinstance(vals[0], dict):
            all_keys = sorted({k for d in vals for k in d.keys()})
            out = {}
            for k in all_keys:
                arr = np.array([d.get(k, np.nan) for d in vals], dtype=float)
                if np.all(np.isnan(arr)):
                    continue
                v = np.nanmedian(arr) if reducer == "median" else np.nanmean(arr)
                out[int(k)] = float(v)
            return out
        # list/array aggregation
        arrs = [np.asarray(v, dtype=float) for v in vals]
        A = np.stack(arrs, axis=0)
        agg = np.nanmedian(A, axis=0) if reducer == "median" else np.nanmean(A, axis=0)
        return agg.tolist()

    def _clean_float(x, default=0.0):
        if x is None:
            return default
        try:
            xf = float(x)
            if np.isnan(xf) or np.isinf(xf):
                return default
            return xf
        except Exception:
            return default

    # 1) robust scalar pres_logit_offset
    #    Prefer scalar from folds; else mean of per-item; else mean of per-meal; else 0.0
    scalar_offsets = [c.get("pres_logit_offset") for c in calib_dump if c.get("pres_logit_offset") is not None]
    if len(scalar_offsets):
        pres_logit_offset = float(np.nanmean(scalar_offsets))
    else:
        per_item_list = [np.asarray(c.get("pres_logit_offsets"))
                        for c in calib_dump if c.get("pres_logit_offsets") is not None]
        if len(per_item_list):
            pres_logit_offset = float(np.nanmean(np.concatenate(per_item_list)))
        else:
            per_meal_list = [np.asarray(c.get("pres_logit_offsets_per_meal"))
                            for c in calib_dump if c.get("pres_logit_offsets_per_meal") is not None]
            if len(per_meal_list):
                pres_logit_offset = float(np.nanmean(np.stack(per_meal_list, axis=0)))
            else:
                pres_logit_offset = 0.0
    pres_logit_offset = _clean_float(pres_logit_offset, default=0.0)

    # 2) presence_threshold_mean: guard all-NaN case
    ptm_values = [c.get("presence_threshold_mean", np.nan) for c in calib_dump]
    ptm = float(np.nanmedian(ptm_values))
    if np.isnan(ptm) or np.isinf(ptm):
        ptm = 0.5  # sane default

    frozen_calib = {
        "presence_threshold_kind": calib_dump[0].get("presence_threshold_kind", "scalar") if calib_dump else "scalar",
        "presence_threshold": _agg_list("presence_threshold", "median"),
        "presence_threshold_mean": ptm,
        "pres_logit_offsets": _agg_list("pres_logit_offsets", "mean"),
        "pres_logit_offset": pres_logit_offset,
        "pres_logit_offsets_per_meal": _agg_list("pres_logit_offsets_per_meal", "mean"),
        "pres_temp": _clean_float(np.nanmean([c.get("pres_temp", 1.0) for c in calib_dump]) if calib_dump else 1.0, default=1.0),
        "pres_temp_per_meal": _agg_list("pres_temp_per_meal", "mean"),
        "provenance": {"built_from": "cv", "folds": folds}
    }

    calib_path = os.path.join(outdir, "cv", "frozen_calibration.json")
    with open(calib_path, "w") as f:
        # prevent NaN from ever being written
        json.dump(frozen_calib, f, indent=2, allow_nan=False)
    print(f"✅ Frozen calibration saved to: {calib_path}")


    return df_summary, fold_tables



def _reindex_pairs_and_table(C_pair: np.ndarray, pair_table: Dict[int, Tuple[int,int]]):
    """
    Densify pair codes to 0..K-1 and remap pair_table accordingly.
    Returns: C_new, pair_table_new, old2new_map
    """
    uniq = np.unique(C_pair)
    old2new = {int(old): int(i) for i, old in enumerate(uniq)}
    C_new = np.array([old2new[int(c)] for c in C_pair], dtype=np.int32)
    pair_table_new = {old2new[int(old)]: pair_table[int(old)] for old in uniq}
    return C_new, pair_table_new, old2new


def _can_stratify(y: np.ndarray, test_size: float) -> bool:
    # Each class needs at least 2 samples, and at least 1 expected in each split
    counts = np.bincount(y)
    if len(counts) == 0: 
        return False
    if counts.min() < 2:
        return False
    # Expect at least one sample per class in test and train
    if (counts * test_size).min() < 1 or (counts * (1 - test_size)).min() < 1:
        return False
    return True


# ----------------- SANITY SAMPLING HELPERS -----------------
def _sigmoid_np(x):
    x = np.clip(x, -80, 80).astype(np.float32)
    return 1.0 / (1.0 + np.exp(-x))

def _logit_np(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps).astype(np.float32)
    return np.log(p) - np.log1p(-p)

def _apply_cal_to_probs_if_any(model, probs, C_pair, M):
    # Always push to logit space, apply cal if available, then sigmoid back
    logits = _logit_np(probs)
    if hasattr(model, "_apply_all_presence_cal"):
        logits = model._apply_all_presence_cal(logits, C_pair, M)
    return _sigmoid_np(logits)

def _apply_hard_masking(model, logits, C_pair):
    """
    Apply hard masking to logits using the pair allowed gate.
    Sets disallowed items to -inf.
    """
    try:
        if hasattr(model.decoder, 'get_layer') and \
           any(getattr(layer, "name", "") == "pair_allowed_gate" for layer in model.decoder.layers):
            gate_layer = model.decoder.get_layer("pair_allowed_gate")
            gate = gate_layer.get_weights()[0]  # [K+1, D], 1=disallowed
            g = gate[C_pair]                   # [N, D]
            logits = np.where(g > 0.5, -np.inf, logits)
    except Exception as e:
        print(f"Warning: Could not apply hard masking: {e}")
    return logits

def _get_pair_expected_count(model, C_pair, M, X_train=None, C_train=None, M_train=None):
    """
    Pair-specific expected counts from training; else returns a vector filled with fallback.
    """
    if X_train is not None and C_train is not None:
        out = []
        for pair in np.unique(C_pair):
            mask = (C_train == pair)   # pair→meal; no need to AND with M
            if np.any(mask):
                true_counts = (X_train[mask] > 1e-6).sum(axis=1)
                out.append((pair, int(np.round(np.mean(true_counts)))))
        mp = dict(out)
        return np.array([mp.get(int(p), 8) for p in C_pair], dtype=int)
    else:
        return np.full(len(C_pair), 8, dtype=int)

def _decode_presence_probs_from_z(model, z, C_pair, M, apply_hard_masking=True, 
                                 X_train=None, C_train=None, M_train=None):
    """
    Robustly get presence probabilities for batches of z, C_pair, M.
    Works with presence-only or full models; applies calibration even if the
    model returns probability heads.
    """
    # Preferred modern API
    if hasattr(model, "predict_heads_from_z"):
        out = model.predict_heads_from_z(z, C_pair, M, batch_size=256)

        # If tuple/list, pick the first head that matches (B, D)
        if isinstance(out, (list, tuple)):
            for t in out:
                if isinstance(t, np.ndarray) and t.ndim == 2 and t.shape[0] == z.shape[0]:
                    # If it looks like probabilities, still apply calibration in logit space
                    if np.all((t >= -1e-6) & (t <= 1.0 + 1e-6)):
                        return _apply_cal_to_probs_if_any(model, t, C_pair, M)
                    # Otherwise treat as logits
                    logits = t.astype(np.float32)
                    if hasattr(model, "_apply_all_presence_cal"):
                        logits = model._apply_all_presence_cal(logits, C_pair, M)
                    if apply_hard_masking:
                        logits = _apply_hard_masking(model, logits, C_pair)
                    return _sigmoid_np(logits)

            # Fallback: treat the first as logits
            logits = np.asarray(out[0], dtype=np.float32)
            if hasattr(model, "_apply_all_presence_cal"):
                logits = model._apply_all_presence_cal(logits, C_pair, M)
            if apply_hard_masking:
                logits = _apply_hard_masking(model, logits, C_pair)
            return _sigmoid_np(logits)

        # Single array
        arr = np.asarray(out, dtype=np.float32)
        if np.all((arr >= -1e-6) & (arr <= 1.0 + 1e-6)):
            return _apply_cal_to_probs_if_any(model, arr, C_pair, M)
        if hasattr(model, "_apply_all_presence_cal"):
            arr = model._apply_all_presence_cal(arr, C_pair, M)
        if apply_hard_masking:
            arr = _apply_hard_masking(model, arr, C_pair)
        return _sigmoid_np(arr)

    # Fallback: call decoder directly
    out = model.decoder.predict([z, C_pair, M], verbose=0)
    if isinstance(out, (list, tuple)):
        for t in out:
            if isinstance(t, np.ndarray) and t.ndim == 2 and t.shape[0] == z.shape[0]:
                if np.all((t >= -1e-6) & (t <= 1.0 + 1e-6)):
                    return _apply_cal_to_probs_if_any(model, t, C_pair, M)
                logits = t.astype(np.float32)
                if hasattr(model, "_apply_all_presence_cal"):
                    logits = model._apply_all_presence_cal(logits, C_pair, M)
                if apply_hard_masking:
                    logits = _apply_hard_masking(model, logits, C_pair)
                return _sigmoid_np(logits)
        logits = np.asarray(out[0], dtype=np.float32)
    else:
        logits = np.asarray(out, dtype=np.float32)

    if np.all((logits >= -1e-6) & (logits <= 1.0 + 1e-6)):
        return _apply_cal_to_probs_if_any(model, logits, C_pair, M)

    if hasattr(model, "_apply_all_presence_cal"):
        logits = model._apply_all_presence_cal(logits, C_pair, M)
    if apply_hard_masking:
        logits = _apply_hard_masking(model, logits, C_pair)
    return _sigmoid_np(logits)

# ==== Sanity-sample analysis + invoke-only helpers ==============================================

def _try_model_load(model_dir: str):
    """Load a saved presence-only model (presence_only_model/)."""
    try:
        print(f"[loader] Trying MultiMealPresenceOnlyCVAE.load('{model_dir}')")
        return MultiMealPresenceOnlyCVAE.load(model_dir)
    except Exception as e:
        raise RuntimeError(f"Could not load model from '{model_dir}' ({e}). "
                           f"Make sure you pass the right --model_dir.") from e

def _read_stoplist(stoplist_arg: str) -> set:
    if not stoplist_arg or not stoplist_arg.strip():
        return set()
    import os, pandas as pd
    # file?
    if os.path.exists(stoplist_arg):
        try:
            s = pd.read_csv(stoplist_arg, header=None)[0].astype(str).str.strip()
            return set(x for x in s if x)
        except Exception:
            pass
        try:
            with open(stoplist_arg, "r") as f:
                return set(x.strip() for x in f if x.strip())
        except Exception:
            return set()
    # comma separated
    return set(x.strip() for x in stoplist_arg.split(",") if x.strip())


def _coerce_samples_to_itemsets(
    df: "pd.DataFrame",
    *,
    presence_eps: "str|float" = "auto",
    pair_table: dict | None = None,
    stoplist: set | None = None,
):
    """
    Accepts:
      • precomputed list column: one of ['items','present_items','top_items'] (string or list)
      • or wide matrix with per-item columns (probabilities or grams)
    Returns a tidy DataFrame with columns:
      ['meal','cluster_id','pair','sample_idx','items']  (items=list[str])
    """
    import numpy as np, pandas as pd

    stoplist = stoplist or set()

    # Ensure meal/cluster exist (restore from pair if needed)
    if "meal" not in df.columns or "cluster_id" not in df.columns:
        # Try to map from common column name variations
        if "meal_name" in df.columns and "orig_cluster_id" in df.columns:
            df = df.copy()
            df["meal"] = df["meal_name"]
            df["cluster_id"] = df["orig_cluster_id"]
        elif "meal_idx" in df.columns and "orig_cluster_id" in df.columns:
            df = df.copy()
            df["meal"] = df["meal_idx"]
            df["cluster_id"] = df["orig_cluster_id"]
        elif pair_table is not None and "pair_code" in df.columns:
            meals, cids = [], []
            for p in df["pair_code"].tolist():
                m, cid = pair_table[int(p)]
                meals.append(m); cids.append(cid)
            df = df.copy()
            df["meal"] = meals
            df["cluster_id"] = cids
        elif pair_table is not None and "pair" in df.columns:
            meals, cids = [], []
            for p in df["pair"].tolist():
                m, cid = pair_table[int(p)]
                meals.append(m); cids.append(cid)
            df = df.copy()
            df["meal"] = meals
            df["cluster_id"] = cids
        if "meal" not in df.columns or "cluster_id" not in df.columns:
            raise ValueError(
                "Samples must have 'meal' and 'cluster_id', or 'pair' with a valid pair_table, "
                "or ('meal_name'/'meal_idx' and 'orig_cluster_id')."
            )


    # normalize list-like if already present
    for col in ["items", "present_items", "top_items", "topK_names"]:
        if col in df.columns:
            items = df[col]
            # string → split on separators
            if items.dtype == object:
                def _to_list(x):
                    if isinstance(x, list):
                        return [str(t) for t in x]
                    if isinstance(x, (tuple, set)):
                        return [str(t) for t in x]
                    try:
                        import numpy as _np
                        if isinstance(x, _np.ndarray):
                            return [str(t) for t in x.tolist()]
                    except Exception:
                        pass
                    if isinstance(x, str):
                        if x.strip().startswith("["):
                            # JSON-style in CSV
                            try:
                                import json
                                arr = json.loads(x)
                                return [str(t) for t in arr]
                            except Exception:
                                pass
                        for sep in [";", ",", " "]:
                            if sep in x:
                                return [t for t in x.split(sep) if t]
                        return [x] if x else []
                    return []
                l = items.apply(_to_list)
            else:
                l = items.astype(str).apply(lambda s: [s] if s else [])
            # apply stoplist
            l = l.apply(lambda lst: [t for t in lst if t not in stoplist])
            out = df[["meal", "cluster_id", "pair"]].copy() if "pair" in df.columns else df[["meal", "cluster_id"]].copy()
            out["sample_idx"] = df.get("sample_idx", pd.Series(range(len(df))))
            out["items"] = l
            return out

    # Otherwise, interpret as wide matrix of per-item scores.
    # Decide which columns are item columns — anything not in id_cols.
    id_like = {"meal", "cluster_id", "pair", "sample_idx", "meal_name", "meal_idx", "orig_cluster_id", "pair_code", "items_str", "items", "topK_names", "topK_probs", "emp_topK_names", "emp_support_n", "overlap_at_K"}
    item_cols = [c for c in df.columns if c not in id_like and np.issubdtype(df[c].dtype, np.number)]
    if not item_cols:
        raise ValueError("Samples file must include a list-like items column or numeric per-item columns.")

    X = df[item_cols].to_numpy(dtype=float)
    # AUTO ε: probs (<=1.5) vs grams (>1.5)
    if presence_eps == "auto":
        xmax = float(np.nanmax(X))
        if xmax <= 1.5:
            eps = 0.35  # calibrated prob threshold default
        else:
            eps = 10.0  # grams default
    else:
        eps = float(presence_eps)

    present_mask = (X >= eps)
    items = []
    for i in range(present_mask.shape[0]):
        lst = [item_cols[j] for j in np.flatnonzero(present_mask[i])]
        if stoplist:
            lst = [t for t in lst if t not in stoplist]
        items.append(lst)

    out = df[["meal", "cluster_id", "pair"]].copy() if "pair" in df.columns else df[["meal", "cluster_id"]].copy()
    out["sample_idx"] = df.get("sample_idx", pd.Series(range(len(df))))
    out["items"] = items
    return out



def _near_duplicate_rate_for_meal(rep_sets: list[set], jaccard_thresh: float = 0.90) -> float:
    """
    Compute near-duplicate rate for a single meal:
      - Build a graph where nodes are clusters and an edge exists if
        Jaccard(rep_i, rep_j) >= jaccard_thresh.
      - duplicate_rate = 1 - (#connected_components / #clusters)
    """
    n = len(rep_sets)
    if n <= 1:
        return 0.0
    parent = list(range(n))
    rank = [0]*n
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra == rb: return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # connect near-duplicate clusters
    for i in range(n):
        Ai = rep_sets[i]
        for j in range(i+1, n):
            Aj = rep_sets[j]
            if not Ai and not Aj:
                sim = 1.0
            else:
                denom = len(Ai | Aj)
                sim = (len(Ai & Aj) / denom) if denom else 1.0
            if sim >= jaccard_thresh:
                union(i, j)

    comps = len({find(i) for i in range(n)})
    return 1.0 - (comps / float(n))


def _get_food_subcategory(food_code: int) -> str:
    """Map 8-digit food code to WWEIA subcategory based on first 4 digits."""
    code = int(food_code)
    # Extract first 4 digits for WWEIA category
    wweia_code = code // 10000
    
    if 1002 <= wweia_code < 1200:
        return "Milk"
    elif 1200 <= wweia_code < 1300:
        return "Flavored Milk"
    elif 1400 <= wweia_code < 1500:
        return "Dairy Drinks"
    elif 1600 <= wweia_code < 1700:
        return "Cheese"
    elif 1800 <= wweia_code < 1900:
        return "Yogurt"
    elif 2000 <= wweia_code < 2100:
        return "Meats"
    elif 2200 <= wweia_code < 2300:
        return "Poultry"
    elif 2400 <= wweia_code < 2500:
        return "Seafood"
    elif 2500 <= wweia_code < 2600:
        return "Eggs"
    elif 2600 <= wweia_code < 2700:
        return "Cured Meats"
    elif 2800 <= wweia_code < 2900:
        return "Plant Proteins"
    elif 3000 <= wweia_code < 3100:
        return "Mixed Meat Dishes"
    elif 3100 <= wweia_code < 3200:
        return "Mixed Bean Dishes"
    elif 3200 <= wweia_code < 3300:
        return "Mixed Grain Dishes"
    elif 3400 <= wweia_code < 3500:
        return "Asian Dishes"
    elif 3500 <= wweia_code < 3600:
        return "Mexican Dishes"
    elif 3600 <= wweia_code < 3700:
        return "Pizza"
    elif 3700 <= wweia_code < 3800:
        return "Sandwiches"
    elif 3800 <= wweia_code < 3900:
        return "Soups"
    elif 4000 <= wweia_code < 4100:
        return "Cooked Grains"
    elif 4200 <= wweia_code < 4300:
        return "Breads/Rolls"
    elif 4400 <= wweia_code < 4500:
        return "Quick Breads"
    elif 4600 <= wweia_code < 4700:
        return "Ready-to-Eat Cereals"
    elif 4800 <= wweia_code < 4900:
        return "Cooked Cereals"
    elif 5000 <= wweia_code < 5100:
        return "Savory Snacks"
    elif 5200 <= wweia_code < 5300:
        return "Crackers"
    elif 5400 <= wweia_code < 5500:
        return "Snack Bars"
    elif 5500 <= wweia_code < 5600:
        return "Sweet Bakery"
    elif 5700 <= wweia_code < 5800:
        return "Candy"
    elif 5800 <= wweia_code < 5900:
        return "Desserts"
    elif 6000 <= wweia_code < 7000:
        return "Fruits"
    elif 6400 <= wweia_code < 6500:
        return "Vegetables"
    elif 6800 <= wweia_code < 6900:
        return "Potatoes"
    elif 7000 <= wweia_code < 7100:
        return "100% Juice"
    elif 7100 <= wweia_code < 7200:
        return "Diet Beverages"
    elif 7200 <= wweia_code < 7300:
        return "Sweetened Beverages"
    elif 7300 <= wweia_code < 7400:
        return "Coffee/Tea"
    elif 7500 <= wweia_code < 7600:
        return "Alcoholic Beverages"
    elif 7700 <= wweia_code < 7800:
        return "Plain Water"
    elif 7800 <= wweia_code < 7900:
        return "Flavored Water"
    elif 8000 <= wweia_code < 8100:
        return "Fats/Oils"
    elif 8400 <= wweia_code < 8500:
        return "Condiments/Sauces"
    elif 8800 <= wweia_code < 8900:
        return "Sugars"
    elif 9000 <= wweia_code < 9100:
        return "Baby Foods"
    elif 9200 <= wweia_code < 9300:
        return "Baby Beverages"
    else:
        return "Other"


def _generate_subcategory_overlap_analysis(
    distributions_df: pd.DataFrame, analysis_dir: str, k_examples: int = 3, k_real: int = 3, seed: int = 7
) -> None:
    """Overlap@K by food subcategory (subcategory_overlap_analysis.csv, subcategory_overlap_summary.csv)."""
    print("🔍 Generating subcategory overlap analysis...")
    
    # Add subcategory information to the distributions
    distributions_df = distributions_df.copy()
    distributions_df['subcategory'] = distributions_df['item'].apply(_get_food_subcategory)
    
    # Calculate overlap@K by subcategory for each cluster
    subcategory_results = []
    
    for (meal, cluster_id), group in distributions_df.groupby(['meal', 'cluster_id']):
        meal_name = {0: 'Breakfast', 1: 'Lunch', 2: 'Dinner'}.get(meal, f'Meal_{meal}')

        # Calculate overlap for each subcategory
        for subcategory in group['subcategory'].unique():
            subcat_group = group[group['subcategory'] == subcategory]
            
            # Get top items in this subcategory for real and generated
            real_subcat_items = subcat_group.nlargest(k_real, 'p_real')['item'].tolist()
            gen_subcat_items = subcat_group.nlargest(k_examples, 'p_gen')['item'].tolist()
            
            # Calculate overlap for this subcategory
            overlap = len(set(real_subcat_items) & set(gen_subcat_items)) / max(k_real, k_examples)
            
            # Generic cluster label
            cluster_name = f"Cluster_{cluster_id}"
            
            subcategory_results.append({
                'meal': meal,
                'meal_name': meal_name,
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'subcategory': subcategory,
                'overlap_at_k': overlap,
                'n_real_items': len(real_subcat_items),
                'n_gen_items': len(gen_subcat_items),
                'real_top_items': real_subcat_items[:3],  # Top 3 for inspection
                'gen_top_items': gen_subcat_items[:3]
            })
    
    # Convert to DataFrame and save
    subcat_df = pd.DataFrame(subcategory_results)
    subcat_out = os.path.join(analysis_dir, "subcategory_overlap_analysis.csv")
    subcat_df.to_csv(subcat_out, index=False)
    print(f"📊 Subcategory overlap analysis → {subcat_out}")
    
    # Create summary by subcategory
    subcat_summary = subcat_df.groupby('subcategory').agg({
        'overlap_at_k': ['mean', 'std', 'count'],
        'n_real_items': 'mean',
        'n_gen_items': 'mean'
    }).round(3)
    
    subcat_summary.columns = ['overlap_mean', 'overlap_std', 'n_clusters', 'avg_real_items', 'avg_gen_items']
    subcat_summary = subcat_summary.sort_values('overlap_mean', ascending=False)
    
    summary_out = os.path.join(analysis_dir, "subcategory_overlap_summary.csv")
    subcat_summary.to_csv(summary_out)
    print(f"📈 Subcategory overlap summary → {summary_out}")
    
    # Print top and bottom performing subcategories
    print("\n🏆 Top performing subcategories (highest overlap@K):")
    for subcat, row in subcat_summary.head(5).iterrows():
        print(f"  {subcat}: {row['overlap_mean']:.3f} ± {row['overlap_std']:.3f} (n={row['n_clusters']})")
    
    print("\n⚠️  Bottom performing subcategories (lowest overlap@K):")
    for subcat, row in subcat_summary.tail(5).iterrows():
        print(f"  {subcat}: {row['overlap_mean']:.3f} ± {row['overlap_std']:.3f} (n={row['n_clusters']})")


def plot_training_trajectories(csv_path, output_path=None):
    """
    Create a single-panel scatter plot showing training trajectories.
    
    Axes: x = JS divergence (lower is better), y = Pearson correlation (higher is better)
    - One point per (meal,cluster,epoch)
    - Connect same (meal,cluster) across epochs with lines + arrows
    - Color = entropy_ratio (diverging colormap, 1.0 is neutral)
    - Size = sample size
    Saved next to the CSV as <name>_trajectories.png unless output_path is given.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Load the monitoring data
    df = pd.read_csv(csv_path)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique (meal, cluster) pairs for trajectories
    meal_cluster_pairs = df[['meal', 'cluster']].drop_duplicates()
    
    # Color mapping for entropy ratio (diverging colormap)
    entropy_vals = df['entropy_ratio'].values
    entropy_min, entropy_max = entropy_vals.min(), entropy_vals.max()
    entropy_center = 1.0
    
    # Create diverging colormap centered at 1.0
    if entropy_center >= entropy_min and entropy_center <= entropy_max:
        # Normalize so 1.0 is at the center
        vmin = min(entropy_min, 2 - entropy_max)  # symmetric around 1.0
        vmax = max(entropy_max, 2 - entropy_min)
    else:
        vmin, vmax = entropy_min, entropy_max
    
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    cmap = plt.cm.RdYlBu_r  # Red-Yellow-Blue reversed
    
    # Plot trajectories for each (meal, cluster) pair
    for _, (meal, cluster) in meal_cluster_pairs.iterrows():
        pair_data = df[(df['meal'] == meal) & (df['cluster'] == cluster)].sort_values('epoch')
        
        if len(pair_data) < 2:
            continue
            
        # Plot line with arrows
        ax.plot(pair_data['js'], pair_data['pearson'], 
                alpha=0.3, linewidth=1, color='gray')
        
        # Add arrows to show direction
        for i in range(len(pair_data) - 1):
            x1, y1 = pair_data.iloc[i]['js'], pair_data.iloc[i]['pearson']
            x2, y2 = pair_data.iloc[i+1]['js'], pair_data.iloc[i+1]['pearson']
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', alpha=0.5, color='gray', lw=0.8))
    
    # Plot all points with color and size encoding
    scatter = ax.scatter(df['js'], df['pearson'], 
                        c=df['entropy_ratio'], s=df['G'] * 0.5,  # Size based on sample size
                        cmap=cmap, norm=norm, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Entropy Ratio (gen/real)', rotation=270, labelpad=20)

    # Labels and title
    ax.set_xlabel('Jensen-Shannon Divergence (lower = better)', fontsize=12)
    ax.set_ylabel('Pearson Correlation (higher = better)', fontsize=12)
    ax.set_title('Training Trajectories: Fidelity vs Diversity\n' + 
                'Color = Entropy Ratio, Size = Sample Size, Arrows = Training Direction', 
                fontsize=14)
    
    # Invert x-axis so lower JS divergence is on the right (better)
    ax.invert_xaxis()
    
    # Add quadrant labels
    ax.text(0.05, 0.95, 'High Fidelity\nHigh Diversity', transform=ax.transAxes, 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            verticalalignment='top')
    ax.text(0.95, 0.95, 'High Fidelity\nLow Diversity', transform=ax.transAxes, 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
            verticalalignment='top', horizontalalignment='right')
    ax.text(0.05, 0.05, 'Low Fidelity\nHigh Diversity', transform=ax.transAxes, 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
            verticalalignment='bottom')
    ax.text(0.95, 0.05, 'Low Fidelity\nLow Diversity', transform=ax.transAxes, 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
            verticalalignment='bottom', horizontalalignment='right')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    if output_path is None:
        output_path = csv_path.replace('.csv', '_trajectories.png')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training trajectories plot saved to: {output_path}")
    return output_path


def plot_training_trajectories_clean(monitor_csv, outdir, max_pairs_per_meal=6):
    """
    Create cleaner trajectory plots without spaghetti lines.
    
    Creates two plots:
    1. Start→End arrows only, faceted by meal (no mid-epoch spaghetti)
    2. Aggregate ribbon: median path over epochs with 20–80% band across clusters
    """
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    
    mon = pd.read_csv(monitor_csv)
    mon = mon.dropna(subset=["js","pearson"])

    # pick a small, stable subset per meal by sample size
    mon["key"] = list(zip(mon["meal"], mon["cluster"]))
    tops = (mon.groupby(["meal","cluster"])["n_real"].max()
              .reset_index()
              .sort_values(["meal","n_real"], ascending=[True,False])
              .groupby("meal").head(max_pairs_per_meal))
    keep = set(zip(tops["meal"], tops["cluster"]))
    mon = mon[mon["key"].isin(keep)].copy()

    # color map centered at entropy_ratio=1.0
    er = mon["entropy_ratio"].to_numpy(float)
    er = np.where(np.isfinite(er), er, 1.0)
    vmin = min(er.min(), 2-er.max()); vmax = max(er.max(), 2-er.min())
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    cmap = plt.cm.RdYlBu_r

    # --- Panel A: Start→End arrows, faceted by meal ---
    meals = {0:"Breakfast",1:"Lunch",2:"Dinner"}
    fig, axes = plt.subplots(1, 3, figsize=(14,4.6), sharey=True)
    for mi, ax in enumerate(axes):
        sub = mon[mon["meal"]==mi]
        for key, g in sub.groupby("key"):
            g = g.sort_values("epoch")
            x0,y0 = g.iloc[0][["js","pearson"]]
            x1,y1 = g.iloc[-1][["js","pearson"]]
            c = cmap(norm(g.iloc[-1]["entropy_ratio"]))
            ax.arrow(x0, y0, x1-x0, y1-y0,
                     length_includes_head=True, head_width=0.008,
                     head_length=0.006, lw=0.8, color=c, alpha=0.9)
            ax.scatter([x0,x1],[y0,y1], s=[25,35], c=[c,c], edgecolors="k", linewidths=0.6)
            ax.text(x1, y1, str(int(g["cluster"].iloc[0])), fontsize=7,
                    ha="left", va="bottom")
        ax.axvline(0.20, ls="--", lw=1, color="0.75"); ax.axhline(0.90, ls="--", lw=1, color="0.75")
        ax.set_title(meals.get(mi,str(mi)))
        ax.set_xlabel("JS divergence (lower = better)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Pearson correlation (higher = better)")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist())
    cb.set_label("Entropy ratio  N1_gen / N1_real (≈1.0 on-target)")
    for ax in axes: ax.invert_xaxis(); ax.set_ylim(0.5,1.0)
    fig.suptitle("Training trajectories — start→end only", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "trajectories_start_end_by_meal.png"), dpi=200)
    plt.close(fig)

    # --- Panel B: Aggregate ribbon (median ± quantile over clusters) ---
    agg = (mon.groupby("epoch")
             .agg(js_med=("js","median"),
                  js_lo =("js",lambda x: x.quantile(0.2)),
                  js_hi =("js",lambda x: x.quantile(0.8)),
                  r_med =("pearson","median"),
                  r_lo =("pearson",lambda x: x.quantile(0.2)),
                  r_hi =("pearson",lambda x: x.quantile(0.8)))
             .reset_index())
    fig, ax = plt.subplots(figsize=(7.2,4.6))
    ax.plot(agg["js_med"], agg["r_med"], lw=2)
    ax.fill_between(agg["js_med"], agg["r_lo"], agg["r_hi"], alpha=0.2, label="ρ 20–80%")
    ax.fill_betweenx(agg["r_med"], agg["js_lo"], agg["js_hi"], alpha=0.15, label="JS 20–80%")
    ax.scatter(agg["js_med"].iloc[[0,-1]], agg["r_med"].iloc[[0,-1]], s=[30,40], edgecolors="k", zorder=3)
    ax.axvline(0.20, ls="--", lw=1, color="0.75"); ax.axhline(0.90, ls="--", lw=1, color="0.75")
    ax.set_xlabel("JS divergence (lower = better)"); ax.set_ylabel("Pearson correlation")
    ax.set_title("Aggregate training trajectory (median across clusters)")
    ax.grid(True, alpha=0.3); ax.invert_xaxis(); ax.set_ylim(0.5,1.0); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "trajectories_aggregate_ribbon.png"), dpi=200)
    plt.close(fig)
    
    print(f"📊 Clean trajectory plots saved to: {outdir}")
    return outdir


def analyze_conditioned_samples(samples_dir, analysis_dir,
                                presence_eps="auto",
                                rep_threshold=0.4,
                                dup_flag_threshold=0.90,
                                pair_table=None,
                                stoplist=None,
                                divergence="js"):
    """
    Compare the sample meals in samples_dir with the real meals of each (meal, cluster).

    Reads the first *.parquet (else *.jsonl, else *.csv) file in samples_dir and reloads the
    real meals with load_all_meal_data. Writes to analysis_dir:
    real_vs_generated_distributions.csv, real_vs_generated_summary.csv,
    overview_real_vs_generated.csv/.png/.svg (the per-archetype table behind Figure 3A, plotted
    by meal type; `make eval` draws Figure 3A itself), n1_real_vs_gen.png, coverage_curves_by_meal.png,
    representative_meals.csv, subcategory_overlap_analysis.csv, subcategory_overlap_summary.csv,
    jaccard_heatmap_<meal>.png, duplicate_rates.csv, diversity_by_meal.csv,
    intra_inter_separation.csv and report.md.

    - presence_eps: 'auto' | float
    - stoplist: set of item codes to exclude from diversity metrics
    - divergence: 'js', 'kl' or 'skl'; recorded in the summary CSV (the overview plot uses JS)
    """
    import os, glob, pandas as pd, numpy as np, matplotlib.pyplot as plt

    os.makedirs(analysis_dir, exist_ok=True)
    
    comp_rows, summ_rows = [], []
    path = None
    for pat in ["*.parquet", "*.jsonl", "*.csv"]:
        cand = sorted(glob.glob(os.path.join(samples_dir, pat)))
        if cand:
            path = cand[0]; break
    if path is None:
        print(f"[analyze] No samples found in {samples_dir}")
        return

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)

    # Coerce to tidy itemsets
    s = _coerce_samples_to_itemsets(df, presence_eps=presence_eps,
                                    pair_table=pair_table, stoplist=stoplist)

    # ---- build representative sets per (meal, cluster) ----
    rep = (s.explode("items")
             .assign(one=1)
             .groupby(["meal","cluster_id","items"])["one"].sum()
             .reset_index())
    n_per = s.groupby(["meal","cluster_id"])["sample_idx"].nunique().reset_index()
    rep = rep.merge(n_per, on=["meal","cluster_id"], how="left", suffixes=("","_n"))
    rep["support"] = rep["one"] / rep["sample_idx"]
    repsets = (rep[rep["support"] >= rep_threshold]
               .groupby(["meal","cluster_id"])["items"].apply(list).reset_index()
               .rename(columns={"items":"rep_items"}))

    # ---- Add food names and cluster names to representative meals ----
    food_name_mapping = _load_food_name_mapping()
    
    # ---- Real vs Generated: per-(meal,cluster) distributions ----
    from collections import Counter

    def _js_divergence(p, q, eps=1e-12):
        p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
        p = (p + eps); p /= p.sum()
        q = (q + eps); q /= q.sum()
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
        kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
        return 0.5 * (kl_pm + kl_qm)

    def _kl_divergence(p, q, eps=1e-12):
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        p = np.clip(p, eps, None); p = p / p.sum()
        q = np.clip(q, eps, None); q = q / q.sum()
        return float(np.sum(p * (np.log(p) - np.log(q))))

    def _pearson_rho(p, q):
        ps = pd.Series(np.asarray(p, dtype=float))
        qs = pd.Series(np.asarray(q, dtype=float))
        if float(ps.std(ddof=0)) == 0.0 or float(qs.std(ddof=0)) == 0.0:
            return np.nan
        return float(np.corrcoef(ps, qs)[0, 1])

    # Load real data the same way training did
    meal_types = ["Breakfast", "Lunch", "Dinner"]
    X_all, C_all, M_all, feature_names, meal_type_mapping, _pair_table_real, orig_cluster_ids = load_all_meal_data(meal_types)
    idx2meal = {v: k for k, v in (meal_type_mapping or {}).items()}
    
    # Choose presence eps for real grams
    if presence_eps == "auto":
        real_eps = 10.0
    else:
        real_eps = float(presence_eps) if float(presence_eps) > 1.5 else 10.0

    X_real_bin = (np.asarray(X_all, dtype=float) > real_eps).astype(np.uint8)
    feat_index = {f: i for i, f in enumerate(feature_names)}
    
    # Helper for the overview figure's diversity axis
    def _compute_food_diversity(X_grams, min_grams=5.0):
        """Compute Shannon diversity normalized by log(K) for food items."""
        if len(X_grams) == 0:
            return np.nan
            
        try:
            diversities = []
            for meal in X_grams:
                # Filter foods with sufficient grams
                valid_foods = meal >= min_grams
                if not np.any(valid_foods):
                    diversities.append(0.0)
                    continue
                    
                meal_filtered = meal[valid_foods]
                K = len(meal_filtered)
                
                if K <= 1:
                    diversities.append(0.0)
                    continue
                
                # Compute proportions
                p = meal_filtered / np.sum(meal_filtered)
                p = np.maximum(p, 1e-10)  # Avoid log(0)
                
                # Shannon entropy
                H = -np.sum(p * np.log(p))
                
                # Normalize by log(K) to get [0,1] range
                diversity = H / np.log(K)
                diversities.append(diversity)
            
            return float(np.mean(diversities))
        except Exception as e:
            print(f"[WARN] Food diversity computation failed: {e}")
            return np.nan

    # Generated pairs present in s
    gen_pairs = s[["meal", "cluster_id"]].drop_duplicates().itertuples(index=False, name=None)

    for (meal_idx, clu_id) in gen_pairs:
        sg = s[(s["meal"] == meal_idx) & (s["cluster_id"] == clu_id)]
        n_gen = int(sg["sample_idx"].nunique())

        # generated prevalence per item
        cnt_gen = Counter()
        by_sample = sg.groupby("sample_idx")["items"].sum()
        for items_list in by_sample:
            cnt_gen.update(set(items_list))

        # real prevalence per item
        mask_real = (M_all == meal_idx) & (orig_cluster_ids == clu_id)
        n_real = int(mask_real.sum())
        if n_real == 0:
            continue
        real_mean = X_real_bin[mask_real].mean(axis=0)

        # align over union of items (sorted: set order depends on the per-process string hash seed)
        items_union = sorted(set(feature_names) | set(cnt_gen.keys()), key=str)
        p_real, p_gen, items_aligned = [], [], []
        for code in items_union:
            items_aligned.append(code)
            j = feat_index.get(code, None)
            p_real.append(float(real_mean[j]) if j is not None else 0.0)
            p_gen.append((cnt_gen[code] / max(1, n_gen)) if code in cnt_gen else 0.0)

        js = _js_divergence(p_real, p_gen)
        kl_pq = _kl_divergence(p_real, p_gen)
        kl_qp = _kl_divergence(p_gen, p_real)
        skl   = 0.5 * (kl_pq + kl_qp)

        if divergence == "js":
            div_value = js
            div_key   = "js_divergence"
        elif divergence == "kl":
            div_value = kl_pq
            div_key   = "kl_divergence_p||q"
        else:  # "skl"
            div_value = skl
            div_key   = "symmetric_kl"

        rho = _pearson_rho(p_real, p_gen)
        # ----- K* = median real item count for this (meal, cluster) -----
        real_item_counts = X_real_bin[mask_real].sum(axis=1)
        K_star = int(np.median(real_item_counts))
        if K_star < 1:  # guard
            K_star = 1
        K_star = min(K_star, len(items_aligned))  # can’t exceed available items

        # Top-K* lists by prevalence
        order_real = np.argsort(-np.asarray(p_real))
        order_gen  = np.argsort(-np.asarray(p_gen))
        real_top   = [items_aligned[i] for i in order_real[:K_star]]
        gen_top    = [items_aligned[i] for i in order_gen[:K_star]]

        overlap = len(set(real_top) & set(gen_top)) / float(max(1, K_star))
        
        meal_name = idx2meal.get(meal_idx, str(meal_idx))

        # Food diversity of the generated meals, each present food counted as 100 g (so it is
        # the share of generated meals with two or more foods)
        X_gen_bin = np.zeros((n_gen, len(feature_names)), dtype=float)
        for i, (_, row) in enumerate(sg.iterrows()):
            for item in row['items']:
                if item in feat_index:
                    X_gen_bin[i, feat_index[item]] = 1.0
        X_gen_grams = X_gen_bin * 100.0
        food_diversity = _compute_food_diversity(X_gen_grams)

        for code, pr, pg in zip(items_aligned, p_real, p_gen):
            comp_rows.append({
                "meal": int(meal_idx),
                "meal_name": idx2meal.get(meal_idx, str(meal_idx)),
                "cluster_id": int(clu_id),
                "item": str(code),
                "p_real": float(pr),
                "p_gen": float(pg),
                "n_real_meals": n_real,
                "n_gen_meals": n_gen,
            })

        # Get cluster name for this meal and cluster
        cluster_name = _get_cluster_name(meal_name, clu_id)
        
        summ_rows.append({
            "meal": int(meal_idx),
            "meal_name": meal_name,
            "cluster_id": int(clu_id),
            "cluster_name": cluster_name,
            "n_real_meals": n_real,
            "n_gen_meals": n_gen,
            "js_divergence": float(js),
            "kl_divergence_p||q": float(kl_pq),
            "kl_divergence_q||p": float(kl_qp),
            "symmetric_kl": float(skl),
            "chosen_divergence": div_key,
            "divergence_value_for_plot": float(div_value),
            "pearson_rho": float(rho) if np.isfinite(rho) else np.nan,
            "overlap_at_k": float(overlap),
            "k_star": int(K_star),
            "food_diversity": float(food_diversity) if np.isfinite(food_diversity) else np.nan,
        })

    # Write CSVs now
    comp_path = os.path.join(analysis_dir, "real_vs_generated_distributions.csv")
    pd.DataFrame(comp_rows).to_csv(comp_path, index=False)
    print(f"📝 Real vs Generated (per item) → {comp_path}")

    summ_path = os.path.join(analysis_dir, "real_vs_generated_summary.csv")
    pd.DataFrame(summ_rows).to_csv(summ_path, index=False)
    print(f"📝 Real vs Generated (summary) → {summ_path}")

    # ===== Overview (the layout of Figure 3A): per-archetype JS divergence, food diversity, size and
    # coverage@N, written to overview_real_vs_generated.csv and plotted =====
    try:
        sumdf = pd.DataFrame(summ_rows).copy()     # has js_divergence, pearson_rho, n_real_meals, n_gen_meals, meal_name, cluster_id
        comp_df = pd.DataFrame(comp_rows).copy()   # has per-item rows: meal, cluster_id, p_real, p_gen, ...

        # ---- helpers ----
        def _eff_species_num(p):
            p = np.asarray(p, float)
            p = np.clip(p, 1e-12, None)
            s = p.sum()
            if s <= 0: return 0.0
            p = p / s
            return float(np.exp(-np.sum(p*np.log(p))))

        def _coverage_at_n(real_sets, gen_sets, n):
            if len(real_sets) == 0 or len(gen_sets) == 0:
                return np.nan
            r = set().union(*real_sets[:min(n, len(real_sets))])
            g = set().union(*gen_sets[:min(n, len(gen_sets))])
            return len(g) / max(1, len(r))

        def _bootstrap_js_rho(p_real, p_gen, n_real, n_gen, B=200, seed=7):
            """Parametric bootstrap: Binomial around per-item presence, then re-normalize for JS and Pearson."""
            rng = np.random.default_rng(seed)
            p_real = np.clip(np.asarray(p_real, float), 0.0, 1.0)
            p_gen  = np.clip(np.asarray(p_gen,  float), 0.0, 1.0)
            xs, ys = [], []
            for _ in range(B):
                cr = rng.binomial(max(int(n_real),1), p_real).astype(float)
                cg = rng.binomial(max(int(n_gen), 1), p_gen ).astype(float)
                # prevalence estimates
                prb = cr / max(int(n_real),1)
                pgb = cg / max(int(n_gen), 1)
                # normalize to distributions for JS
                def _norm(v):
                    v = np.clip(v, 1e-12, None)
                    s = v.sum()
                    return v / (s if s>0 else 1.0)
                prd = _norm(prb); pgd = _norm(pgb)
                # JS
                m = 0.5*(prd+pgd)
                kl_pm = float(np.sum(prd*(np.log(prd)-np.log(m))))
                kl_qm = float(np.sum(pgd*(np.log(pgd)-np.log(m))))
                jsb = 0.5*(kl_pm+kl_qm)
                # Pearson across items (guard zero-variance)
                if np.std(prb)==0 or np.std(pgb)==0:
                    rhob = np.nan
                else:
                    rhob = float(np.corrcoef(prb, pgb)[0,1])
                xs.append(jsb); ys.append(rhob)
            return np.nanstd(xs), np.nanstd(ys)

        # ---- compute entropy-ratio (N1_gen/N1_real) and coverage@N for each (meal,cluster) ----
        # per-pair arrays
        pair_stats = {}
        for (meal, clu), g in comp_df.groupby(["meal","cluster_id"]):
            pr = g["p_real"].to_numpy(float)
            pg = g["p_gen"].to_numpy(float)
            n_real = int(g["n_real_meals"].iloc[0])
            n_gen  = int(g["n_gen_meals"].iloc[0])
            N1r = _eff_species_num(pr)
            N1g = _eff_species_num(pg)
            er  = (N1g / N1r) if N1r > 0 else np.nan

            # coverage@N using sets from real (X_real_bin) and generated (s)
            mask_real = (M_all == int(meal)) & (orig_cluster_ids == int(clu))
            cols = np.array(feature_names)
            real_sets = [set(cols[np.flatnonzero(row)]) for row in np.asarray(X_real_bin[mask_real], int)]
            gen_sets  = [set(lst) for lst in s[(s["meal"]==meal) & (s["cluster_id"]==clu)]["items"].tolist()]
            NN = max(1, min(1000, len(real_sets), len(gen_sets)))  # cap like the monitor
            cov = _coverage_at_n(real_sets, gen_sets, NN)

            # bootstrap errs (optional; skips if not enough info)
            try:
                xse, yse = _bootstrap_js_rho(pr, pg, n_real, n_gen, B=200, seed=7)
            except Exception:
                xse, yse = np.nan, np.nan

            pair_stats[(int(meal), int(clu))] = {
                "entropy_ratio": er, "coverage_at_N": cov,
                "js_se": xse, "rho_se": yse
            }

        # attach to sumdf
        er_list, cov_list, js_se_list, rho_se_list = [], [], [], []
        for _, r in sumdf.iterrows():
            key = (int(r["meal"]), int(r["cluster_id"]))
            st = pair_stats.get(key, {})
            er_list.append(st.get("entropy_ratio", np.nan))
            cov_list.append(st.get("coverage_at_N", np.nan))
            js_se_list.append(st.get("js_se", np.nan))
            rho_se_list.append(st.get("rho_se", np.nan))
        sumdf["entropy_ratio"] = er_list
        sumdf["coverage_at_N"] = cov_list
        sumdf["js_se"] = js_se_list
        sumdf["rho_se"] = rho_se_list

        sumdf.to_csv(os.path.join(analysis_dir, "overview_real_vs_generated.csv"), index=False)
        # Coloured by meal type here. `make eval` draws Figure 3A from this table, coloured by each
        # archetype's change in RDI deviation (src/utils/cluster_overview.py).
        from src.utils.cluster_overview import plot_cluster_overview
        out_png = os.path.join(analysis_dir, "overview_real_vs_generated.png")
        out_svg = os.path.join(analysis_dir, "overview_real_vs_generated.svg")
        plot_cluster_overview(sumdf, out_png, out_svg, title="Meal Cluster Quality: Fidelity and Diversity")

        # embed in report.md
        try:
            with open(os.path.join(analysis_dir, "report.md"), "a") as f:
                f.write("\n## Consolidated overview (JS vs ρ)\n\n")
                f.write("![overview](overview_real_vs_generated.png)\n")
        except Exception:
            pass
    except Exception as e:
        print(f"[overview] Failed to build consolidated plot: {e}")


    try:
        n1_rows = []
        for (meal, clu), g in comp_df.groupby(["meal","cluster_id"]):
            pr = g["p_real"].to_numpy(float); pg = g["p_gen"].to_numpy(float)
            n1r = _eff_species_num(pr); n1g = _eff_species_num(pg)
            n1_rows.append({"meal":meal,"cluster_id":clu,"N1_real":n1r,"N1_gen":n1g})
        n1df = pd.DataFrame(n1_rows)
        fig, ax = plt.subplots(figsize=(6.5,6))
        ax.scatter(n1df["N1_real"], n1df["N1_gen"], s=28, edgecolor="k", alpha=0.9)
        lim = float(max(n1df["N1_real"].max(), n1df["N1_gen"].max()))
        ax.plot([0,lim],[0,lim],"--",color="0.6")
        ax.set_xlabel("N1_real = exp(H_real)"); ax.set_ylabel("N1_gen = exp(H_gen)")
        ax.set_title("Effective species number (diversity) — real vs generated")
        ax.grid(True, alpha=0.3)
        out_png = os.path.join(analysis_dir, "n1_real_vs_gen.png")
        fig.tight_layout(); fig.savefig(out_png, dpi=180); plt.close(fig)
        print(f"🖼  N1 scatter → {out_png}")
        with open(os.path.join(analysis_dir, "report.md"), "a") as f:
            f.write("\n## Diversity (N1) scatter\n\n![n1](n1_real_vs_gen.png)\n")
    except Exception as e:
        print(f"[n1] failed: {e}")

    try:
        fig, axes = plt.subplots(1, 3, figsize=(14,4.5), sharey=True)
        meal_names = {0:"Breakfast",1:"Lunch",2:"Dinner"}
        for mi, ax in zip(sorted(s["meal"].unique()), axes):
            # gen
            sg = s[s["meal"]==mi].sort_values(["cluster_id","sample_idx"])
            gen_sets = [set(lst) for lst in sg["items"].tolist()]
            # real
            mask = (M_all==mi)
            real_sets = [set(np.flatnonzero(row)) for row in np.asarray(X_real_bin[mask], int)]
            def _curve(sets):
                seen=set(); xs=[]; ys=[]
                for i,st in enumerate(sets, start=1):
                    seen |= st; xs.append(i); ys.append(len(seen))
                return np.array(xs), np.array(ys)
            if gen_sets:
                xg, yg = _curve(gen_sets); ax.plot(xg, yg, label="Generated")
            if real_sets:
                xr, yr = _curve(real_sets); ax.plot(xr, yr, label="Real")
            ax.set_title(meal_names.get(int(mi), str(mi)))
            ax.set_xlabel("# samples"); ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("Unique foods covered")
        axes[-1].legend()
        out_png = os.path.join(analysis_dir, "coverage_curves_by_meal.png")
        fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
        print(f"🖼  Coverage curves → {out_png}")
        with open(os.path.join(analysis_dir, "report.md"), "a") as f:
            f.write("\n## Coverage curves\n\n![cov](coverage_curves_by_meal.png)\n")
    except Exception as e:
        print(f"[coverage] failed: {e}")

    # Convert food codes to names in representative meals
    repsets["rep_item_names"] = repsets["rep_items"].apply(
        lambda items: _convert_food_codes_to_names(items, food_name_mapping)
    )
    
    # Add cluster names (robust to duplicated columns / empty frames)
    meal_names = {0: "Breakfast", 1: "Lunch", 2: "Dinner"}
    if repsets.empty:
        repsets["cluster_name"] = []
    else:
        meals = repsets["meal"].to_numpy()
        cids = repsets["cluster_id"].to_numpy()
        repsets["cluster_name"] = [
            _get_cluster_name(meal_names.get(int(m), str(m)), int(cid))
            for m, cid in zip(meals, cids)
        ]
    
    # ---- Save representative meals for inspection ----
    rep_out = os.path.join(analysis_dir, "representative_meals.csv")
    repsets.to_csv(rep_out, index=False)
    print(f"📝 Representative meals → {rep_out}")
    
    # Generate subcategory overlap analysis
    comp_df = pd.DataFrame(comp_rows)
    _generate_subcategory_overlap_analysis(
        comp_df, analysis_dir, k_examples=3, k_real=3, seed=7
    )

    # Also print nicely for quick inspection
    for meal in sorted(repsets["meal"].unique()):
        meal_name = meal_names[meal]
        print(f"\n=== {meal_name} ===")
        sub = repsets[repsets["meal"] == meal]
        for _, row in sub.iterrows():
            cluster_id = row["cluster_id"]
            cluster_name = row["cluster_name"]
            items = ", ".join(row["rep_items"])
            item_names = ", ".join(row["rep_item_names"])
            print(f"  {cluster_name} (ID {cluster_id}): {items}")
            print(f"    Names: {item_names}")


    # ---- Jaccard within each meal (+ heatmap figure) ----
    meals = sorted(s["meal"].unique().tolist())
    dup_rows, div_rows, sep_rows = [], [], []
    for meal in meals:
        sub = repsets[repsets["meal"]==meal].copy()
        cl = sub["cluster_id"].tolist()
        sets = [set(x) for x in sub["rep_items"].tolist()]
        K = len(sets)
        J = np.zeros((K,K), float)
        for i in range(K):
            for j in range(K):
                a, b = sets[i], sets[j]
                J[i,j] = len(a & b) / max(1, len(a | b))
        # heatmap
        fig, ax = plt.subplots(figsize=(10,10))
        im = ax.imshow(J, vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{J[i,j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"Jaccard similarity of representative sets — {meal}")
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(cl, rotation=45); ax.set_yticklabels(cl)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Jaccard")
        out_png = os.path.join(analysis_dir, f"jaccard_heatmap_{meal}.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close(fig)

        # near-duplicate rate via Jaccard threshold (connected components)
        dup_rate = _near_duplicate_rate_for_meal(sets, jaccard_thresh=float(dup_flag_threshold))
        dup_rows.append({
            "meal": meal,
            "duplicate_rate": float(dup_rate),
            "near_dup_threshold": float(dup_flag_threshold)
        })
        # also compute unique_values for diversity calc
        # Convert sets to sorted tuples, handling variable lengths
        tuples = [tuple(sorted(x)) for x in sets]
        # Use set to get unique tuples instead of np.unique (which fails with variable lengths)
        unique_values = list(set(tuples))

        # diversity ratio = unique representative sets / total samples
        total_samples = int(n_per[n_per["meal"]==meal]["sample_idx"].sum())
        div_rows.append({"meal": meal,
                         "unique_sets": int(len(unique_values)),
                         "total_samples": total_samples,
                         "diversity_ratio": (len(unique_values) / max(1, total_samples))})

        # intra/inter separation (median)
        med_intra = np.median([J[i,i] for i in range(K)]) if K else np.nan
        if K > 1:
            med_inter = np.median([J[i,j] for i in range(K) for j in range(K) if i!=j])
        else:
            med_inter = np.nan
        sep_rows.append({"meal": meal,
                         "median_intra_jaccard": med_intra,
                         "median_inter_jaccard": med_inter,
                         "separation": (med_intra - med_inter) if (np.isfinite(med_intra) and np.isfinite(med_inter)) else np.nan})

    # save CSV summaries
    pd.DataFrame(dup_rows).to_csv(os.path.join(analysis_dir, "duplicate_rates.csv"), index=False)
    pd.DataFrame(div_rows).to_csv(os.path.join(analysis_dir, "diversity_by_meal.csv"), index=False)
    pd.DataFrame(sep_rows).to_csv(os.path.join(analysis_dir, "intra_inter_separation.csv"), index=False)

    # write concise markdown report
    lines = [
        "# Conditioned Samples Quality Report",
        f"- Samples dir: `{samples_dir}`",
        f"- Presence ε: **{presence_eps}**",
        f"- Representative threshold: **{int(rep_threshold*100)}%**",
        f"- Near-duplicate Jaccard threshold: **{dup_flag_threshold:.2f}**",
        f"- Stop-list size: **{len(stoplist or [])}**",
        ""
    ]
    for meal in meals:
        lines.append(f"## Meal {meal}")
        lines.append(f"![Jaccard heatmap {meal}](jaccard_heatmap_{meal}.png)")
        lines.append("")
    with open(os.path.join(analysis_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"📝 Analysis → {analysis_dir}/report.md")


def maybe_analyze_sanity_samples(*, outdir: str, presence_eps="auto", rep_threshold=0.4,
                                 dup_flag_threshold=0.90, pair_table=None, stoplist=None, divergence="js"):
    """Analyse <outdir>/sanity_samples into <outdir>/sanity_samples/analysis/."""
    import os
    samples_dir  = os.path.join(outdir, "sanity_samples")
    analysis_dir = os.path.join(samples_dir, "analysis")
    analyze_conditioned_samples(samples_dir, analysis_dir,
                                presence_eps=presence_eps,
                                rep_threshold=rep_threshold,
                                dup_flag_threshold=dup_flag_threshold,
                                pair_table=pair_table,
                                stoplist=stoplist,
                                divergence=divergence)


def maybe_generate_diversity_plot(*, outdir: str):
    """Plot the training trajectories in <outdir>/diversity_monitor.csv, if it exists."""
    import os

    # Look for diversity monitor CSV in the output directory
    monitor_csv = os.path.join(outdir, "diversity_monitor.csv")

    if os.path.exists(monitor_csv):
        print(f"📊 Found monitoring data: {monitor_csv}")
        try:
            # All-epoch trajectory plot (diversity_monitor_trajectories.png)
            plot_path = plot_training_trajectories(monitor_csv)
            print(f"✅ Diversity plot generated: {plot_path}")
            
            # Generate clean trajectory plots
            analysis_dir = os.path.join(outdir, "sanity_samples", "analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            plot_training_trajectories_clean(monitor_csv, analysis_dir, max_pairs_per_meal=6)
            print(f"✅ Clean trajectory plots generated in: {analysis_dir}")
            
        except Exception as e:
            print(f"⚠️  Failed to generate diversity plot: {e}")
    else:
        print(f"ℹ️  No monitoring data found at: {monitor_csv}")
        print("   (Monitoring CSV will be generated during training if monitor.enabled=true in config)")


def _get_feature_names(model, model_dir: str) -> List[str]:
    # Try model attributes first
    for attr in ["feature_names_", "feature_names", "features_", "features"]:
        v = getattr(model, attr, None)
        if v is not None and isinstance(v, (list, tuple)) and len(v):
            return list(map(str, v))
    # Try common sidecar files
    import json, os
    for fname in ["feature_names.json", "features.json", "model_meta.json"]:
        p = os.path.join(model_dir, fname)
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    obj = json.load(f)
                if isinstance(obj, list):
                    return list(map(str, obj))
                if isinstance(obj, dict):
                    cand = obj.get("feature_names") or obj.get("features")
                    if isinstance(cand, list):
                        return list(map(str, cand))
            except Exception:
                pass
    raise RuntimeError("Could not find feature names in model or sidecar files.")

def _save_model_robust(model, model_dir: str, feature_names: List[str], logger):
    os.makedirs(model_dir, exist_ok=True)
    
    # Debug: Check existing files before saving
    model_files = ["cvae.keras", "decoder.keras", "encoder.keras", "calibration.json"]
    print(f"[debug] _save_model_robust saving to: {model_dir}")
    for fname in model_files:
        fpath = os.path.join(model_dir, fname)
        if os.path.exists(fpath):
            old_size = os.path.getsize(fpath)
            old_time = os.path.getmtime(fpath)
            print(f"[debug] {fname} exists: {old_size} bytes, modified: {datetime.fromtimestamp(old_time).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"[debug] {fname} does not exist")
    
    try:
        print(f"[debug] Starting model save at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        model.save(model_dir, feature_names=feature_names)
        print(f"[debug] Model save completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Debug: Verify files were actually saved/updated
        for fname in model_files:
            fpath = os.path.join(model_dir, fname)
            if os.path.exists(fpath):
                new_size = os.path.getsize(fpath)
                new_time = os.path.getmtime(fpath)
                print(f"[debug] {fname} after save: {new_size} bytes, modified: {datetime.fromtimestamp(new_time).strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"[debug] ERROR: {fname} missing after save!")
                
    except Exception as e:
        print(f"⚠️  Model save encountered an issue: {e}")
        import traceback
        traceback.print_exc()
    print(f"✅ Model saved to: {model_dir}")


def _latent_dim_from_model(model, default=64):
    try:
        shp = getattr(model.decoder, "input_shape", None)
        if shp is not None:
            if isinstance(shp, (list, tuple)):
                z = shp[0][1]
            else:
                z = shp[1]
            if isinstance(z, int) and z > 0:
                return z
    except Exception:
        pass
    for attr in ["z_dim", "latent_dim"]:
        v = getattr(model, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return default


def _apply_temp_and_nucleus(p, temp=1.0, top_p=1.0):
    import numpy as np
    # temperature on logits
    if temp and temp > 0 and abs(temp - 1.0) > 1e-6:
        eps = 1e-8
        logits = np.log(np.clip(p, eps, 1 - eps)) - np.log1p(-np.clip(p, eps, 1 - eps))
        logits /= float(temp)
        p = 1.0 / (1.0 + np.exp(-logits))
    # nucleus (keep smallest set whose cumulative prob >= top_p; renormalize)
    if top_p < 1.0:
        # sort descending per row
        idx = np.argsort(-p, axis=1)
        ps  = np.take_along_axis(p, idx, axis=1)
        csum = np.cumsum(ps, axis=1)
        mask = (csum <= top_p)
        # ensure at least 1 kept
        first = np.zeros_like(mask)
        first[:, 0] = True
        keep = mask | first
        kept = np.where(keep, ps, 0.0)
        denom = kept.sum(axis=1, keepdims=True) + 1e-8
        ps = kept / denom
        # restore original order
        p = np.zeros_like(p)
        np.put_along_axis(p, idx, ps, axis=1)
    return p


def _load_food_name_mapping():
    """Load food code to name mapping from the food_to_nutrient CSV file"""
    import pandas as pd
    try:
        _repo_root = Path(__file__).resolve().parents[1]
        food_csv = _repo_root / "data" / "mappings" / "food_to_nutrient.csv"
        food_df = pd.read_csv(food_csv)
        # Create mapping from food code to name
        food_df["Food code"] = food_df["Food code"].astype(str).str.zfill(8)
        name_mapping = dict(zip(food_df["Food code"], food_df["Main food description"]))
        print(f"Loaded food name mapping: {len(name_mapping)} food codes")
        return name_mapping
    except Exception as e:
        print(f"Warning: Could not load food name mapping: {e}")
        return {}


def _convert_food_codes_to_names(food_codes, name_mapping):
    """Convert list of food codes to food names using the mapping"""
    if not name_mapping:
        return food_codes
    
    names = []
    for code in food_codes:
        # Try to find the code in the mapping
        name = name_mapping.get(str(code), code)
        names.append(name)
    return names


def _get_cluster_name(meal_type, cluster_id):
    """Human-readable cluster name from config/cluster_names.json, else "Cluster {id}"."""
    try:
        import json
        from pathlib import Path
        names = json.loads(Path("config/cluster_names.json").read_text(encoding="utf-8"))
        return names.get(str(meal_type).lower(), {}).get(str(int(cluster_id)), f"Cluster {cluster_id}")
    except Exception:
        return f"Cluster {cluster_id}"


def _expected_n_from_calibration_or_probs(model, P, fallback=8):
    """
    Items per sample = count(P >= thr), clipped to 1..12. thr is the presence threshold in
    model.calibration_ / model.calibration if present, else 0.35 (MultiMealPresenceOnlyCVAE has
    neither attribute). Returns `fallback` per row if P is not a 2-D array.
    """
    thr = 0.35
    try:
        cal = getattr(model, "calibration_", None) or getattr(model, "calibration", None) or {}
        t = cal.get("presence_threshold_mean", cal.get("presence_threshold", None))
        if isinstance(t, dict):
            vals = [float(v) for v in t.values() if v is not None]
            if len(vals): thr = float(np.median(vals))
        elif t is not None:
            thr = float(t)
    except Exception:
        pass

    if isinstance(P, np.ndarray) and P.ndim == 2:
        n = (P >= float(thr)).sum(axis=1)
        n = np.clip(n, 1, 12).astype(int)
        return n
    return np.full((P.shape[0] if hasattr(P, "shape") else 1,), int(fallback), dtype=int)

def write_conditioned_samples_with_model(
    model,
    meal_type_mapping: dict,
    pair_table: dict,
    feature_names: list,
    outdir: str,
    n_per: int = 5,
    seed: int = 42,
    temp: float = 1.0,
    nucleus_p: float = 1.0,
    topk_for_md: int = 12,
    X_train: np.ndarray = None,
    C_train: np.ndarray = None,
    M_train: np.ndarray = None,
):
    """
    Write n_per sample meals per (meal, cluster) pair to outdir.

    For each pair: z ~ N(0, I) (seeded), presence probabilities decoded from z with
    _decode_presence_probs_from_z (model calibration applied), then temperature/nucleus, then
    model.diversify_top_n keeps n items per sample (n from the training meals if
    X_train/C_train/M_train are given, else from _expected_n_from_calibration_or_probs).
    Writes conditioned_samples.jsonl, .csv, .parquet (if supported) and .md, with columns
    pair, meal, meal_name, cluster_id, cluster_name, sample_idx, items, items_str,
    item_names, item_names_str.
    """
    import os, json, numpy as np, pandas as pd
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # build reverse meal map
    idx_to_meal = {v: k for k, v in meal_type_mapping.items()}

    # Load food name mapping
    food_name_mapping = _load_food_name_mapping()

    rows = []
    md_lines = ["# Conditioned Samples\n"]

    # group pairs by meal for stable iteration
    pairs_sorted = sorted(pair_table.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    for pair, (meal_idx, orig_cid) in pairs_sorted:
        meal_name = idx_to_meal.get(meal_idx, str(meal_idx))
        print(f"[samples] pair={pair:>3} → meal_idx={meal_idx} ({meal_name}), cluster_id={orig_cid}")

        # condition tensors
        C = np.full((n_per,), pair, dtype=np.int32)   # pair code if your model expects C as "pair"
        # Some models expect separate (C, M). We pass both to be safe:
        M = np.full((n_per,), meal_idx, dtype=np.int32)
        # z ~ N(0, I)
        zdim = _latent_dim_from_model(model, default=64)
        z = rng.standard_normal(size=(n_per, zdim)).astype("float32")

        # Sanity check against decoder input
        try:
            dec_shapes = getattr(model.decoder, "input_shape", None)
            if dec_shapes:
                shapes = dec_shapes if isinstance(dec_shapes, (list, tuple)) else [dec_shapes]
                z_dims = [s[1] for s in shapes if isinstance(s, tuple) and len(s) >= 2 and isinstance(s[1], int)]
                if z_dims and zdim not in z_dims:
                    print(f"[warn] zdim={zdim} not in decoder inputs {z_dims}; using {max(z_dims)}")
                    zdim = max(z_dims)
                    z = rng.standard_normal(size=(n_per, zdim)).astype("float32")
        except Exception:
            pass

        # decode presence probs robustly (handles logits/probs + gating + calibration)
        P = _decode_presence_probs_from_z(
            model, z, C, M,
            apply_hard_masking=True,
            X_train=X_train, C_train=C_train, M_train=M_train
        )
        # temperature / top-p on probs
        P = _apply_temp_and_nucleus(P, temp=temp, top_p=nucleus_p)

        # Prefer train-derived per-pair counts; else use calibration/probs
        if X_train is not None and C_train is not None and M_train is not None:
            expected_counts = _get_pair_expected_count(model, C, M, X_train, C_train, M_train)
            n_list = np.clip(expected_counts, 1, 12).astype(int)
        else:
            n_list = _expected_n_from_calibration_or_probs(model, P)
        
        # Diversify and cap beverages
        binary_sel = []
        debug_diversify = os.getenv("DEBUG_DIVERSIFY", "0") == "1"
        
        for i in range(P.shape[0]):
            n_i = int(n_list[i])
            sel_i = model.diversify_top_n(
                P[i], n_i,
                feature_names=feature_names,
                max_beverages=1,               # tweakable
                diversity_lambda=1,          # tweakable
                seed=seed + pair + i,
                food_name_mapping=food_name_mapping,
                debug=debug_diversify
            )
            binary_sel.append(sel_i)
        
        present_mask = np.stack(binary_sel, axis=0).astype(bool)  # shape [N, D], 0/1

        # Build rows (and lightweight MD)
        for i in range(n_per):
            items = [feature_names[j] for j in np.flatnonzero(present_mask[i])]
            # Convert food codes to names
            item_names = _convert_food_codes_to_names(items, food_name_mapping)
            # Get cluster name
            cluster_name = _get_cluster_name(meal_name, orig_cid)
            
            rows.append({
                "pair": int(pair),
                "meal": int(meal_idx),
                "meal_name": meal_name,
                "cluster_id": int(orig_cid),
                "cluster_name": cluster_name,            # human-readable cluster name
                "sample_idx": int(i),
                "items": items,                          # list in JSONL/Parquet
                "items_str": ";".join(map(str, items)),  # CSV-friendly
                "item_names": item_names,                # human-readable names
                "item_names_str": ";".join(item_names),  # CSV-friendly names
            })

            # pretty MD (top-K by prob for human inspection only)
            order = np.argsort(-P[i])
            topk_names = [f"{_convert_food_codes_to_names([feature_names[j]], food_name_mapping)[0]} ({P[i, j]:.2f})"
                         for j in order[:int(topk_for_md)]]
            
            md_lines += [
                f"## {meal_name} — {cluster_name} (cluster {orig_cid}) — sample {i}",
                f"- items: {', '.join(items) if items else '(none)'}",
                f"- item_names: {', '.join(item_names) if item_names else '(none)'}",
                f"- top{topk_for_md}: {', '.join(topk_names)}",
                ""
            ]

    df = pd.DataFrame(rows)

    # Write JSONL, CSV, Parquet, MD
    js_path = os.path.join(outdir, "conditioned_samples.jsonl")
    with open(js_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    csv_path = os.path.join(outdir, "conditioned_samples.csv")
    df.to_csv(csv_path, index=False)
    try:
        pq_path = os.path.join(outdir, "conditioned_samples.parquet")
        df.to_parquet(pq_path, index=False)
    except Exception:
        pq_path = None
    md_path = os.path.join(outdir, "conditioned_samples.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"📝 Sanity samples → {js_path}")
    print(f"📝 Sanity samples → {csv_path}")
    if pq_path:
        print(f"📝 Sanity samples → {pq_path}")
    print(f"📝 Sanity samples → {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Meal CVAE Training")
    parser.add_argument("--config", type=str, default="cvae_multi_meal.json")
    parser.add_argument("--meals", nargs="*", default=None)
    parser.add_argument("--outdir", type=str, default="outputs/runs/eval_run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--cv_folds", type=int, default=0)
    parser.add_argument("--cluster_filter", type=str, default="cluster_filter_default.json")
    parser.add_argument("--invoke_only", action="store_true", default=False,
                        help="Skip training. Load a saved model and (optionally) generate sanity samples + run analysis.")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory of a saved model when --invoke_only is set.")
    parser.add_argument("--sanity_n", type=int, default=1000,
                        help="# samples per (meal,cluster) for sanity checking.")
    parser.add_argument("--rep_threshold", type=float, default=0.50,
                        help="Consensus threshold (0-1) for representative sets.")
    parser.add_argument("--dup_flag_threshold", type=float, default=0.90,
                        help="Duplicate-rate threshold (0-1) to flag in report.")
    parser.add_argument("--no_analysis", action="store_true",
                        help="Skip conditioned-samples analysis step.")
    parser.add_argument("--analysis_presence_eps", type=str, default="auto",
                        help="Presence binarization for the --invoke_only analysis: 'auto' (recommended), or a float (e.g., 0.35 for probs, 10 for grams). "
                             "After training, the analysis uses the config's presence_eps instead.")
    parser.add_argument("--stoplist", type=str, default="",
                        help="Comma-separated item codes OR a file path (one code per line) to exclude from diversity diagnostics only (--invoke_only analysis).")
    parser.add_argument("--sanity_topk_for_md", type=int, default=12,
                        help="Top-K to show in the human-readable MD file (NOT used for diversity metrics).")
    parser.add_argument("--sanity_temp", type=float, default=1.0,
                        help="Temperature for presence probabilities before the top-n item selection (1.0 = none).")
    parser.add_argument("--sanity_nucleus_p", type=float, default=1.0,
                        help="Nucleus top-p applied to presence probs before the top-n item selection (1.0 = off).")
    parser.add_argument("--no_cal", action="store_true", default=True,
                        help="Disable presence calibration (identity) for sampling & analysis (always on: the flag defaults to True).")
    parser.add_argument("--divergence", choices=["js", "kl", "skl"], default="kl",
                        help="Divergence recorded as divergence_value_for_plot in the real-vs-generated summary: js, kl (P||Q), or skl (0.5*(KL(P||Q)+KL(Q||P))). "
                             "The overview plot uses JS.")

    args = parser.parse_args()

    print("🍽️  Multi-Meal CVAE")
    print("   (Presence-only mode - no grams prediction)")
    # initialize file logger at logs/cvae_train/cvae_train.log and mirror stdout/stderr to it
    try:
        from pathlib import Path as _P
        _log_path = _P.cwd() / 'logs' / 'cvae_train' / 'cvae_train.log'
        os.makedirs(_log_path.parent, exist_ok=True)
        logger = _setup_file_logger('cvae_train', str(_log_path))
        sys.stdout = _StreamToLogger(logger, logging.INFO)
        sys.stderr = _StreamToLogger(logger, logging.ERROR)
    except Exception:
        logger = logging.getLogger(__name__)
    set_global_seed(args.seed)

    import json
    # Config + meals (lightweight)
    with open(args.config, "r") as f:
        cfg_json = json.load(f)
    cm = cfg_json.get("cvae_model", {})
    meal_types = args.meals if args.meals else cm.get("meals", ["Breakfast", "Lunch", "Dinner"])

    # ---------- Invoke-only: no heavy data loading ----------
    if args.invoke_only:
        default_model_dir = os.path.join(args.outdir, "presence_only_model")
        model_dir = args.model_dir or default_model_dir
        print(f"\n🧪 Invoke-only mode. Loading model from: {model_dir}")
        model = _try_model_load(model_dir)

        # Prefer CV aggregate; else fall back to any single-run file
        cands = [os.path.join(args.outdir, "cv", "frozen_calibration.json")]
        if model_dir:  # model_dir = args.model_dir or default_model_dir
            cands.append(os.path.join(model_dir, "calibration.json"))
        cands.append(os.path.join(args.outdir, "calibration.json"))
        loaded = False
        for cp in cands:
            if os.path.exists(cp):
                with open(cp, "r") as f:
                    model.import_calibration(json.load(f), freeze=True)
                print(f"[calibration] Loaded {cp}")
                loaded = True
                break
        if not loaded:
            # freeze whatever is in-memory so decoding uses it
            model.import_calibration(model.export_calibration(), freeze=True)
            print("[calibration] No file found; using in-model calibration.")

        calib_path = os.path.join(args.outdir, "cv", "frozen_calibration.json")
        if os.path.exists(calib_path):
            with open(calib_path, "r") as f:
                frozen = json.load(f)
            model.import_calibration(frozen, freeze=True)
            print(f"[calibration] Loaded frozen calibration from {calib_path}")
        else:
            # Fall back: use whatever the model currently has and freeze it
            model.import_calibration(model.export_calibration(), freeze=True)
            print("[calibration] No frozen file found; freezing in-model calibration.")

        if args.no_cal:
            # 1) Make presence calibration a no-op on logits
            if hasattr(model, "_apply_all_presence_cal"):
                model._apply_all_presence_cal = lambda logits, C, M: logits
            # 2) Freeze an identity/neutral calibration so anything reading from the model sees identity
            model.import_calibration({"pres_logit_offset": 0.0, "pres_temp": 1.0}, freeze=True)
            print("[calibration] Disabled (identity) for this run.")

        # Try to get artifacts for sampling
        # 1) pair_table (required)
        pair_path = os.path.join(args.outdir, "pair_table.json")
        if not os.path.exists(pair_path):
            raise RuntimeError(f"Missing {pair_path}. Needed to map pair codes to (meal, cluster).")
        with open(pair_path, "r") as f:
            _pt = json.load(f)
        pair_table = {int(k): (int(v[0]), int(v[1])) for k, v in _pt.items()}

        # 2) meal mapping (from config is fine)
        meal_type_mapping = {m: i for i, m in enumerate(meal_types)}

        # 3) feature names (from model or sidecar)
        feature_names = _get_feature_names(model, model_dir)

        # Generate sanity samples, then analyse them (the analysis is skipped if sampling fails)
        try:
            sanity_dir = os.path.join(args.outdir, "sanity_samples")
            write_conditioned_samples_with_model(
                model,
                meal_type_mapping=meal_type_mapping,
                pair_table=pair_table,
                feature_names=feature_names,
                outdir=sanity_dir,
                n_per=args.sanity_n,
                seed=args.seed,
                temp=args.sanity_temp,
                nucleus_p=args.sanity_nucleus_p,
                topk_for_md=args.sanity_topk_for_md,
            )
            if not args.no_analysis:
                stopset = _read_stoplist(args.stoplist)
                maybe_analyze_sanity_samples(
                    outdir=args.outdir,
                    presence_eps=args.analysis_presence_eps,
                    rep_threshold=args.rep_threshold,
                    dup_flag_threshold=args.dup_flag_threshold,
                    pair_table=pair_table,
                    stoplist=stopset,
                    divergence=args.divergence,
                )
                # Generate diversity plot if monitoring data exists
                maybe_generate_diversity_plot(outdir=args.outdir)

        except Exception as e:
            print(f"[invoke] Sample generation failed (continuing to analysis if files exist): {e}")
        print("\n✅ Invoke-only path complete.")
        return

    # ---------- Training path (load data only when needed) ----------
    print(f"Training on meal types: {meal_types}")
    X_all, C_all, M_all, feature_names, meal_type_mapping, pair_table, orig_cluster_ids = load_all_meal_data(meal_types)
    if X_all is None:
        print("❌ Failed to load any data!")
        return

    # Optional cluster filter
    meal_cluster_filter = _parse_cluster_filter(meal_type_mapping, args.cluster_filter)
    if meal_cluster_filter:
        print(f"🔎 Applying cluster filter: { {k: sorted(list(v)) for k,v in meal_cluster_filter.items()} }")
        X_all, C_all, M_all, orig_cluster_ids = _apply_cluster_filter(X_all, C_all, M_all, orig_cluster_ids, meal_cluster_filter)
        if len(X_all) == 0:
            print("❌ Cluster filter removed all rows. Nothing to train.")
            return

    print(f"  After filter → samples: {len(X_all)}, features: {X_all.shape[1]}, "
          f"pairs: {len(np.unique(C_all))}, meals: {len(np.unique(M_all))}")

    # Densify pair codes & update lookup
    C_all, pair_table, _ = _reindex_pairs_and_table(C_all, pair_table)
    # Save pair table for invoke-only runs
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "pair_table.json"), "w") as f:
        json.dump({int(k): [int(v[0]), int(v[1])] for k, v in pair_table.items()}, f, indent=2)

    # Build config for model
    config = MultiMealCVAEConfig({"cvae_model": cm})
    config.input_dim = X_all.shape[1]
    config.num_clusters = len(np.unique(C_all))
    config.meal_types = meal_types
    if "cluster_embedding_dim" in cm: config.cluster_embedding_dim = cm["cluster_embedding_dim"]
    if "meal_embedding_dim" in cm:    config.meal_embedding_dim = cm["meal_embedding_dim"]

    raw_config = {
        "cvae_model": {
            "presence_pos_weight": config.presence_pos_weight,
            "presence_eps": config.presence_eps,
            "kl_free_bits": 0.0,
            "kl_warmup_epochs": getattr(config, "kl_warmup_epochs", 10),
            "kl_annealing_epochs": getattr(config, "kl_annealing_epochs", getattr(config, "kl_warmup_epochs", 10)),
            "kl_weight": 0.01,
            "cluster_embedding_dim": 8,
            "meal_embedding_dim": 8,
            "focal_gamma": config.focal_gamma,
            "learning_rate": config.learning_rate,
            "cond_dim": 256,
            "film_blocks": 3,
            "lambda_ac": 0.9,
            "guidance_drop_prob": 0.2,
            "presence_pos_weight_max": 15,
            "ac_detach": False
        },
        "pair_table": pair_table,
        # Diversity monitor settings from the JSON config
        "monitor": cfg_json.get("monitor", {}),
    }

    # Splits
    epochs   = args.epochs if args.epochs is not None else config.epochs
    batch_sz = args.batch_size if args.batch_size is not None else config.batch_size
    patience = args.patience if args.patience is not None else getattr(config, "patience", 15)

    test_size_ho = 0.10
    stratify_arg = C_all if _can_stratify(C_all, test_size_ho) else None
    if stratify_arg is None:
        print("⚠️  Falling back to non-stratified holdout split due to rare classes.")
    X_dev, X_ho, C_dev, C_ho, M_dev, M_ho = train_test_split(
        X_all, C_all, M_all, test_size=test_size_ho, random_state=42, stratify=stratify_arg
    )

    def _final_train_eval(X_tr, C_tr, M_tr, X_val, C_val, M_val):
        model, train_time = train_multi_meal_model(
            X_tr, C_tr, M_tr, X_val, C_val, M_val,
            config, raw_config, logger, epochs, batch_sz, patience, outdir=args.outdir
        )
        # === save model + calibration ===
        model_dir = os.path.join(args.outdir, "model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Debug: Check if model files exist before saving
        model_files = ["cvae.keras", "decoder.keras", "encoder.keras", "calibration.json"]
        print(f"[debug] Model saving to: {model_dir}")
        for fname in model_files:
            fpath = os.path.join(model_dir, fname)
            if os.path.exists(fpath):
                old_size = os.path.getsize(fpath)
                old_time = os.path.getmtime(fpath)
                print(f"[debug] {fname} exists: {old_size} bytes, modified: {datetime.fromtimestamp(old_time).strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"[debug] {fname} does not exist")
        
        # Save the model
        print(f"[debug] Starting model.save() at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        model.save(model_dir)  # writes encoder/decoder/cvae + calibration.json
        print(f"[debug] model.save() completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Debug: Verify files were actually saved/updated
        for fname in model_files:
            fpath = os.path.join(model_dir, fname)
            if os.path.exists(fpath):
                new_size = os.path.getsize(fpath)
                new_time = os.path.getmtime(fpath)
                print(f"[debug] {fname} after save: {new_size} bytes, modified: {datetime.fromtimestamp(new_time).strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"[debug] ERROR: {fname} missing after save!")

        # Also write it to {outdir}/cv/frozen_calibration.json (read by --invoke_only); this
        # replaces the cross-validation aggregate written by run_cross_validation
        os.makedirs(os.path.join(args.outdir, "cv"), exist_ok=True)
        calib_path = os.path.join(args.outdir, "cv", "frozen_calibration.json")
        print(f"[debug] Saving calibration to: {calib_path}")
        with open(calib_path, "w") as f:
            json.dump(model.export_calibration(), f, indent=2)
        print(f"[debug] Calibration saved: {os.path.getsize(calib_path)} bytes")
        
        print(f"[calibration] Saved to: {os.path.join(model_dir,'calibration.json')} "
            f"and {os.path.join(args.outdir,'cv','frozen_calibration.json')}")
        print(f"✅ Model successfully saved to: {model_dir}")

        # Tiny-val evaluation (and plots)
        val_results = evaluate_by_meal_type(model, X_val, C_val, M_val, meal_type_mapping, config, logger)
        create_comprehensive_visualizations(
            model, val_results, X_val, C_val, M_val, meal_type_mapping, config,
            output_dir=os.path.join(args.outdir, "eval_panels", "val_tiny")
        )
        # Holdout (never recalibrates)
        ho_results = evaluate_by_meal_type(model, X_ho, C_ho, M_ho, meal_type_mapping, config, logger)
        create_comprehensive_visualizations(
            model, ho_results, X_ho, C_ho, M_ho, meal_type_mapping, config,
            output_dir=os.path.join(args.outdir, "eval_panels", "holdout")
        )
        
        # Print pair evaluation summaries for easy comparison
        print_pair_evaluation_summary(val_results, "Validation Set")
        print_pair_evaluation_summary(ho_results, "Holdout Set")
        
        return model, train_time, val_results, ho_results

    # Optional CV (for calibration freezing), then final train
    if args.cv_folds and args.cv_folds > 1:
        print(f"\n📚 Running {args.cv_folds}-fold cross-validation on the dev set...")
        _ = run_cross_validation(
            X_dev, C_dev, M_dev,
            folds=args.cv_folds,
            config=config, raw_config=raw_config, logger=logger,
            epochs=epochs, batch_size=batch_sz, patience=patience,
            meal_type_mapping=meal_type_mapping, outdir=args.outdir
        )

        # Final on full dev with tiny internal val
        print("\n🧪 Training final model on full dev (with tiny internal val) and evaluating on holdout...")
        tiny_val_size = 0.10
        stratify_tiny = C_dev if _can_stratify(C_dev, tiny_val_size) else None
        if stratify_tiny is None:
            print("⚠️  Falling back to non-stratified tiny-val split.")
        X_tr_full, X_val_tiny, C_tr_full, C_val_tiny, M_tr_full, M_val_tiny = train_test_split(
            X_dev, C_dev, M_dev, test_size=tiny_val_size, random_state=42, stratify=stratify_tiny
        )

        model, train_time, val_results, ho_results = _final_train_eval(
            X_tr_full, C_tr_full, M_tr_full, X_val_tiny, C_val_tiny, M_val_tiny
        )

        # Re-import cv/frozen_calibration.json (by now the final model's own calibration)
        calib_path = os.path.join(args.outdir, "cv", "frozen_calibration.json")
        if os.path.exists(calib_path):
            with open(calib_path, "r") as f:
                frozen = json.load(f)
            model.import_calibration(frozen, freeze=True)
            print(f"[calibration] Using frozen calibration from {calib_path}")

    else:
        # Final on dev with tiny internal val
        print("\n🧪 Training final model on full dev (with tiny internal val) and evaluating on holdout...")
        tiny_val_size = 0.10
        stratify_tiny = C_dev if _can_stratify(C_dev, tiny_val_size) else None
        if stratify_tiny is None:
            print("⚠️  Falling back to non-stratified tiny-val split.")
        X_tr_full, X_val_tiny, C_tr_full, C_val_tiny, M_tr_full, M_val_tiny = train_test_split(
            X_dev, C_dev, M_dev, test_size=tiny_val_size, random_state=42, stratify=stratify_tiny
        )
        model, train_time, val_results, ho_results = _final_train_eval(
            X_tr_full, C_tr_full, M_tr_full, X_val_tiny, C_val_tiny, M_val_tiny
        )

    # --------- Common finalize & reporting ---------
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "pair_table.json"), "w") as f:
        json.dump({int(k): [int(v[0]), int(v[1])] for k, v in pair_table.items()}, f, indent=2)
    print(f"[pairs] Saved pair_table → {os.path.join(args.outdir, 'pair_table.json')}")

    model_dir = os.path.join(args.outdir, "presence_only_model")
    _save_model_robust(model, model_dir, feature_names, logger)

    if args.no_cal:
        model.import_calibration({"pres_logit_offset": 0.0, "pres_temp": 1.0}, freeze=True)
        print("[calibration] Disabled (identity) for this run.")
    else:
        model.import_calibration(model.export_calibration(), freeze=True)
        print("[calibration] Using model’s current calibration (frozen) for generation.")

    sanity_dir = os.path.join(args.outdir, "sanity_samples")
    write_conditioned_samples_with_model(
        model,
        meal_type_mapping=meal_type_mapping,
        pair_table=pair_table,
        feature_names=feature_names,
        outdir=sanity_dir,
        n_per=args.sanity_n,
        seed=args.seed,
        temp=args.sanity_temp,
        nucleus_p=args.sanity_nucleus_p,
        topk_for_md=args.sanity_topk_for_md,
        # X_train/C_train/M_train not passed: items per sample come from
        # _expected_n_from_calibration_or_probs
    )
    if not args.no_analysis:
        maybe_analyze_sanity_samples(
            outdir=args.outdir,
            presence_eps=getattr(config, "presence_eps", 0.0),
            rep_threshold=args.rep_threshold,
            dup_flag_threshold=args.dup_flag_threshold,
            divergence=args.divergence
        )
        # Generate diversity plot if monitoring data exists
        maybe_generate_diversity_plot(outdir=args.outdir)

    # Save results CSV
    def _flatten(results, split_name):
        rows = []
        for meal_type, res in results.items():
            row = {"split": split_name, "Meal": meal_type,
                   "n_samples": res['n_samples'], "n_clusters": res['n_clusters']}
            row.update(res['metrics'])
            rows.append(row)
        return rows

    df_rows = _flatten(val_results, "val_tiny") + _flatten(ho_results, "holdout")
    df = pd.DataFrame(df_rows)
    results_path = os.path.join(args.outdir, "multi_meal_results.csv")
    df.to_csv(results_path, index=False)
    print(f"✅ Results saved to: {results_path}")

    # Pretty summary
    print("\n📈 Multi-Meal CVAE Results Summary:")
    has_r2_present = 'r2_conditional_present_item' in df.columns and not df['r2_conditional_present_item'].isna().all()
    key_cols = ([
        "Meal", "n_samples", "n_clusters",
        "r2_conditional_present_item", "r2_count",
        "f1_micro", "f1_macro", "auroc_presence", "auprc_presence",
        "mae_count", "rmse_count", "count_bias"
    ] if has_r2_present else [
        "Meal", "n_samples", "n_clusters",
        "f1_micro", "f1_macro", "auroc_presence", "auprc_presence",
        "brier_score", "ece", "mae_count", "rmse_count", "count_bias"
    ])
    show_cols = [c for c in key_cols if c in df.columns]
    if len(show_cols) > 1:
        _print_pretty_table(df[["split"] + show_cols])

    avg_r2_present = df.get('r2_conditional_present_item', pd.Series(dtype=float)).mean()
    if pd.isna(avg_r2_present):
        avg_r2_present = df.get('r2_count', pd.Series(dtype=float)).mean()

    overall_metrics = {
        "Total samples": X_all.shape[0],
        "Total features": X_all.shape[1],
        "Total clusters": len(np.unique(C_all)),
        "Meal types": len(meal_types),
        "Training time (s)": train_time,
        "Avg R² present": avg_r2_present,
        "Avg F1 micro": df.get('f1_micro', pd.Series(dtype=float)).mean(),
        "Avg AUROC": df.get('auroc_presence', pd.Series(dtype=float)).mean(),
    }
    print("\n🎯 Overall Performance:")
    for k, v in overall_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n🎉 Training pipeline complete.")

if __name__ == "__main__":
    main()

