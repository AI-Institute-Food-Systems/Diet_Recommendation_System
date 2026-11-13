#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diversity and fidelity monitor for CVAE training (a Keras callback).

DiversityFidelityMonitor samples meals from the model during training and compares them
with the validation meals of each (meal, cluster) pair: JS divergence, Pearson correlation,
entropy ratio, coverage and self-Jaccard distance. It writes one CSV row per pair and epoch
(diversity_monitor.csv) and can optionally early-stop on a composite score.

MultiMealPresenceOnlyCVAE.fit adds it when the config's "monitor.enabled" is true, as in
config/cvae_multi_meal.json (Makefile target `cvae_train`).
"""

import os
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr


class DiversityFidelityMonitor(tf.keras.callbacks.Callback):
    """
    Every `every` epochs:
      • Samples G meals per (meal,cluster) from the current model
      • Compares generated vs real distributions on:
          - JS divergence (lower=better)
          - Pearson correlation (higher=better)
          - Entropy ratio exp(H_gen)/exp(H_real)  (≈1.0 ideal)
          - Coverage@N (unique items in N gen / unique items in N real)
          - Self-Jaccard distance among generated meals (↑ means more variety)
      • Logs per-(meal,cluster) rows + medians to CSV
      • (optional) Early-stops on a composite score that balances fidelity & diversity
    """
    def __init__(
        self, parent, Xv, Cv, Mv,
        pres_eps=1e-6, G=1000, every=2, seed=0,
        outdir=None, csv_name="diversity_monitor.csv",
        bootstraps=0,  # real-vs-real baseline reps (0 = skip)
        cov_N=1000,
        early_stop=False, patience=8, min_delta=2e-3
    ):
        super().__init__()
        self.p = parent
        self.Xv = np.asarray(Xv, np.float32)
        self.Cv = np.asarray(Cv, np.int32).ravel()
        self.Mv = np.asarray(Mv, np.int32).ravel()
        self.pres_eps = float(pres_eps)
        self.G = int(G)
        self.every = int(every)
        self.rng = np.random.default_rng(seed)
        self.cov_N = int(cov_N)
        self.outdir = outdir or "."
        os.makedirs(self.outdir, exist_ok=True)
        self.csv_path = os.path.join(self.outdir, csv_name)
        self.epoch_rows = []
        self.bootstraps = int(bootstraps)
        self.early_stop = bool(early_stop)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_score = -np.inf
        self.best_weights = None
        self.bad_count = 0

        # pre-index validation by (meal,cluster)
        self.groups = {}
        for m in np.unique(self.Mv):
            for c in np.unique(self.Cv[self.Mv == m]):
                idx = np.where((self.Mv == m) & (self.Cv == c))[0]
                if idx.size > 0:
                    self.groups[(int(m), int(c))] = idx

        # optional real-vs-real baselines
        self.rvrl_js_q = None
        self.rvrl_pr_q = None
        if self.bootstraps > 0:
            js_vals, pr_vals = [], []
            for (m, c), idx in self.groups.items():
                X = (self.Xv[idx] > self.pres_eps).astype(np.int8)
                if len(X) < 80:
                    continue
                B = min(50, self.bootstraps)
                for _ in range(B):
                    self.rng.shuffle(X, axis=0)
                    a = X[: len(X)//2]
                    b = X[len(X)//2 :]
                    p = self._freq(a.sum(axis=0))
                    q = self._freq(b.sum(axis=0))
                    js_vals.append(self._js(p, q))
                    pr_vals.append(self._pear(p, q))
            if js_vals:
                self.rvrl_js_q = float(np.quantile(js_vals, 0.90))
            if pr_vals:
                self.rvrl_pr_q = float(np.quantile(pr_vals, 0.10))

    # ---------- small helpers ----------
    @staticmethod
    def _freq(counts):
        counts = counts.astype(np.float64)
        s = counts.sum()
        if s <= 0:
            return np.full_like(counts, 1.0/len(counts))
        p = np.clip(counts / s, 1e-12, 1.0)
        p /= p.sum()
        return p

    @staticmethod
    def _js(p, q):  # divergence in bits (bounded [0,1])
        return float(jensenshannon(p, q, base=2.0)**2)

    @staticmethod
    def _pear(p, q):
        if p.std() == 0 or q.std() == 0:
            return 0.0
        return float(pearsonr(p, q).statistic)

    @staticmethod
    def _eff_species_num(p):
        return float(np.exp(-(p * np.log(p)).sum()))

    def _coverage_at_n(self, real_sets, gen_sets, n):
        r = set().union(*real_sets[: min(n, len(real_sets))])
        g = set().union(*gen_sets[: min(n, len(gen_sets))])
        return len(g) / max(1, len(r))

    def _self_jaccard_distance(self, gen_sets, pairs=2000):
        m = len(gen_sets)
        if m < 2:
            return 0.0
        idx = self.rng.integers(0, m, size=(min(pairs, m*m), 2))
        vals = []
        for i, j in idx:
            A, B = gen_sets[i], gen_sets[j]
            u = len(A | B)
            if u > 0: vals.append(1 - len(A & B) / u)
        return float(np.mean(vals)) if vals else 0.0

    # ---------- core logic ----------
    def on_epoch_end(self, epoch, logs=None):
        # only run every K epochs
        if (epoch % self.every) != (self.every - 1):
            return

        per_group = []
        for (m, c), idx in self.groups.items():
            X_real = (self.Xv[idx] > self.pres_eps).astype(np.int8)
            n_real = len(X_real)
            if n_real == 0:
                continue

            # match sample size
            G = min(self.G, n_real)

            # sample z ~ N(0, I); decode conditioned on (m,c)
            z = self.rng.normal(size=(G, self.p._latent_dim)).astype(np.float32)
            C = np.full(G, c, np.int32)
            M = np.full(G, m, np.int32)
            P = self.p.decode_from_z(z, C, M)  # probs (G, D)
            X_gen = self.p.convert_probabilities_to_binary(P, method="bernoulli", seed=int(self.rng.integers(0, 1<<31)))

            # take G from real to match
            ridx = self.rng.choice(n_real, G, replace=False)
            R = X_real[ridx]

            # frequency vectors
            # Convert TensorFlow tensors to numpy if needed
            if hasattr(R, 'numpy'):
                R = R.numpy()
            if hasattr(X_gen, 'numpy'):
                X_gen = X_gen.numpy()
            
            p = self._freq(R.sum(axis=0))
            q = self._freq(X_gen.sum(axis=0))

            js = self._js(p, q)
            pr = self._pear(p, q)
            er = self._eff_species_num(q) / max(1e-12, self._eff_species_num(p))
            # coverage + self-jaccard
            real_sets = [set(np.where(r > 0)[0].tolist()) for r in R]
            gen_sets  = [set(np.where(g > 0)[0].tolist()) for g in X_gen]
            cov = self._coverage_at_n(real_sets, gen_sets, n=min(self.cov_N, G))
            sjd = self._self_jaccard_distance(gen_sets)

            per_group.append({
                "epoch": epoch + 1,
                "meal": int(m),
                "cluster": int(c),
                "n_real": int(n_real),
                "G": int(G),
                "js": js,
                "pearson": pr,
                "entropy_ratio": er,
                "coverage": cov,
                "self_jaccard": sjd,
            })

        if not per_group:
            return

        # aggregate (medians are robust)
        med_js  = float(np.median([r["js"] for r in per_group]))
        med_pr  = float(np.median([r["pearson"] for r in per_group]))
        med_er  = float(np.median([r["entropy_ratio"] for r in per_group]))
        med_cov = float(np.median([r["coverage"] for r in per_group]))
        med_sjd = float(np.median([r["self_jaccard"] for r in per_group]))

        # composite score (you can tweak weights)
        score = (-med_js) + 0.5*med_pr - 0.3*abs(1.0 - med_er) + 0.2*med_cov

        # print a concise line
        rvrl_info = ""
        if self.rvrl_js_q is not None and self.rvrl_pr_q is not None:
            rvrl_info = f"  [rvrl q90 JS≈{self.rvrl_js_q:.3f}, q10 r≈{self.rvrl_pr_q:.3f}]"
        print(f"  [monitor] medJS={med_js:.3f}  medr={med_pr:.3f}  ER={med_er:.2f}  Cov={med_cov:.2f}  SJD={med_sjd:.2f}  score={score:.3f}{rvrl_info}")

        # write CSV (append)
        # per-group rows + one aggregate row per epoch
        for r in per_group:
            self.epoch_rows.append({**r, "med_js": med_js, "med_pearson": med_pr,
                                    "med_entropy_ratio": med_er, "med_coverage": med_cov,
                                    "med_self_jaccard": med_sjd, "score": score})
        try:
            import pandas as pd
            pd.DataFrame(self.epoch_rows).to_csv(self.csv_path, index=False)
        except Exception:
            # no pandas? write minimal CSV
            import csv
            keys = list(self.epoch_rows[0].keys())
            write_header = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                if write_header: w.writeheader()
                for row in self.epoch_rows[-len(per_group):]:
                    w.writerow(row)

        # optional early stop on composite
        if self.early_stop:
            improved = (score > self.best_score + self.min_delta)
            if improved:
                self.best_score = score
                self.bad_count = 0
                self.best_weights = self.model.get_weights()
            else:
                self.bad_count += 1
                if self.bad_count >= self.patience:
                    print("  [monitor] early-stop on composite score. Restoring best weights.")
                    if self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
                    self.model.stop_training = True
