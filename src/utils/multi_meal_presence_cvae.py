#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-meal presence-only CVAE: the generative model of the pipeline.

MultiMealPresenceOnlyCVAE models which foods appear in a meal (presence/absence, no grams),
conditioned on a (meal type, cluster) pair code and the meal type. The decoder uses FiLM blocks,
classifier-free condition dropout (CondDrop, with a NULL id), a fixed per-pair log-odds prior and
an allowed-food mask built from the training meals. After training it calibrates a temperature
and logit offsets on the validation split. It also provides prediction, evaluation, save/load and
the top-n item selection used to write sample meals.

Trained by cli/run_multi_meal_cvae.py (Makefile target `cvae_train`), which saves it to
<run>/presence_only_model/; cli/evaluate.py loads it from there.
"""

from __future__ import annotations

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional

from sklearn.metrics import average_precision_score

import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.saving import register_keras_serializable as register_keras_serializable_new

# Shared config class and encoder pre-processing layer
from .multi_meal_cvae import (
    MultiMealCVAEConfig,
    Log1pZScore,
)

def _stable_sigmoid(x):
    # clip to avoid overflow in exp
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _unpack_decoder_outputs(dec_out):
    # The decoder returns [p_presence, logits, ac_logits, cluster_id, meal_id]
    return dec_out[0], dec_out[1]


def _safe_save(model, path_stem):
    """Safely save a model, handling any errors"""
    try:
        # Add .keras extension if not present
        if not path_stem.endswith('.keras'):
            path_stem += '.keras'
        
        # Try to save without custom_objects (Keras 3 doesn't support it in save)
        model.save(path_stem)
        return True
    except Exception as e:
        print(f"Warning: Failed to save model to {path_stem}: {e}")
        # Try fallback to SavedModel format
        try:
            alt_path = path_stem.replace('.keras', '_savedmodel')
            model.export(alt_path)
            print(f"Fallback: Saved as SavedModel to {alt_path}")
            return True
        except Exception as e2:
            print(f"Error: Both .keras and SavedModel save failed: {e2}")
            return False


@register_keras_serializable_new(package="multi_meal")
def sampling(args):
    """Reparameterization trick for VAE"""
    z_mean, z_logvar = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_logvar) * epsilon


@register_keras_serializable_new(package="multi_meal")
class FiLM(KL.Layer):
    """Feature-wise Linear Modulation: h -> h * (1 + γ(c)) + β(c)."""
    def __init__(self, dim, name=None, **kwargs):
        # Accept extra Keras-serialization kwargs (e.g., trainable, dtype)
        super().__init__(name=name, **kwargs)
        self.to_gamma = KL.Dense(dim, name=f"{name}_gamma")
        self.to_beta  = KL.Dense(dim, name=f"{name}_beta")

    def call(self, h, c):
        gamma = self.to_gamma(c)             # (B, D)
        beta  = self.to_beta(c)              # (B, D)
        scale = 1.0 + gamma
        return h * scale + beta


class CondDrop(KL.Layer):
    def __init__(self, drop_prob: float, null_id: int, return_mask: bool=False, name=None, **kwargs):
        # Accept extra Keras-serialization kwargs (e.g., trainable, dtype) for safe deserialization
        super().__init__(name=name, **kwargs)
        self.p = drop_prob
        self.null_id = null_id
        self.return_mask = return_mask

    def call(self, c, training=None):
        if self.p <= 0.:
            return (c, tf.zeros_like(c, tf.float32)) if self.return_mask else c

        def train_path():
            rnd  = tf.random.uniform(tf.shape(c), 0., 1.)
            mask = tf.less(rnd, self.p)  # True→drop to NULL
            dropped = tf.where(mask, tf.fill(tf.shape(c), tf.cast(self.null_id, c.dtype)), c)
            return (dropped, tf.cast(mask, tf.float32)) if self.return_mask else dropped

        return tf.__internal__.smart_cond.smart_cond(
            tf.cast(training, tf.bool) if training is not None else tf.constant(False),
            train_path,
            lambda: (c, tf.zeros_like(c, tf.float32)) if self.return_mask else c
        )

@register_keras_serializable_new(package="multi_meal")
class ScalarScale(KL.Layer):
    """Trainable scalar scaling layer for bias heads."""
    def build(self, input_shape):
        self.w = self.add_weight("w", shape=(), initializer="zeros", trainable=True)
    def call(self, x):
        return x * self.w


class NullZeroCallback(tf.keras.callbacks.Callback):
    """Force NULL-id embedding rows to stay exactly zero."""
    def __init__(self, parent):
        super().__init__()
        self.p = parent

    def on_epoch_end(self, epoch, logs=None):
        # Decoder: cluster NULL row
        try:
            null_c = int(self.p.decoder._null_cluster_id)  # set in _build_presence_decoder
            self.p._set_null_zero(self.p.decoder.get_layer("cluster_embedding_dec"), null_c)
        except Exception:
            pass
        # Decoder: meal NULL row
        try:
            null_m = int(self.p.decoder._null_meal_id)     # set in _build_presence_decoder
            self.p._set_null_zero(self.p.decoder.get_layer("meal_type_embedding_dec"), null_m)
        except Exception:
            pass
        # Encoder: cluster NULL row (index = num_clusters)
        try:
            self.p._set_null_zero(self.p.encoder.get_layer("cluster_embedding"),
                                  self.p.config.num_clusters)
        except Exception:
            pass


@register_keras_serializable_new(package="multi_meal")
class PresenceOnlyCVAELossLayer(tf.keras.layers.Layer):
    def __init__(self, raw_config=None, **kwargs):
        super().__init__(**kwargs)
        self.raw_config = raw_config or {}
        cfg = self.raw_config
        c = cfg.get("cvae_model", cfg)

        self.presence_eps   = c.get("presence_eps", 1e-6)
        self.pos_weight     = c.get("presence_pos_weight", 0.4)
        self.pos_weight_max = c.get("presence_pos_weight_max", 4.0)

        # The config "kl_weight" is the target beta, not the multiplier itself
        self.kl_weight_target   = float(c.get("kl_weight", 0.1))
        self.kl_free_bits       = float(c.get("kl_free_bits", 0.0))
        self.kl_annealing       = bool(c.get("kl_annealing", True))
        self.kl_annealing_epochs = int(c.get("kl_annealing_epochs", c.get("kl_warmup_epochs", 10)))
        self.focal_gamma = float(c.get("focal_gamma", 0.0))   
        
        # Multiplier used in the loss: 0.0 at start (the target if kl_annealing is off);
        # KLWeightScheduler sets it each epoch
        self.kl_weight = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="kl_weight")
        if not self.kl_annealing:
            self.kl_weight.assign(self.kl_weight_target)
        
    def call(self, inputs, training=None):
        # Unpack
        presence_logits, z_mean, z_logvar, x_true = inputs

        # -------- Presence target (binarize grams) --------
        y_presence = tf.cast(x_true > self.presence_eps, tf.float32)

        # -------- Dynamic class weight (scalar) --------
        total = tf.cast(tf.size(y_presence), tf.float32)
        pos   = tf.reduce_sum(y_presence)
        neg   = total - pos

        dyn_pw = tf.where(pos > 0.0, neg / (pos + 1e-8), 1.0)
        pos_weight = tf.clip_by_value(dyn_pw * float(self.pos_weight), 1.0, self.pos_weight_max)

        # -------- Presence loss --------
        if self.focal_gamma > 0.0:
            presence_loss_per_elem = self._focal_loss(y_presence, presence_logits, pos_weight)
        else:
            presence_loss_per_elem = tf.nn.weighted_cross_entropy_with_logits(
                labels=y_presence, logits=presence_logits, pos_weight=pos_weight
            )
        presence_loss = tf.reduce_mean(presence_loss_per_elem)

        # -------- KL (per-dimension, with free-bits) --------
        # KL_dim = 0.5 * (exp(logvar) + mu^2 - 1 - logvar)   [B, Z]
        kl_dim = 0.5 * (tf.exp(z_logvar) + tf.square(z_mean) - 1.0 - z_logvar)

        if self.kl_free_bits > 0.0:
            # free-bits per dim (in nats): only penalize the surplus
            fb = tf.convert_to_tensor(self.kl_free_bits, dtype=kl_dim.dtype)
            kl_dim = tf.maximum(kl_dim - fb, 0.0)

        kl_per_sample = tf.reduce_sum(kl_dim, axis=-1)   # [B]
        kl_loss = tf.reduce_mean(kl_per_sample)          # scalar

        # -------- Annealed beta (set by callback) --------
        # self.kl_weight is a tf.Variable that your callback assigns each epoch.
        beta = tf.cast(self.kl_weight, kl_loss.dtype)

        # -------- Count matching loss --------
        p = tf.nn.sigmoid(presence_logits)
        true_count = tf.reduce_sum(tf.cast(x_true > self.presence_eps, p.dtype), axis=1)
        pred_count = tf.reduce_sum(p, axis=1)
        count_loss = tf.reduce_mean(tf.keras.losses.huber(true_count, pred_count, delta=1.0))
        lambda_count = tf.convert_to_tensor(self.raw_config.get("cvae_model", {}).get("lambda_count", 0.1), dtype=tf.float32)

        # -------- Gate penalty: none (the allowed-food mask is applied in the decoder) --------
        gate_penalty = 0.0

        # -------- Total loss --------
        total_loss = presence_loss + beta * kl_loss + lambda_count * count_loss + gate_penalty

        # Wire into Keras
        self.add_loss(total_loss)

        # passthrough (model uses .add_loss)
        return x_true


    def _focal_loss(self, y_true, y_pred_logits, pos_weight):
        """Focal loss implementation for binary classification"""
        # Convert logits to probabilities
        y_pred = tf.nn.sigmoid(y_pred_logits)
        
        # Compute cross-entropy
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred_logits)
        
        # Apply pos_weight
        ce_weighted = tf.where(y_true == 1, ce * pos_weight, ce)
        
        # Compute focal weight: (1 - p)^gamma for positive class, p^gamma for negative class
        p_t = tf.where(y_true == 1, y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - p_t, self.focal_gamma)
        
        # Apply focal weight
        focal_loss = focal_weight * ce_weighted
        
        return focal_loss


class KLWeightScheduler(tf.keras.callbacks.Callback):
    def __init__(self, layer_name="cvae_loss", beta_max=0.1, warmup_epochs=10,
                 cyclical=False, cycle_len=8, floor=0.0):
        super().__init__()
        self.layer_name = layer_name
        self.beta_max = float(beta_max)
        self.warmup_epochs = int(warmup_epochs)
        self.cyclical = bool(cyclical)
        self.cycle_len = int(cycle_len)
        self.floor = float(floor)

    def _beta_at(self, epoch):
        if self.warmup_epochs <= 0:
            return self.beta_max
        if not self.cyclical:
            # linear warmup then hold
            t = min(1.0, max(0.0, epoch / self.warmup_epochs))
            return self.floor + t * (self.beta_max - self.floor)
        # simple triangular cycles after warmup
        if epoch < self.warmup_epochs:
            t = epoch / self.warmup_epochs
            return self.floor + t * (self.beta_max - self.floor)
        # cycles
        phase = (epoch - self.warmup_epochs) % self.cycle_len
        t = 1.0 - abs((phase / (self.cycle_len/2.0)) - 1.0)  # up then down
        return self.floor + t * (self.beta_max - self.floor)

    def on_epoch_begin(self, epoch, logs=None):
        loss_layer = self.model.get_layer(self.layer_name)
        beta = self._beta_at(epoch)
        loss_layer.kl_weight.assign(beta)


class MultiMealPresenceOnlyCVAE:
    """
    Multi-meal CVAE that only predicts presence/absence (see the module docstring).
    Trained with fit(); predict_heads() returns calibrated presence probabilities, and
    evaluate(), export_calibration()/import_calibration() and save()/load() serve the runner
    and the evaluation.
    """
    
    def __init__(self, config: MultiMealCVAEConfig, raw_config: Optional[Dict[str, Any]] = None):
        self.config = config
        self.raw_config = raw_config or {}

        # Build valid candidate pairs per meal from pair_table (if present)
        self._meal_to_pairs = {}
        pt = (self.raw_config or {}).get("pair_table", None)
        if pt:
            for pid_str, (m_idx, _orig_cluster) in pt.items():
                pid = int(pid_str) if isinstance(pid_str, str) else int(pid_str)
                self._meal_to_pairs.setdefault(int(m_idx), []).append(pid)

        # Architecture / conditioning settings from raw_config["cvae_model"]
        cfg = self.raw_config.get("cvae_model", self.raw_config)
        self.cond_dim = int(cfg.get("cond_dim", 128))  # read but not used by the architecture
        self.film_blocks = int(cfg.get("film_blocks", 2))
        self.use_bias_heads = bool(cfg.get("use_bias_heads", False))
        self.lambda_ac = float(cfg.get("lambda_ac", 0.5))  # read but not used (no auxiliary-classifier loss)
        self.guidance_drop_prob = float(cfg.get("guidance_drop_prob", 0.5))  # decoder cluster-condition dropout
        self.meal_guidance_drop_prob = float(cfg.get("meal_guidance_drop_prob", 0.3))  # decoder meal-condition dropout
        self.ac_detach = bool(cfg.get("ac_detach", False))  # read but not used
        self.encoder_guidance_drop_prob = float(cfg.get("encoder_guidance_drop_prob", 1.0))
        # priors + masking
        self.use_pair_prior          = bool(cfg.get("use_pair_prior", True))
        self.pair_prior_scale        = float(cfg.get("pair_prior_scale", 1.0))
        self.use_pair_allowed_mask   = bool(cfg.get("use_pair_allowed_mask", True))
        self.mask_strength           = float(cfg.get("mask_strength", 12.0))
        self.allowed_topk            = cfg.get("allowed_topk", "median")   # int | "median" | "quantile:q"
        self.min_prev                = float(cfg.get("min_prev", 0.005))

        # Optional cost-bin conditioning
        self.use_cost_bin            = bool(cfg.get("use_cost_bin", False))
        self.num_cost_bins           = int(cfg.get("num_cost_bins", 3))
        self._b_emb_dim              = int(cfg.get("cost_embedding_dim", 4))
        self.cost_guidance_drop_prob = float(cfg.get("cost_guidance_drop_prob", 0.3))

        # Robust config fallbacks to avoid attribute errors
        self._meal_types = getattr(self.config, "meal_types", ["Breakfast", "Lunch", "Dinner"])
        self._n_meals = len(self._meal_types)
        self._latent_dim = getattr(self.config, "latent_dim", getattr(self.config, "z_dim", 32))
        self._enc_sizes = getattr(self.config, "encoder_sizes", [512, 512])
        self._dropout = getattr(self.config, "dropout_rate", 0.0)
        self._lr = getattr(self.config, "learning_rate", 1e-3)
        self._pres_eps = getattr(self.config, "presence_eps", 1e-6)
        self._c_emb_dim = getattr(self.config, "cluster_embedding_dim", 32)
        self._m_emb_dim = getattr(self.config, "meal_embedding_dim", 8)
        
        # Build models
        self.encoder = None
        self.decoder = None
        self.cvae = None
        
        # Calibration state
        self.calibration_state = {}
        
        # Only build models if config has valid dimensions
        if config.input_dim > 0 and config.num_clusters > 0:
            self._build_models()
    
    def _build_presence_decoder(self):
        n_foods    = self.config.input_dim
        n_clusters = int(self.config.num_clusters)
        n_meals    = self._n_meals

        # Inputs
        z_in = KL.Input(shape=(self._latent_dim,), name="z_input")
        c_in = KL.Input(shape=(), dtype="int32", name="cluster_input_dec")
        m_in = KL.Input(shape=(), dtype="int32", name="meal_type_input_dec")
        # Optional cost-bin input
        if self.use_cost_bin:
            b_in = KL.Input(shape=(), dtype="int32", name="cost_bin_input_dec")
        else:
            b_in = None

        # Classifier-free guidance dropout (decoder side)
        c_in_dec, c_dropmask = CondDrop(self.guidance_drop_prob, self.config.num_clusters,
                                        return_mask=True, name="cf_guidance_drop_dec")(c_in)
        m_in_dec, m_dropmask = CondDrop(getattr(self, "meal_guidance_drop_prob", 0.3), 
                                        self._n_meals, return_mask=True, name="mf_guidance_drop_dec")(m_in)
        if self.use_cost_bin:
            b_in_dec, b_dropmask = CondDrop(self.cost_guidance_drop_prob, 
                                            self.num_cost_bins, return_mask=True, 
                                            name="bf_guidance_drop_dec")(b_in)

        # Embeddings (add +1 row for NULL)
        c_emb = KL.Embedding(n_clusters + 1, self._c_emb_dim, name="cluster_embedding_dec",
                            embeddings_initializer="zeros")(c_in_dec)
        m_emb = KL.Embedding(n_meals + 1,   self._m_emb_dim, name="meal_type_embedding_dec",
                            embeddings_initializer="zeros")(m_in_dec)
        if self.use_cost_bin:
            b_emb = KL.Embedding(self.num_cost_bins + 1, self._b_emb_dim, name="cost_bin_embedding_dec",
                                  embeddings_initializer="zeros")(b_in_dec)

        c_vec = KL.Flatten()(c_emb)
        m_vec = KL.Flatten()(m_emb)
        if self.use_cost_bin:
            b_vec = KL.Flatten()(b_emb)

        # Conditioning vector for FiLM
        if self.use_cost_bin:
            cond = KL.Concatenate(name="cond_vec")([c_vec, m_vec, b_vec])
        else:
            cond = KL.Concatenate(name="cond_vec")([c_vec, m_vec])

        # Backbone
        if self.use_cost_bin:
            h = KL.Concatenate()([z_in, c_vec, m_vec, b_vec])
        else:
            h = KL.Concatenate()([z_in, c_vec, m_vec])
        for i in range(self.film_blocks):
            h = KL.Dense(512, activation="gelu", name=f"dec_dense{i+1}")(h)
            h = FiLM(512, name=f"film_{i}")(h, cond)

        # ---- Presence logits base
        logits_base = KL.Dense(n_foods, activation=None, name="presence_logits_base")(h)

        # ---- Optional bias heads
        if self.use_bias_heads:
            self.cluster_logit_bias = KL.Embedding(n_clusters + 1, n_foods, name="cluster_logit_bias",
                                                embeddings_initializer="zeros")
            self.meal_logit_bias    = KL.Embedding(n_meals + 1,   n_foods, name="meal_logit_bias",
                                                embeddings_initializer="zeros")
            cb = KL.Flatten()(self.cluster_logit_bias(c_in_dec))  # (B, D)
            mb = KL.Flatten()(self.meal_logit_bias(m_in_dec))     # (B, D)

            bias_sum    = KL.Add(name="logit_bias_sum")([cb, mb])     # (B, D)
            bias_scaled = ScalarScale(name="bias_scale")(bias_sum)    # (B, D)
            logits = KL.Add(name="presence_logits_with_bias")([logits_base, bias_scaled])
        else:
            logits = KL.Activation("linear", name="presence_logits_linear")(logits_base)

        # ---- Pair prior (optional)
        if self.use_pair_prior:
            self.pair_logit_bias = KL.Embedding(n_clusters + 1, n_foods, name="pair_logit_bias",
                                                embeddings_initializer="zeros", trainable=False)
            prior_flat = KL.Flatten()(self.pair_logit_bias(c_in_dec))
            logits = KL.Add(name="presence_logits_pre_mask")([logits, prior_flat * self.pair_prior_scale])

        # ---- Allowed-set mask (optional)
        if self.use_pair_allowed_mask:
            self.pair_allowed_gate = KL.Embedding(n_clusters + 1, n_foods, name="pair_allowed_gate",
                                                  embeddings_initializer="zeros", trainable=False)
            gate_flat = KL.Flatten()(self.pair_allowed_gate(c_in_dec))  # 0=allowed, 1=disallowed
            ms = float(self.mask_strength)
            logits = KL.Lambda(lambda t: t[0] - ms * t[1], name="presence_logits_masked")([logits, gate_flat])

        # ---- Final named tensor
        logits = KL.Activation("linear", name="presence_logits")(logits)
        p_presence = KL.Activation("sigmoid", name="presence_prob")(logits)

        # Auxiliary cluster classifier head (so we can keep L_ac / A_cluster_acc)
        ac_logits = KL.Dense(n_clusters, activation=None, name="ac_logits")(h)

        if self.use_cost_bin:
            decoder = tf.keras.Model(
                [z_in, c_in, m_in, b_in],
                [p_presence, logits, ac_logits, c_in_dec, m_in_dec],
                name="decoder_presence_only"
            )
        else:
            decoder = tf.keras.Model(
                [z_in, c_in, m_in],
                [p_presence, logits, ac_logits, c_in_dec, m_in_dec],
                name="decoder_presence_only"
            )

        # stash NULL ids for guidance etc. (ensure Python ints, not tensors)
        decoder._null_cluster_id = int(self.config.num_clusters)
        decoder._null_meal_id    = int(self._n_meals)
        if self.use_cost_bin:
            decoder._null_cost_id   = int(self.num_cost_bins)

        return decoder

    
    def _set_null_zero(self, layer, null_id):
        """Ensure NULL cluster embedding is zero for unconditional behavior"""
        w = layer.get_weights()
        if not w: return
        W = w[0]
        if 0 <= null_id < W.shape[0]:
            W[null_id] = 0.0
            layer.set_weights([W])

    def _build_models(self):
        """Build encoder, decoder, and full CVAE with dual conditioning (presence-only)"""
        cfg = self.config
        D = cfg.input_dim

        # -------- Inputs --------
        x_in = tf.keras.Input(shape=(D,), name="meal_input")
        c_in = tf.keras.Input(shape=(), dtype="int32", name="cluster_input")
        m_in = tf.keras.Input(shape=(), dtype="int32", name="meal_type_input")
        if self.use_cost_bin:
            b_in = tf.keras.Input(shape=(), dtype="int32", name="cost_bin_input")
        else:
            b_in = None

        # -------- Dual conditioning embeddings --------
        # Use a *separate* encoder drop prob so z does not memorize cluster
        c_in_enc = CondDrop(self.encoder_guidance_drop_prob, cfg.num_clusters,
                            name="cf_guidance_drop_enc")(c_in)

        # Cluster embedding (+1 row for NULL id)
        c_emb = tf.keras.layers.Embedding(
            cfg.num_clusters + 1, self._c_emb_dim, name="cluster_embedding",
            embeddings_initializer="zeros"
        )(c_in_enc)
        c_vec = tf.keras.layers.Flatten(name="cluster_flatten")(c_emb)

        # Meal embedding
        m_emb = tf.keras.layers.Embedding(
            self._n_meals, self._m_emb_dim, name="meal_type_embedding"
        )(m_in)
        m_vec = tf.keras.layers.Flatten(name="meal_type_flatten")(m_emb)
        # Cost-bin embedding (encoder side, no NULL row needed here)
        if self.use_cost_bin:
            b_emb = tf.keras.layers.Embedding(self.num_cost_bins + 1, self._b_emb_dim, name="cost_bin_embedding")(b_in)
            b_vec = tf.keras.layers.Flatten(name="cost_bin_flatten")(b_emb)

        # -------- Encoder --------
        stats = (self.raw_config or {}).get("encoder_preproc")
        if stats and stats.get("type") == "zscore_log1p":
            x_enc = Log1pZScore(mean=stats["mean"], std=stats["std"], name="enc_preproc")(x_in)
        else:
            x_enc = x_in

        if self.use_cost_bin:
            x_cond = tf.keras.layers.Concatenate(name="enc_concat")([x_enc, c_vec, m_vec, b_vec])
        else:
            x_cond = tf.keras.layers.Concatenate(name="enc_concat")([x_enc, c_vec, m_vec])

        h = x_cond
        for i, units in enumerate(self._enc_sizes):
            h = tf.keras.layers.Dense(units, activation="relu", name=f"enc_dense_{i}")(h)
            if self._dropout > 0:
                h = tf.keras.layers.Dropout(self._dropout, name=f"enc_dropout_{i}")(h)

        # Latent
        z_mean   = tf.keras.layers.Dense(self._latent_dim, name="z_mean")(h)
        z_logvar = tf.keras.layers.Dense(self._latent_dim, name="z_logvar")(h)
        z        = tf.keras.layers.Lambda(sampling, name="z_sampling")([z_mean, z_logvar])

        if self.use_cost_bin:
            self.encoder = tf.keras.Model([x_in, c_in, m_in, b_in], [z_mean, z_logvar, z], name="encoder")
        else:
            self.encoder = tf.keras.Model([x_in, c_in, m_in], [z_mean, z_logvar, z], name="encoder")

        # Keep encoder's NULL row truly zero
        self._set_null_zero(self.encoder.get_layer("cluster_embedding"), cfg.num_clusters)

        # -------- Decoder (presence-only with FiLM) --------
        self.decoder = self._build_presence_decoder()
        if self.use_cost_bin:
            decoder_outputs = self.decoder([z, c_in, m_in, b_in])
        else:
            decoder_outputs = self.decoder([z, c_in, m_in])
        p_pres, d_logits, ac_logits, c_used, m_used = decoder_outputs

        # -------- Loss wiring --------
        loss_layer = PresenceOnlyCVAELossLayer(self.raw_config, name="cvae_loss")
        x_passthrough = loss_layer([d_logits, z_mean, z_logvar, x_in])
        
        # Build model first
        if self.use_cost_bin:
            self.cvae = tf.keras.Model([x_in, c_in, m_in, b_in], x_passthrough, name="cvae")
        else:
            self.cvae = tf.keras.Model([x_in, c_in, m_in], x_passthrough, name="cvae")

        self.cvae.compile(optimizer=tf.keras.optimizers.Adam(self._lr, clipnorm=0.5))


    def fit(self, X_train, C_train, M_train, X_val, C_val, M_val, 
            epochs=100, batch_size=128, patience=10, logger=None, B_train=None, B_val=None):
        """Train the presence-only CVAE"""
        import numpy as np
        
        # Initialize presence bias
        try:
            if len(X_train) > 0:
                # prevalence -> log-odds bias
                prev = np.clip((X_train > 0).mean(axis=0).astype(np.float32), 1e-4, 1 - 1e-4)
                bias = np.log(prev / (1.0 - prev)).astype(np.float32)

                base = self.decoder.get_layer("presence_logits_base")  # <- Dense
                W, b = base.get_weights()
                base.set_weights([W, bias])
        except Exception as e:
            if logger:
                logger.warning(f"Presence bias init skipped: {e}")
        # Compute pair-wise priors + allowed masks BEFORE training so they affect learning
        try:
            self._install_pair_priors_and_masks(X_train, C_train)
        except Exception as e:
            if logger:
                logger.warning(f"Skipping pair prior/mask install: {e}")

        class CleanProgressBar(tf.keras.callbacks.Callback):
            """Custom progress bar that updates in place instead of printing new lines"""
            def __init__(self):
                self.epoch = 0
                self.total_epochs = 0
                
            def on_train_begin(self, logs=None):
                self.total_epochs = self.params.get('epochs', 1)
                
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch = epoch + 1
                print(f"\nEpoch {self.epoch}/{self.total_epochs}")
                
            def on_epoch_end(self, epoch, logs=None):
                # Print key metrics in a clean format
                loss = logs.get('loss', 0)
                val_loss = logs.get('val_loss', 0)
                
                print(f"  loss: {loss:.4f} - val_loss: {val_loss:.4f}")

        class ConditioningDiagnostics(tf.keras.callbacks.Callback):
            def __init__(self, parent, Xv, Cv, Mv, pres_eps=1e-6, sample=1024):
                self.parent = parent; self.Xv = Xv; self.Cv = Cv; self.Mv = Mv
                self.pres_eps = pres_eps; self.sample = sample

            def on_epoch_end(self, epoch, logs=None):
                import numpy as np
                n = len(self.Xv); s = min(self.sample, n)
                idx = np.random.default_rng(123).choice(n, s, replace=False)
                Xb, Cb, Mb = self.Xv[idx], self.Cv[idx], self.Mv[idx]

                # Build one cluster-free latent for all probes
                # IMPORTANT: feed c=NULL into the encoder so z doesn't carry cluster-specific info
                c_null = np.full(len(Xb), self.parent.config.num_clusters, np.int32)  # NULL cluster ID
                z_mu_null = self.parent.encoder.predict([Xb, c_null, Mb], batch_size=256, verbose=0)[0]

                def ce_mean(y, p):
                    p = np.clip(p, 1e-6, 1-1e-6)
                    return float(np.mean(-(y*np.log(p) + (1-y)*np.log(1-p))))

                # unconditional (NULL) – ensure NULL gives identity FiLM (gamma=1, beta=0)
                p_u, _, _, _, _ = self.parent.decoder.predict([z_mu_null, c_null, Mb], batch_size=256, verbose=0)

                # conditioned on true cluster
                p_c, _, ac_logits, _, _ = self.parent.decoder.predict([z_mu_null, Cb, Mb], batch_size=256, verbose=0)

                # shuffle within the same meal only
                Cb_shuf = Cb.copy()
                for m in np.unique(Mb):
                    idx_meal = np.where(Mb == m)[0]
                    if len(idx_meal) > 1:
                        # simple derangement: roll by 1 within meal
                        Cb_shuf[idx_meal] = Cb[idx_meal[np.roll(np.arange(len(idx_meal)), 1)]]
                    # else keep as-is (size==1)
                frac_same = float((Cb_shuf == Cb).mean())
                print(f"  [cond] shuffle_same_frac={frac_same:.3f}")
                p_s, _, _, _, _ = self.parent.decoder.predict([z_mu_null, Cb_shuf, Mb], batch_size=256, verbose=0)

                y = (Xb > self.pres_eps).astype(np.float32)

                # ΔCE > 0 means conditioning on the true cluster helps
                for m in np.unique(Mb):
                    j = (Mb == m)
                    d_null_m = ce_mean(y[j], p_u[j]) - ce_mean(y[j], p_c[j])
                    d_shuf_m = ce_mean(y[j], p_s[j]) - ce_mean(y[j], p_c[j])
                    print(f"    meal={int(m)}: ΔCE(null→cond)={d_null_m:.4f}  ΔCE(shuf→cond)={d_shuf_m:.4f}")

                # Calculate auxiliary classifier accuracy
                ac_acc = float((ac_logits.argmax(axis=1) == Cb).mean())
                print(f"  [cond] ac_acc={ac_acc:.3f}")
                if logs is not None:
                    logs["ac_acc"] = ac_acc

                # dec@1 restricted to clusters valid for each meal
                S = min(self.sample//2, len(Xb))
                idx2 = np.random.default_rng(456).choice(len(Xb), S, replace=False)
                X2, C2, M2 = Xb[idx2], Cb[idx2], Mb[idx2]
                z2 = self.parent.encoder.predict([X2, np.full(S, self.parent.config.num_clusters, np.int32), M2],
                                                 batch_size=256, verbose=0)[0]
                y2 = (X2 > self.pres_eps).astype(np.float32)

                best = np.empty(S, dtype=np.int32)
                rand_baseline_parts = []
                for m in np.unique(M2):
                    ids = np.where(M2 == m)[0]
                    # valid candidates only for this meal:
                    cand_cs = np.array(self.parent._meal_to_pairs.get(int(m), []), dtype=np.int32)
                    if cand_cs.size == 0:
                        # fallback: all pairs (should be rare)
                        cand_cs = np.arange(self.parent.config.num_clusters, dtype=np.int32)

                    ces = []
                    for cj in cand_cs:
                        Cj = np.full(len(ids), cj, np.int32)
                        pj, _, _, _, _ = self.parent.decoder.predict([z2[ids], Cj, M2[ids]], batch_size=256, verbose=0)
                        pj = np.clip(pj, 1e-6, 1-1e-6)
                        ces.append(np.mean(-(y2[ids]*np.log(pj) + (1-y2[ids])*np.log(1-pj)), axis=1))
                    ces = np.stack(ces, axis=1)
                    best[ids] = cand_cs[ces.argmin(axis=1)]

                    rand_baseline_parts.append(1.0 / len(cand_cs))

                dec_top1 = float((best == C2).mean())
                rand_baseline = float(np.mean(rand_baseline_parts))
                if logs is not None:
                    logs["cond_dec_top1"] = dec_top1
                    logs["cond_dec_top1_random"] = rand_baseline
                print(f"  [cond] ... dec@1={dec_top1:.3f}  (random≈{rand_baseline:.3f})")

        # KLWeightScheduler is the only callback that sets the KL weight (β)
        callbacks = [
            CleanProgressBar(),
            ConditioningDiagnostics(self, X_val, C_val, M_val, pres_eps=self._pres_eps),
            KLWeightScheduler(
                layer_name=self.cvae.get_layer("cvae_loss").name,
                beta_max=self.cvae.get_layer("cvae_loss").kl_weight_target,
                warmup_epochs=self.cvae.get_layer("cvae_loss").kl_annealing_epochs,
                cyclical=True, cycle_len=8, floor=0.0
            ),
            NullZeroCallback(self),
        ]
        
        # optional diversity/fidelity monitor (logs CSV + can early-stop on composite)
        mon_cfg = (self.raw_config or {}).get("monitor", {})
        if mon_cfg.get("enabled", False):
            from .diversity_monitor import DiversityFidelityMonitor
            
            callbacks.append(
                DiversityFidelityMonitor(
                    parent=self,
                    Xv=X_val, Cv=C_val, Mv=M_val,
                    pres_eps=self._pres_eps,
                    G=int(mon_cfg.get("G", 1000)),
                    every=int(mon_cfg.get("every", 2)),
                    seed=int(mon_cfg.get("seed", 0)),
                    outdir=(self.raw_config or {}).get("outdir", "monitor_logs"),
                    csv_name=str(mon_cfg.get("csv_name", "diversity_monitor.csv")),
                    bootstraps=int(mon_cfg.get("bootstraps", 0)),  # 200 is nice; 0 to skip
                    cov_N=int(mon_cfg.get("coverage_n", 1000)),
                    early_stop=bool(mon_cfg.get("early_stop", False)),
                    patience=int(mon_cfg.get("patience", 8)),
                    min_delta=float(mon_cfg.get("min_delta", 2e-3)),
                )
            )
        if patience and patience > 0:
            callbacks += [
                EarlyStopping(monitor="val_loss", mode="min", patience=patience,
                            restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=max(1, (patience or 2)//2), min_lr=1e-6, verbose=0),
            ]

        # Training
        # Prepare inputs (optionally include cost-bin)
        if self.use_cost_bin:
            # Default B as NULL id if not provided
            if B_train is None:
                B_train = np.full(len(X_train), int(self.num_cost_bins), dtype=np.int32)
            if B_val is None:
                B_val = np.full(len(X_val), int(self.num_cost_bins), dtype=np.int32)
            train_inp = [X_train, C_train, M_train, B_train]
            val_inp   = [X_val,   C_val,   M_val,   B_val]
        else:
            train_inp = [X_train, C_train, M_train]
            val_inp   = [X_val,   C_val,   M_val]

        history = self.cvae.fit(
            train_inp, X_train,
            validation_data=(val_inp, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0  # Suppress default progress bar, use custom CleanProgressBar
        )

        # Post-training calibration: optimize temperature scaling and count offset
        self._calibrate_temperature(X_val, C_val, M_val)
        self._calibrate_count_offset(X_val, C_val, M_val)
        self.calibration_state = self.export_calibration()
        
        return history

    def _install_pair_priors_and_masks(self, X, C):
        import numpy as np
        D = self.config.input_dim
        eps = float(self._pres_eps)

        Xb = (np.asarray(X, np.float32) > eps).astype(np.float32)
        C  = np.asarray(C, np.int32).reshape(-1)
        n_pairs = int(np.max(C)) + 1

        # CRITICAL: Ensure embedding size matches actual number of pairs
        assert n_pairs == self.config.num_clusters, \
            f"Mismatch: n_pairs={n_pairs} vs config.num_clusters={self.config.num_clusters}"

        priors = np.zeros((n_pairs, D), dtype=np.float32)
        gate   = np.zeros((n_pairs, D), dtype=np.float32)  # 0=allow, 1=disallow

        for k in range(n_pairs):
            idx = (C == k)
            if not np.any(idx):
                continue
            Y = Xb[idx]                                 # (Nk, D) binary
            prev = Y.mean(axis=0)                       # prevalence per food in this pair

            # choose top-k representative foods
            if isinstance(self.allowed_topk, int):
                k_keep = int(self.allowed_topk)
            elif isinstance(self.allowed_topk, str) and self.allowed_topk.startswith("quantile:"):
                q = float(self.allowed_topk.split(":")[1])
                k_keep = int(np.quantile(Y.sum(axis=1), q))
            else:
                # "median" or anything else → median count
                k_keep = int(np.median(Y.sum(axis=1)))
            k_keep = int(np.clip(k_keep, 1, D))

            by_prev = np.argsort(prev)                  # ascending
            keep_idx = by_prev[-k_keep:]

            allow = np.zeros(D, dtype=bool)
            allow[keep_idx] = True
            allow |= (prev >= self.min_prev)            # also allow anything above min prevalence

            # gate: 1 for disallowed → subtract mask_strength in logits
            gate[k] = (~allow).astype(np.float32)

            # prior: log-odds of prevalence (clipped)
            p = np.clip(prev, 1e-4, 1-1e-4)
            priors[k] = np.log(p/(1-p)).astype(np.float32)

        # add NULL/UNCOND row (last row) as zeros so CF-dropout works cleanly
        priors_full = np.zeros((n_pairs + 1, D), dtype=np.float32)
        priors_full[:n_pairs] = priors
        gate_full = np.zeros((n_pairs + 1, D), dtype=np.float32)
        gate_full[:n_pairs] = gate

        # set decoder weights if layers exist
        try:
            if self.use_pair_prior:
                emb = self.decoder.get_layer("pair_logit_bias")
                emb.set_weights([priors_full])
            if self.use_pair_allowed_mask:
                embm = self.decoder.get_layer("pair_allowed_gate")
                embm.set_weights([gate_full])
        except Exception:
            pass

    def _calibrate_temperature(self, X_val, C_val, M_val):
        """Calibrate temperature scaling for better calibration"""
        from sklearn.metrics import log_loss
        import numpy as np
        
        # Get validation predictions
        p_presence = self.predict_heads(X_val, C_val, M_val)
        y_true_presence = (X_val > self._pres_eps).astype(np.int32)
        
        # Optimize temperature
        temperatures = np.logspace(-1, 1, 20)  # 0.1 to 10
        best_temp = 1.0
        best_loss = float('inf')
        
        for temp in temperatures:
            # Clamp probabilities away from 0/1 before logit to reduce inf
            p = np.clip(p_presence, 1e-6, 1 - 1e-6).astype(np.float64)
            logit = np.log(p / (1 - p))
            z = np.clip(logit / float(temp), -80.0, 80.0)  # <= safe for float64
            p_calibrated = 1.0 / (1.0 + np.exp(-z))
            try:
                loss = log_loss(y_true_presence.ravel(), p_calibrated.ravel())
                if loss < best_loss:
                    best_loss = loss
                    best_temp = temp
            except:
                continue
                
        self.presence_temperature = best_temp

    def _calibrate_count_offset(self, X_val, C_val, M_val):
        """Calibrate scalar logit offset to match mean counts"""
        from scipy.optimize import minimize_scalar
        import numpy as np
        
        # Get validation predictions
        p_presence = self.predict_heads(X_val, C_val, M_val)
        y_true_presence = (X_val > self._pres_eps).astype(np.int32)
        true_counts = y_true_presence.sum(axis=1)
        
        p_presence = p_presence.astype(np.float64)
        def count_mae(offset):
            logits = np.log(np.clip(p_presence,1e-6,1-1e-6) / (1-np.clip(p_presence,1e-6,1-1e-6))) + float(offset)
            logits = np.clip(logits, -80.0, 80.0)
            p_cal = 1.0 / (1.0 + np.exp(-logits))
            pred_counts = p_cal.sum(axis=1)
            return np.mean(np.abs(true_counts - pred_counts))
        
        # Find optimal offset
        result = minimize_scalar(count_mae, bounds=(-2, 2), method='bounded')
        self.count_logit_offset = result.x if result.success else 0.0

        # Per-meal offsets and temperatures
        offsets = {}
        temperatures = {}
        meals = np.unique(M_val)
        for m in meals:
            idx = (M_val == m)
            if idx.sum() < 50:  # skip tiny splits
                continue
            p_m = self.predict_heads(X_val[idx], C_val[idx], M_val[idx])
            y_true_presence_m = (X_val[idx] > self._pres_eps).astype(np.int32)
            true_counts_m = y_true_presence_m.sum(axis=1)

            # Per-meal offset calibration (MAE)
            def mae_m(off):
                logits = np.log(np.clip(p_m,1e-6,1-1e-6) / (1-np.clip(p_m,1e-6,1-1e-6))) + off
                logits = logits.astype(np.float64)
                logits = np.clip(logits, -80.0, 80.0)
                p_cal = 1.0 / (1.0 + np.exp(-logits))
                return np.mean(np.abs(true_counts_m - p_cal.sum(axis=1)))

            res = minimize_scalar(mae_m, bounds=(-2, 2), method='bounded')
            if res.success:
                offsets[int(m)] = float(res.x)

            # Per-meal temperature calibration
            def temp_loss(temp):
                p_clipped = np.clip(p_m, 1e-6, 1 - 1e-6).astype(np.float64)
                logit = np.log(p_clipped / (1 - p_clipped))
                z = np.clip(logit / float(temp), -80.0, 80.0)
                p_cal = 1.0 / (1.0 + np.exp(-z))
                try:
                    from sklearn.metrics import log_loss
                    return log_loss(y_true_presence_m.ravel(), p_cal.ravel())
                except:
                    return float('inf')

            temp_res = minimize_scalar(temp_loss, bounds=(0.1, 10.0), method='bounded')
            if temp_res.success:
                temperatures[int(m)] = float(temp_res.x)

        self.count_logit_offset_per_meal = offsets
        self.presence_temperature_per_meal = temperatures

        # Per (meal, cluster) calibration for stronger count calibration
        offsets_mc = {}
        for m in np.unique(M_val):
            for c in np.unique(C_val[M_val==m]):
                idx = (M_val==m) & (C_val==c)
                if idx.sum() < 80:  # guardrails
                    continue
                p_m = self.predict_heads(X_val[idx], C_val[idx], M_val[idx])
                y_true_m = (X_val[idx] > self._pres_eps).astype(np.int32)
                true_counts_m = y_true_m.sum(axis=1)
                def mae(off):
                    logits = np.log(np.clip(p_m,1e-6,1-1e-6) / np.clip(1-p_m,1e-6,1-1e-6)) + off
                    p_cal = 1/(1+np.exp(-np.clip(logits,-80,80)))
                    return np.mean(np.abs(true_counts_m - p_cal.sum(axis=1)))
                res = minimize_scalar(mae, bounds=(-2,2), method='bounded')
                if res.success: 
                    offsets_mc[(int(m),int(c))] = float(res.x)
        self.count_logit_offset_per_meal_cluster = offsets_mc

    def predict_heads(self, X, C, M, batch_size: int = 256, use_mean: bool = True, B=None):
        """Calibrated presence probabilities (N, D) for meals X with pair codes C and meal types M."""
        logits = self._predict_components(X, C, M, batch_size=batch_size, use_mean=use_mean, B=B)
        logits = self._apply_all_presence_cal(logits, C, M)
        # apply hard gate using actual pair ids (C)
        logits = self.apply_hard_gate_at_inference(logits, np.asarray(C, np.int32))
        return _stable_sigmoid(logits)

    def _predict_components(self, X, C, M, batch_size: int = 1024, use_mean: bool = True, B=None):
        """Forward pass to get decoder heads (presence only)"""
        import numpy as np

        X = np.asarray(X, dtype=np.float32)
        C = np.asarray(C).astype("int32").reshape(-1)
        # 🔒 Clip any stray values (shouldn't happen if you filtered)
        C = np.clip(C, 0, self.config.num_clusters)  # allow NULL==num_clusters
        M = np.clip(np.asarray(M).astype("int32").reshape(-1), 0, self._n_meals - 1)

        # Encode
        if self.use_cost_bin:
            if B is None:
                B = np.full(len(X), int(self.num_cost_bins), dtype=np.int32)
            z_mean, z_logvar, z = self.encoder.predict([X, C, M, B], batch_size=batch_size, verbose=0)
        else:
            z_mean, z_logvar, z = self.encoder.predict([X, C, M], batch_size=batch_size, verbose=0)
        z_used = z_mean if use_mean else z

        # Decode (presence only): [p_presence, logits, ac_logits, c_used, m_used]
        if self.use_cost_bin:
            if B is None:
                B = np.full(len(C), int(self.num_cost_bins), dtype=np.int32)
            decoder_outputs = self.decoder.predict([z_used, C, M, B], batch_size=batch_size, verbose=0)
        else:
            decoder_outputs = self.decoder.predict([z_used, C, M], batch_size=batch_size, verbose=0)
        if isinstance(decoder_outputs, list):
            p_presence, presence_logits, ac_logits, _, _ = decoder_outputs
        else:
            presence_logits = decoder_outputs
        return presence_logits

    def _apply_all_presence_cal(self, logits, C, M):
        """Apply presence calibration"""
        # Ensure proper shapes for broadcasting
        logits = np.asarray(logits, dtype=np.float32)
        M = np.clip(np.asarray(M).reshape(-1), 0, self._n_meals - 1)  # ensure (N,) and clip
        
        if logits.ndim == 1:
            logits = logits[None, :]  # (1, D)
            M = np.broadcast_to(M, (1,))  # (1,)
        
        # Apply temperature scaling for better calibration
        if hasattr(self, 'presence_temperature_per_meal') and self.presence_temperature_per_meal:
            base_temp = float(getattr(self, 'presence_temperature', 1.0))
            temps = np.full(M.shape[0], base_temp, dtype=np.float32)
            for mi, t in self.presence_temperature_per_meal.items():
                temps[M == int(mi)] = float(t)
            logits = logits / temps[:, None]
        elif hasattr(self, 'presence_temperature'):
            logits = logits / float(self.presence_temperature)
        
        # meal-specific offset override
        if hasattr(self, 'count_logit_offset_per_meal') and self.count_logit_offset_per_meal:
            base_off = float(getattr(self, 'count_logit_offset', 0.0))
            offs = np.full(M.shape[0], base_off, dtype=np.float32)
            for mi, off in self.count_logit_offset_per_meal.items():
                offs[M == int(mi)] = float(off)
            logits = logits + offs[:, None]
        elif hasattr(self, 'count_logit_offset'):
            logits = logits + float(self.count_logit_offset)
        
        # per (meal, cluster) offset override (highest priority)
        if hasattr(self, 'count_logit_offset_per_meal_cluster') and self.count_logit_offset_per_meal_cluster:
            C = np.asarray(C).reshape(-1)  # ensure (N,)
            if C.shape[0] == M.shape[0]:  # same batch size
                for (mi, ci), off in self.count_logit_offset_per_meal_cluster.items():
                    mask = (M == int(mi)) & (C == int(ci))
                    if mask.any():
                        logits[mask] += float(off)
        
        return logits

    def optimize_threshold(self, X_val, y_true_grams, C_val, M_val):
        """Optimize threshold for F1 macro score"""
        from sklearn.metrics import f1_score
        import numpy as np
        
        # Make threshold optimization deterministic
        np.random.seed(42)
        
        y_true_grams = np.asarray(y_true_grams, dtype=np.float32)
        y_true_presence = (y_true_grams > self._pres_eps).astype(np.int32)
        
        # Get predictions
        p_presence = self.predict_heads(X_val, C_val, M_val)
        
        # Use all data if small, otherwise sample deterministically
        n_samples = min(1000, len(X_val))
        if len(X_val) <= 1000:
            idx = np.arange(len(X_val))
        else:
            idx = np.random.choice(len(X_val), n_samples, replace=False)
        
        # Try different thresholds
        thresholds = np.linspace(0.1, 0.9, 20)
        best_f1_macro = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (p_presence[idx] >= threshold).astype(np.int32)
            f1_macro = f1_score(y_true_presence[idx].ravel(), y_pred.ravel(), average='macro', zero_division=0)
            if f1_macro > best_f1_macro:
                best_f1_macro = f1_macro
                best_threshold = threshold
                
        return best_threshold

    def evaluate(self, X_test, y_true_grams, C_test, M_test,
                 presence_threshold=None, logger=None, split_name=None):
        """Evaluate presence-only model"""
        import numpy as np
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

        # Ground truth
        y_true_grams = np.asarray(y_true_grams, dtype=np.float32)
        y_true_presence = (y_true_grams > self._pres_eps).astype(np.int32)

        # Get predictions
        p_presence = self.predict_heads(X_test, C_test, M_test)

        # Resolve threshold
        if presence_threshold is None:
            presence_threshold = 0.5
        elif isinstance(presence_threshold, str):
            if presence_threshold == "mean":
                presence_threshold = float(np.mean(p_presence))
            elif presence_threshold == "optimize":
                # Use a subset for threshold optimization
                n_samples = min(1000, len(X_test))
                rng = np.random.default_rng(42)  # reproducible
                idx = rng.choice(len(X_test), n_samples, replace=False)
                presence_threshold = self.optimize_threshold(
                    X_test[idx], y_true_grams[idx], C_test[idx], M_test[idx]
                )
            else:
                presence_threshold = 0.5

        # Binary predictions
        y_pred_presence_bin = (p_presence >= presence_threshold).astype(np.int32)

        # Compute metrics
        pred_counts = p_presence.sum(axis=1).astype(np.float32)
        true_counts = (y_true_grams > self._pres_eps).sum(axis=1)

        # Presence metrics
        f1_micro = float(f1_score(y_true_presence.ravel(), y_pred_presence_bin.ravel(), average="micro"))
        f1_macro = float(f1_score(y_true_presence.ravel(), y_pred_presence_bin.ravel(), average="macro"))
        precision_micro = float(precision_score(y_true_presence.ravel(), y_pred_presence_bin.ravel(), average="micro", zero_division=0))
        precision_macro = float(precision_score(y_true_presence.ravel(), y_pred_presence_bin.ravel(), average="macro", zero_division=0))
        recall_micro = float(recall_score(y_true_presence.ravel(), y_pred_presence_bin.ravel(), average="micro", zero_division=0))
        recall_macro = float(recall_score(y_true_presence.ravel(), y_pred_presence_bin.ravel(), average="macro", zero_division=0))

        # AUROC
        try:
            auroc_presence = float(roc_auc_score(y_true_presence.ravel(), p_presence.ravel()))
        except ValueError:
            auroc_presence = 0.0

        # AUPRC
        try:
            auprc_presence = float(average_precision_score(y_true_presence.ravel(), p_presence.ravel()))
        except ValueError:
            auprc_presence = 0.0

        # Brier score (calibration quality)
        try:
            brier_score = float(np.mean((y_true_presence.ravel() - p_presence.ravel()) ** 2))
        except:
            brier_score = 0.0

        # Expected Calibration Error (ECE)
        try:
            ece = self._compute_ece(y_true_presence.ravel(), p_presence.ravel())
        except:
            ece = 0.0

        # Count metrics
        r2_count = self._safe_r2(true_counts, pred_counts)
        mae_count = float(np.mean(np.abs(true_counts - pred_counts)))
        rmse_count = float(np.sqrt(np.mean((true_counts - pred_counts) ** 2)))
        count_bias = float(np.mean(pred_counts - true_counts))

        return {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "precision_micro": precision_micro,
            "precision_macro": precision_macro,
            "recall_micro": recall_micro,
            "recall_macro": recall_macro,
            "auroc_presence": auroc_presence,
            "auprc_presence": auprc_presence,
            "brier_score": brier_score,
            "ece": ece,
            "r2_count": r2_count,
            "mae_count": mae_count,
            "rmse_count": rmse_count,
            "count_bias": count_bias,
            "presence_threshold": presence_threshold,
            "true_count_mean": float(np.mean(true_counts)),
            "pred_count_mean": float(np.mean(pred_counts)),
        }

    def _safe_r2(self, y_true, y_pred):
        """Safe R2 calculation"""
        try:
            from sklearn.metrics import r2_score
            return float(r2_score(y_true, y_pred))
        except:
            return 0.0

    def _compute_ece(self, y_true, y_pred, n_bins=10):
        """Compute Expected Calibration Error"""
        try:
            import numpy as np
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            ece = 0
            for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
                # Include exact 0.0 in the first bin
                if i == 0:
                    in_bin = (y_pred >= bin_lower) & (y_pred <= bin_upper)
                else:
                    in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
                prop_in_bin = in_bin.mean()

                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            return float(ece)
        except:
            return 0.0

    def export_calibration(self):
        """Export calibration state (minimal for presence-only)"""
        # Convert tuple keys to strings for JSON serialization
        per_meal_cluster = getattr(self, 'count_logit_offset_per_meal_cluster', None)
        if per_meal_cluster:
            per_meal_cluster = {f"{k[0]}_{k[1]}": v for k, v in per_meal_cluster.items()}
        
        return {
            "presence_threshold_kind": "scalar",
            "presence_threshold": [0.5],
            "presence_threshold_mean": 0.5,
            "pres_logit_offsets": None,  # per-item (unused here)
            "pres_logit_offset": float(getattr(self, 'count_logit_offset', 0.0)),
            "pres_logit_offsets_per_meal": getattr(self, 'count_logit_offset_per_meal', None),
            "pres_logit_offsets_per_meal_cluster": per_meal_cluster,
            "pres_temp": float(getattr(self, 'presence_temperature', 1.0)),
            "pres_temp_per_meal": getattr(self, 'presence_temperature_per_meal', None),
            "provenance": {"built_from": "presence_only"}
        }

    def import_calibration(self, calib_dict, freeze=False):
        """Import calibration state (minimal for presence-only)"""
        self.calibration_state = calib_dict.copy()
        # Apply the scalar/per-meal calibration if provided
        self.count_logit_offset = float(calib_dict.get("pres_logit_offset", 0.0))
        self.presence_temperature = float(calib_dict.get("pres_temp", 1.0))
        self.count_logit_offset_per_meal = calib_dict.get("pres_logit_offsets_per_meal", None)
        
        # Convert string keys back to tuples for per-meal-cluster calibration
        per_meal_cluster = calib_dict.get("pres_logit_offsets_per_meal_cluster", None)
        if per_meal_cluster:
            per_meal_cluster = {tuple(map(int, k.split('_'))): v for k, v in per_meal_cluster.items()}
        self.count_logit_offset_per_meal_cluster = per_meal_cluster
        
        self.presence_temperature_per_meal = calib_dict.get("pres_temp_per_meal", None)

    def save(self, base_dir: str, feature_names: Optional[List[str]] = None):
        """Save the presence-only model"""
        os.makedirs(base_dir, exist_ok=True)
        
        # Save models
        _safe_save(self.encoder, os.path.join(base_dir, "encoder"))
        _safe_save(self.decoder, os.path.join(base_dir, "decoder"))
        _safe_save(self.cvae, os.path.join(base_dir, "cvae"))

        # Save config - use the raw_config format that evaluate.py expects
        with open(os.path.join(base_dir, "multi_meal_presence_cvae_config.json"), "w") as f:
            # Save the raw config that was passed to the constructor
            json.dump(self.raw_config, f, indent=2)

        # Save calibration
        with open(os.path.join(base_dir, "calibration.json"), "w") as f:
            json.dump(self.export_calibration(), f, indent=2)
            
        # Save feature names if provided
        if feature_names is not None:
            with open(os.path.join(base_dir, "feature_names.json"), "w") as f:
                json.dump(feature_names, f, indent=2)

    @classmethod
    def load(cls, base_dir: str, raw_config: Optional[Dict[str, Any]] = None, custom_objects: Optional[Dict] = None):
        """Load the presence-only model"""
        # Load config
        with open(os.path.join(base_dir, "multi_meal_presence_cvae_config.json"), "r") as f:
            raw_cfg = json.load(f)
        
        # Extract input_dim and num_clusters from raw config
        input_dim = raw_cfg.get("input_dim", 0)
        if input_dim == 0 and "encoder_preproc" in raw_cfg:
            encoder_preproc = raw_cfg["encoder_preproc"]
            if "mean" in encoder_preproc:
                input_dim = len(encoder_preproc["mean"])
        
        num_clusters = raw_cfg.get("num_clusters", 0)
        if num_clusters == 0 and "pair_table" in raw_cfg:
            num_clusters = len(raw_cfg["pair_table"])
        
        # Create proper cvae config
        cvae_config = raw_cfg.get("cvae_model", raw_cfg).copy()
        cvae_config["input_dim"] = input_dim
        cvae_config["num_clusters"] = num_clusters
        
        # Create config object
        config = MultiMealCVAEConfig({"cvae_model": cvae_config})
        
        # Create model instance
        model = cls(config, raw_cfg)
        
        # Load models with custom objects
        custom_objects = custom_objects or {}
        
        # Try loading .keras files first, then fallback to SavedModel
        encoder_path = os.path.join(base_dir, "encoder.keras")
        decoder_path = os.path.join(base_dir, "decoder.keras")
        cvae_path = os.path.join(base_dir, "cvae.keras")
        
        # Add custom objects for loading
        extended_custom_objects = custom_objects.copy() if custom_objects else {}
        extended_custom_objects.update({
            'PresenceOnlyCVAELossLayer': PresenceOnlyCVAELossLayer,
            'Log1pZScore': Log1pZScore,
            'sampling': sampling,
            'FiLM': FiLM,
            'CondDrop': CondDrop,
        })
        
        # Simple wrapper for SavedModel that mimics Keras model interface
        class SavedModelWrapper:
            def __init__(self, saved_model_path, model_type):
                import tensorflow as tf
                self.saved_model = tf.saved_model.load(saved_model_path)
                self.model_type = model_type
                
            def predict(self, inputs, batch_size=None, verbose=0):
                if isinstance(inputs, list) and len(inputs) == 3:
                    if self.model_type == 'encoder':
                        # Encoder: [meal_input, cluster_input, meal_type_input]
                        result = self.saved_model.signatures['serving_default'](
                            meal_input=inputs[0],
                            cluster_input=inputs[1], 
                            meal_type_input=inputs[2]
                        )
                        # Return in the expected order: [z_mean, z_logvar, z_sampling]
                        return [result['z_mean'], result['z_logvar'], result['z_sampling']]
                    elif self.model_type == 'decoder':
                        # Decoder: [z_input, cluster_input_dec, meal_type_input_dec]
                        result = self.saved_model.signatures['serving_default'](
                            z_input=inputs[0],
                            cluster_input_dec=inputs[1],
                            meal_type_input_dec=inputs[2]
                        )
                        # Return the 5-tuple shape the rest of code expects
                        logits = result.get('presence_logits', None)
                        p = result.get('presence_prob', None)
                        if p is None and logits is not None:
                            p = tf.math.sigmoid(logits)
                        ac = result.get('ac_logits', tf.zeros_like(inputs[1])[..., None])  # safe dummy if absent
                        return [p, logits, ac, inputs[1], inputs[2]]
                    elif self.model_type == 'cvae':
                        # CVAE: [meal_input, cluster_input, meal_type_input]
                        result = self.saved_model.signatures['serving_default'](
                            meal_input=inputs[0],
                            cluster_input=inputs[1],
                            meal_type_input=inputs[2]
                        )
                        return result['cvae_loss']
                else:
                    return self.saved_model.signatures['serving_default'](inputs)
                    
            def __call__(self, inputs, training=None):
                return self.predict(inputs)
                
            def __getattr__(self, name):
                return getattr(self.saved_model, name)
        
        try:
            model.encoder = tf.keras.models.load_model(encoder_path, compile=False, custom_objects=extended_custom_objects, safe_mode=False)
        except Exception as e:
            print(f"Failed to load encoder.keras: {e}")
            # Fall back to the SavedModel export written by _safe_save
            savedmodel_path = os.path.join(base_dir, "encoder_savedmodel")
            if os.path.exists(savedmodel_path):
                try:
                    model.encoder = tf.keras.models.load_model(savedmodel_path, compile=False, custom_objects=extended_custom_objects, safe_mode=False)
                except Exception as e2:
                    print(f"Failed to load encoder SavedModel: {e2}")
                    # Use SavedModel directly for Keras 3 compatibility
                    model.encoder = SavedModelWrapper(savedmodel_path, 'encoder')
            else:
                raise FileNotFoundError(f"Could not find encoder model at {encoder_path} or {savedmodel_path}")
        
        try:
            model.decoder = tf.keras.models.load_model(decoder_path, compile=False, custom_objects=extended_custom_objects, safe_mode=False)
        except Exception as e:
            print(f"Failed to load decoder.keras: {e}")
            # Fall back to the SavedModel export written by _safe_save
            savedmodel_path = os.path.join(base_dir, "decoder_savedmodel")
            if os.path.exists(savedmodel_path):
                try:
                    model.decoder = tf.keras.models.load_model(savedmodel_path, compile=False, custom_objects=extended_custom_objects, safe_mode=False)
                except Exception as e2:
                    print(f"Failed to load decoder SavedModel: {e2}")
                    # Use SavedModel directly for Keras 3 compatibility
                    model.decoder = SavedModelWrapper(savedmodel_path, 'decoder')
            else:
                raise FileNotFoundError(f"Could not find decoder model at {decoder_path} or {savedmodel_path}")
        
        try:
            model.cvae = tf.keras.models.load_model(cvae_path, compile=False, custom_objects=extended_custom_objects, safe_mode=False)
        except Exception as e:
            print(f"Failed to load cvae.keras: {e}")
            # Fall back to the SavedModel export written by _safe_save
            savedmodel_path = os.path.join(base_dir, "cvae_savedmodel")
            if os.path.exists(savedmodel_path):
                try:
                    model.cvae = tf.keras.models.load_model(savedmodel_path, compile=False, custom_objects=extended_custom_objects, safe_mode=False)
                except Exception as e2:
                    print(f"Failed to load cvae SavedModel: {e2}")
                    # Use SavedModel directly for Keras 3 compatibility
                    model.cvae = SavedModelWrapper(savedmodel_path, 'cvae')
            else:
                raise FileNotFoundError(f"Could not find cvae model at {cvae_path} or {savedmodel_path}")
        
        # Load calibration
        calib_path = os.path.join(base_dir, "calibration.json")
        if os.path.exists(calib_path):
            with open(calib_path, "r") as f:
                calib = json.load(f)
            model.import_calibration(calib, freeze=True)
        
        return model

    def predict_heads_from_z(self, z, C, M, batch_size: int = 256, B=None, **kwargs):
        import numpy as np
        C = np.asarray(C, np.int32).reshape(-1)
        M = np.clip(np.asarray(M, np.int32).reshape(-1), 0, self._n_meals - 1)
        if self.use_cost_bin:
            if B is None:
                B = np.full(len(C), int(self.num_cost_bins), dtype=np.int32)
            dec_out = self.decoder.predict([z, C, M, B], batch_size=batch_size, verbose=0)
        else:
            dec_out = self.decoder.predict([z, C, M], batch_size=batch_size, verbose=0)
        p_presence, logits = _unpack_decoder_outputs(dec_out)
        return p_presence, logits

    def decode_from_z(self, z, C, M, B=None):
        """Decode presence probabilities directly from latent z (used by DiversityFidelityMonitor)."""
        if self.use_cost_bin:
            if B is None:
                B = np.full(len(C), int(self.num_cost_bins), dtype=np.int32)
            dec_out = self.decoder([z, np.asarray(C, np.int32), np.asarray(M, np.int32), np.asarray(B, np.int32)], training=False)
        else:
            dec_out = self.decoder([z, np.asarray(C, np.int32), np.asarray(M, np.int32)], training=False)
        p_presence, logits = _unpack_decoder_outputs(dec_out)
        return p_presence

    def convert_probabilities_to_binary(self, P, method="threshold", threshold=0.5, seed=123):
        """
        Convert probabilities to binary for bootstrap/evaluation.
        
        Args:
            P: Probability array ∈ (0,1)
            method: "threshold" or "bernoulli"
            threshold: Threshold for binary conversion (method="threshold")
            seed: Random seed for bernoulli sampling (method="bernoulli")
        
        Returns:
            Binary array ∈ {0,1}
        """
        import os
        import numpy as np
        
        import tensorflow as tf
        
        if method == "bernoulli" or os.getenv("SAMPLE_BERNOULLI", "0") == "1":
            rng = np.random.default_rng(seed)
            # Convert to numpy if it's a TensorFlow tensor
            if hasattr(P, 'numpy'):
                P_np = P.numpy()
                result = (rng.random(P_np.shape) < P_np).astype(np.float32)
                return tf.convert_to_tensor(result)
            else:
                return (rng.random(P.shape) < P).astype(np.float32)
        else:
            thr = float(os.getenv("PRESENCE_THRESH", str(threshold)))
            if hasattr(P, 'numpy'):
                P_np = P.numpy()
                result = (P_np >= thr).astype(np.float32)
                return tf.convert_to_tensor(result)
            else:
                return (P >= thr).astype(np.float32)

    def _is_beverage_by_group_classification(self, food_code, food_name_mapping, debug=False):
        """
        Check if a food code represents a beverage using the same group classification
        system as the simple portion assigner.
        """
        if not food_name_mapping:
            return False
        
        food_name = food_name_mapping.get(str(food_code), "")
        if not food_name:
            return False
        
        from .simple_portion_assigner import wweia_group_and_subgroup
        
        code8 = str(food_code).zfill(8)
        main_group, subgroup = wweia_group_and_subgroup(code8, food_name)
        
        # Check if it's classified as a beverage group
        beverage_groups = ["beverages", "alcoholic_beverages"]
        is_beverage = main_group in beverage_groups
        
        if debug:
            print(f"    Beverage check: {food_code} -> {food_name} -> group={main_group}, subgroup={subgroup} -> is_beverage={is_beverage}")
        
        return is_beverage

    def diversify_top_n(self, P, n_items, feature_names=None, max_beverages=2, 
                       diversity_lambda=0.3, seed=123, food_name_mapping=None, debug=False):
        """
        Select n_items foods per row of P.

        The max_beverages most probable beverages are taken first; the remaining slots are
        then filled with the most probable remaining items (beverages included). The
        diversity_lambda penalty is the same for every candidate, so the second pass is a
        plain ranking by probability. The selection is deterministic.

        Args:
            P: Probability array (N, D) or (D,)
            n_items: Number of items to select
            feature_names: Food codes of the columns of P (for beverage detection)
            max_beverages: Number of top beverages selected first
            diversity_lambda: Penalty weight (does not change the ranking, see above)
            seed: Not used
            food_name_mapping: Food code -> description, for beverage detection
            debug: Whether to print debug information

        Returns:
            Binary array of shape (N, D) marking the selected items ((1, D) for a 1-D P)
        """
        import numpy as np

        P = np.asarray(P, dtype=np.float32)
        if P.ndim == 1:
            P = P[None, :]  # Add batch dimension

        result = np.zeros_like(P, dtype=np.float32)
        
        for i in range(P.shape[0]):
            probs = P[i]
            selected = []
            remaining = list(range(len(probs)))
            
            # First pass: select up to max_beverages beverages
            beverage_indices = []
            beverage_info = []
            if feature_names and food_name_mapping:
                for j, food_code in enumerate(feature_names):
                    if self._is_beverage_by_group_classification(food_code, food_name_mapping, debug=debug):
                        beverage_indices.append(j)
                        food_name = food_name_mapping.get(str(food_code), "Unknown")
                        beverage_info.append((j, food_code, food_name, probs[j]))
            
            if debug:
                print(f"\n[DIVERSIFY DEBUG] Sample {i}:")
                print(f"  Target items: {n_items}, Max beverages: {max_beverages}")
                print(f"  Found {len(beverage_indices)} beverages:")
                for j, code, name, prob in beverage_info:
                    print(f"    [{j}] {code}: {name} (prob={prob:.3f})")
            
            # Select beverages first (up to limit)
            beverage_probs = [(j, probs[j]) for j in beverage_indices if j in remaining]
            beverage_probs.sort(key=lambda x: x[1], reverse=True)
            
            beverages_selected = 0
            for j, prob in beverage_probs[:max_beverages]:
                if len(selected) < n_items:
                    selected.append(j)
                    remaining.remove(j)
                    beverages_selected += 1
                    if debug:
                        food_name = food_name_mapping.get(str(feature_names[j]), "Unknown")
                        print(f"  Selected beverage [{j}]: {feature_names[j]} - {food_name} (prob={prob:.3f})")
            
            if debug:
                print(f"  Selected {beverages_selected} beverages, {len(selected)} total so far")
            
            # Second pass: MMR selection for remaining slots
            while len(selected) < n_items and remaining:
                best_score = -np.inf
                best_idx = None
                
                for j in remaining:
                    # Relevance score (probability)
                    relevance = probs[j]
                    
                    # Diversity penalty (simplified: penalize similar items)
                    diversity_penalty = 0.0
                    if selected:
                        # Simple penalty based on existing selections
                        # In practice, you'd use item embeddings or co-occurrence
                        diversity_penalty = diversity_lambda * len(selected) / len(probs)
                    
                    # MMR score
                    score = relevance - diversity_penalty
                    
                    if score > best_score:
                        best_score = score
                        best_idx = j
                
                if best_idx is not None:
                    selected.append(best_idx)
                    remaining.remove(best_idx)
                    if debug:
                        food_name = food_name_mapping.get(str(feature_names[best_idx]), "Unknown")
                        print(f"  Selected non-beverage [{best_idx}]: {feature_names[best_idx]} - {food_name} (prob={probs[best_idx]:.3f}, score={best_score:.3f})")
                else:
                    break
            
            if debug:
                print(f"  Final selection: {len(selected)} items")
                print(f"  Selected indices: {selected}")
                # Show final selection with names
                final_items = []
                for j in selected:
                    food_name = food_name_mapping.get(str(feature_names[j]), "Unknown")
                    is_beverage = j in beverage_indices
                    final_items.append(f"{feature_names[j]}({food_name}){'[BEV]' if is_beverage else ''}")
                print(f"  Final items: {', '.join(final_items)}")
            
            # Set selected items to 1
            for j in selected:
                result[i, j] = 1.0
        
        return result.squeeze() if P.ndim == 1 else result

    def apply_hard_gate_at_inference(self, logits, C_pair):
        """
        Apply hard gate at inference to completely zero disallowed items.
        This pushes leakage to ~0% without changing training.
        """
        try:
            if hasattr(self.decoder, 'get_layer') and 'pair_allowed_gate' in [layer.name for layer in self.decoder.layers]:
                gate_layer = self.decoder.get_layer("pair_allowed_gate")
                gate_weights = gate_layer.get_weights()[0]  # [K+1, D], 1=disallowed
                gate_batch = gate_weights[C_pair]  # [N, D]
                # Hard zero: set disallowed items to -inf
                logits = np.where(gate_batch > 0.5, -np.inf, logits)
        except Exception as e:
            print(f"Warning: Could not apply hard gate at inference: {e}")
        return logits
