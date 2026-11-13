#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared CVAE configuration and the encoder's input pre-processing layer.

- MultiMealCVAEConfig: model and training settings read from the "cvae_model" section of
  config/cvae_multi_meal.json, used by the presence-only CVAE
  (src/utils/multi_meal_presence_cvae.py).
- Log1pZScore: Keras layer applying log1p and a z-score fitted on the training grams
  (the encoder input pre-processing).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="multi_meal")
class Log1pZScore(tf.keras.layers.Layer):
    def __init__(self, mean, std, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        # store as plain numpy so get_config is JSONable
        self._mean = np.asarray(mean, dtype=np.float32)
        self._std = np.asarray(std, dtype=np.float32)
        self.eps = float(eps)

    def build(self, input_shape):
        # non-trainable weights → saved cleanly as weights, not JSON
        self.mu = self.add_weight(
            name="mu",
            shape=self._mean.shape,
            initializer=tf.constant_initializer(self._mean),
            trainable=False,
        )
        self.sd = self.add_weight(
            name="sd",
            shape=self._std.shape,
            initializer=tf.constant_initializer(self._std),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        x = tf.nn.relu(x)
        x = tf.math.log1p(x)
        return (x - self.mu) / (self.sd + self.eps)

    def get_config(self):
        # everything JSON-safe
        base = super().get_config()
        base.update({"mean": self._mean.tolist(), "std": self._std.tolist(), "eps": self.eps})
        return base


class CVAEConfig:
    def __init__(self, raw_config: Dict[str, Any]):
        """Initialize config from raw_config dictionary, reading from cvae_model section"""
        cm = raw_config.get("cvae_model", {})

        # Data / architecture - read from config
        self.input_dim: int = cm.get("input_dim", 0)  # Will be set later
        self.num_clusters: int = cm.get("num_clusters", 0)  # Will be set later
        self.latent_dim: int = cm.get("latent_dim", 48)
        self.encoder_sizes: List[int] = cm.get("encoder_sizes", [512, 256, 128, 64])
        self.cluster_embedding_dim: int = cm.get("cluster_embedding_dim", 32)
        self.dropout_rate: float = cm.get("dropout_rate", 0.1)

        # Optimization - read from config
        self.learning_rate: float = cm.get("learning_rate", 0.001)
        self.epochs: int = cm.get("epochs", 120)
        self.batch_size: int = cm.get("batch_size", 64)
        self.patience: int = cm.get("patience", 12)

        # Presence loss - read from config
        self.presence_pos_weight: float = cm.get("presence_pos_weight", 5.0)
        self.presence_eps: float = cm.get("presence_eps", 1e-6)
        self.focal_gamma: float = cm.get("focal_gamma", 1.5)

        # KL controls - read from config
        self.kl_free_bits: float = cm.get("kl_free_bits", 0.5)
        self.kl_warmup_epochs: int = cm.get("kl_warmup_epochs", 45)


class MultiMealCVAEConfig(CVAEConfig):
    """Extended config for multi-meal CVAE with meal type conditioning"""

    def __init__(self, raw_config: Dict[str, Any]):
        super().__init__(raw_config)
        cm = raw_config.get("cvae_model", {})

        # Multi-meal specific parameters
        self.meal_types: List[str] = cm.get("meal_types", ["Breakfast", "Lunch", "Dinner"])
        self.meal_embedding_dim: int = cm.get("meal_embedding_dim", 16)

        # Override some defaults for multi-meal training
        self.epochs = cm.get("epochs", 150)        # More epochs for larger dataset
        self.batch_size = cm.get("batch_size", 128)  # Larger batch size
        self.learning_rate = cm.get("learning_rate", 0.0003)  # Slightly lower LR
