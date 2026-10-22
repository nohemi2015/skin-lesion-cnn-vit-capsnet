"""
modelos_v3.py

Model builders for the revision experiments (protocol v3).

The original builders (modelos_reducidos.py) are kept unchanged so that the
original results (v2) can be reproduced and their saved weights reloaded.

Differences with respect to modelos_reducidos.py
------------------------------------------------
1. ViT is now built ENTIRELY in tf_keras (Keras 2), the framework used by the
   Hugging Face TF models. In the original code the Hugging Face model (a
   tf_keras object) was wrapped inside a Keras 3 layer; Keras 3 does not track
   tf_keras variables, so the ViT backbone was never updated (only the dense
   head was trained), even with train_base=True. Here the backbone is trained
   when train_base=True, and frozen when train_base=False.
2. ViT input normalization is selectable:
     "imagenet"   -> mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225)
                     (the normalization used in the original runs)
     "checkpoint" -> mean 0.5, std 0.5 per channel (reference preprocessing
                     of google/vit-base-patch32-224-in21k)
3. CapsNet-R3 accepts the learning rate and an optional input normalization,
   and its layers are named so that intermediate outputs can be inspected.

Architectures, heads, losses and default hyperparameters are otherwise those
of the original pipelines.
"""

import os

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.saving import register_keras_serializable

IMG_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype="float32")
CKPT_MEAN = np.array([0.5, 0.5, 0.5], dtype="float32")
CKPT_STD = np.array([0.5, 0.5, 0.5], dtype="float32")
VIT_CHECKPOINT = "google/vit-base-patch32-224-in21k"

# Only for local smoke tests without internet: random weights / tiny ViT.
SMOKE = os.environ.get("SMOKE_TEST") == "1"


# ---------------------------------------------------------------------
# EfficientNet-B1 (Keras 3) -- identical to the original pipeline
# ---------------------------------------------------------------------
def build_efficientnetb1(train_base=True, lr=1e-4):
    base = keras.applications.EfficientNetB1(
        input_shape=(*IMG_SIZE, 3), include_top=False,
        weights=None if SMOKE else "imagenet")
    base.trainable = train_base
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(255.0)(inputs)          # Keras EfficientNet expects [0, 255]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="EfficientNetB1")
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model


# ---------------------------------------------------------------------
# ViT-Base/32 (tf_keras end to end, so that fine-tuning really happens)
# ---------------------------------------------------------------------
def build_vit_base32(train_base=True, lr=1e-4, norm="checkpoint"):
    import tf_keras
    from transformers import TFViTModel, ViTConfig

    if norm == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif norm == "checkpoint":
        mean, std = CKPT_MEAN, CKPT_STD
    else:
        raise ValueError(norm)

    if SMOKE:
        vit = TFViTModel(ViTConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                                   intermediate_size=128, image_size=224, patch_size=32))
    else:
        vit = TFViTModel.from_pretrained(VIT_CHECKPOINT)
    vit.trainable = train_base

    mean_c = tf.constant(mean.reshape(1, 1, 1, 3))
    std_c = tf.constant(std.reshape(1, 1, 1, 3))
    inputs = tf_keras.Input(shape=(*IMG_SIZE, 3))
    x = tf_keras.layers.Lambda(
        lambda t: tf.transpose((t - mean_c) / std_c, [0, 3, 1, 2]), name="normalize_nchw")(inputs)
    cls = vit(pixel_values=x).last_hidden_state[:, 0, :]          # [CLS] token
    h = tf_keras.layers.Dense(128, activation="relu")(cls)
    h = tf_keras.layers.Dropout(0.3)(h)
    outputs = tf_keras.layers.Dense(1, activation="sigmoid")(h)
    model = tf_keras.Model(inputs, outputs, name=f"ViT_B32_{'ft' if train_base else 'frozen'}_{norm}")
    model.compile(optimizer=tf_keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.004),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf_keras.metrics.AUC(name="auc")])
    model.vit_backbone = vit
    return model


# ---------------------------------------------------------------------
# CapsNet-R3 (Keras 3) -- same architecture as the original pipeline
# ---------------------------------------------------------------------
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + 1e-7)
    return scale * vectors


@register_keras_serializable(package="modelos_v3")
class PrimaryCaps(layers.Layer):
    def __init__(self, num_capsules, dim_capsules, kernel_size, strides, **kwargs):
        super().__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv = layers.Conv2D(filters=num_capsules * dim_capsules, kernel_size=kernel_size,
                                  strides=strides, padding="valid")

    def call(self, x):
        x = self.conv(x)
        shape = tf.shape(x)
        x = tf.reshape(x, (shape[0], shape[1] * shape[2] * self.num_capsules, self.dim_capsules))
        return squash(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(num_capsules=self.num_capsules, dim_capsules=self.dim_capsules,
                   kernel_size=self.kernel_size, strides=self.strides)
        return cfg


@register_keras_serializable(package="modelos_v3")
class DigitCaps(layers.Layer):
    """Output capsules with dynamic routing (Sabour et al., 2017).
    Gradients propagate through all routing iterations."""

    def __init__(self, num_capsules, dim_capsules, routing_iters=3, **kwargs):
        super().__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routing_iters = routing_iters

    def build(self, input_shape):
        self.input_num_capsules = input_shape[1]
        self.input_dim_capsules = input_shape[2]
        self.W = self.add_weight(
            shape=(1, self.input_num_capsules, self.num_capsules,
                   self.dim_capsules, self.input_dim_capsules),
            initializer="glorot_uniform", trainable=True, name="routing_weights")

    def predictions(self, inputs):
        batch = tf.shape(inputs)[0]
        u = tf.tile(inputs[:, :, None, :, None], [1, 1, self.num_capsules, 1, 1])
        W = tf.tile(self.W, [batch, 1, 1, 1, 1])
        return tf.squeeze(tf.matmul(W, u), axis=-1)            # u_hat (batch, in, out, dim)

    def route(self, u_hat):
        batch = tf.shape(u_hat)[0]
        b = tf.zeros((batch, self.input_num_capsules, self.num_capsules))
        c = None
        for i in range(self.routing_iters):
            c = tf.nn.softmax(b, axis=2)                       # over output capsules
            s = tf.reduce_sum(c[..., None] * u_hat, axis=1)    # pre-squash (batch, out, dim)
            v = squash(s)
            if i < self.routing_iters - 1:
                b = b + tf.reduce_sum(u_hat * v[:, None, :, :], axis=-1)
        return v, s, c

    def call(self, inputs):
        v, _, _ = self.route(self.predictions(inputs))
        return v

    def get_config(self):
        cfg = super().get_config()
        cfg.update(num_capsules=self.num_capsules, dim_capsules=self.dim_capsules,
                   routing_iters=self.routing_iters)
        return cfg


@register_keras_serializable(package="modelos_v3")
class CapsuleLength(layers.Layer):
    def call(self, inputs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1) + 1e-7)


@register_keras_serializable(package="modelos_v3")
def margin_loss(y_true, y_pred, m_plus=0.9, m_minus=0.1, lam=0.5):
    """y_true one-hot [benign, malignant]; y_pred capsule lengths."""
    L = y_true * tf.square(tf.maximum(0.0, m_plus - y_pred)) + \
        lam * (1 - y_true) * tf.square(tf.maximum(0.0, y_pred - m_minus))
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))


def build_capsnet(lr=1e-3, routing_iters=3, input_norm="none", weight_decay=0.004,
                  primary_num_capsules=32, primary_dim_capsules=8):
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = inputs
    if input_norm == "imagenet":
        x = layers.Normalization(mean=IMAGENET_MEAN, variance=IMAGENET_STD ** 2, name="input_norm")(x)
    elif input_norm != "none":
        raise ValueError(input_norm)
    x = layers.Conv2D(64, 5, strides=2, activation="relu", padding="valid", name="conv1")(x)
    x = layers.Conv2D(128, 5, strides=2, activation="relu", padding="valid", name="conv2")(x)
    x = layers.Conv2D(256, 5, strides=2, activation="relu", padding="valid", name="conv3")(x)
    primary = PrimaryCaps(primary_num_capsules, primary_dim_capsules, 9, 2, name="primary_caps")(x)
    digit = DigitCaps(2, 16, routing_iters=routing_iters, name="digit_caps")(primary)
    lengths = CapsuleLength(name="capsule_length")(digit)
    model = keras.Model(inputs, lengths, name=f"CapsNet_r{routing_iters}")
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay),
                  loss=margin_loss, metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------
# Pipelines of the revision protocol
# ---------------------------------------------------------------------
PIPELINES = {
    # name: (builder, kwargs, framework)
    "efficientnetb1_ft":    (build_efficientnetb1, dict(train_base=True), "keras3"),
    "vit32_frozen_ckpt":    (build_vit_base32, dict(train_base=False, norm="checkpoint"), "tf_keras"),
    "vit32_ft_ckpt":        (build_vit_base32, dict(train_base=True, norm="checkpoint"), "tf_keras"),
    "vit32_ft_imagenet":    (build_vit_base32, dict(train_base=True, norm="imagenet"), "tf_keras"),
    # Re-implementation of the ViT pipeline as it was actually run in v2
    # (frozen backbone, trained head, ImageNet statistics).
    "vit32_frozen_imagenet":(build_vit_base32, dict(train_base=False, norm="imagenet"), "tf_keras"),
    "capsnet_r3":           (build_capsnet, dict(), "keras3"),
    # CapsNet-R3 revised: lr and input_norm are passed on the command line
    # after the validation-only selection of capsnet_diagnostico.py (gate G6).
    "capsnet_r3_revised":   (build_capsnet, dict(), "keras3"),
}


def build_pipeline(name, **overrides):
    builder, kwargs, framework = PIPELINES[name]
    kw = dict(kwargs)
    kw.update({k: v for k, v in overrides.items() if v is not None})
    return builder(**kw), framework


def count_trainable(model):
    """Number of trainable scalar parameters actually seen by the optimizer."""
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))
