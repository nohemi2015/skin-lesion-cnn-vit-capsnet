"""
modelos_reducidos.py

Model builders for the three evaluated pipelines:

- build_efficientnetb1(): EfficientNet-B1, ImageNet weights (Keras).
- build_vit_base32():     ViT-Base/32, google/vit-base-patch32-224-in21k
                          (Hugging Face), ImageNet mean/std normalization.
- build_capsnet():        CapsNet-R3 (Sabour et al. 2017, 3 routing
                          iterations) adapted to 224x224 RGB, trained
                          from scratch with the margin loss.

Requires: tensorflow, transformers, tf-keras
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable

IMG_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype="float32")


# ---------------------------------------------------------------------
# 1. EfficientNet-B1 (CNN)
# ---------------------------------------------------------------------
def build_efficientnetb1(train_base=False, lr=1e-4):
    base = keras.applications.EfficientNetB1(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = train_base

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    # EfficientNet de Keras espera pixeles en [0,255] con su propio
    # rescaling interno; si el pipeline de datos ya normaliza a [0,1],
    # se deshace aqui (ver nota en el harness de entrenamiento).
    x = layers.Rescaling(255.0)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="EfficientNetB1")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


# ---------------------------------------------------------------------
# 2. ViT-Base (/16 and /32)
# ---------------------------------------------------------------------
class ViTBackbone(layers.Layer):
    """Envuelve el TFViTModel de Hugging Face como una capa de Keras,
    para poder controlar .trainable de forma normal dentro de un
    keras.Model funcional."""

    def __init__(self, train_base=False,
                 checkpoint="google/vit-base-patch16-224-in21k", **kwargs):
        super().__init__(**kwargs)
        from transformers import TFViTModel
        self.checkpoint = checkpoint
        self.vit = TFViTModel.from_pretrained(checkpoint)
        self.vit.trainable = train_base

    def call(self, pixel_values, training=False):
        # TFViTModel espera formato NCHW (channels-first)
        pixel_values_nchw = tf.transpose(pixel_values, [0, 3, 1, 2])
        out = self.vit(pixel_values=pixel_values_nchw, training=training)
        return out.last_hidden_state[:, 0, :]  # token [CLS]


def _build_vit(checkpoint, name, train_base=False, lr=1e-4):
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    # Normalizacion ImageNet estandar (misma que ya usa el pipeline
    # actual segun la Seccion de Data Preprocessing del manuscrito)
    x = layers.Rescaling(1.0)(inputs)  # placeholder si ya viene en [0,1]
    x = (x - IMAGENET_MEAN) / IMAGENET_STD

    vit_backbone = ViTBackbone(train_base=train_base, checkpoint=checkpoint)
    cls_token = vit_backbone(x)

    x = layers.Dense(128, activation="relu")(cls_token)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def build_vit_base16(train_base=False, lr=1e-4):
    return _build_vit("google/vit-base-patch16-224-in21k", "ViT_Base_16", train_base, lr)


def build_vit_base32(train_base=False, lr=1e-4):
    return _build_vit("google/vit-base-patch32-224-in21k", "ViT_Base_32", train_base, lr)


# ---------------------------------------------------------------------
# 3. CapsNet (Sabour et al. 2017), adaptado a 224x224 RGB
# ---------------------------------------------------------------------
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
    return scale * vectors


@register_keras_serializable(package="modelos_reducidos")
class PrimaryCaps(layers.Layer):
    """Conv2D + reshape + squash: convierte un mapa de activaciones
    convolucionales en un conjunto de capsulas primarias."""

    def __init__(self, num_capsules, dim_capsules, kernel_size, strides, **kwargs):
        super().__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.conv = layers.Conv2D(
            filters=num_capsules * dim_capsules,
            kernel_size=kernel_size,
            strides=strides,
            padding="valid",
        )

    def call(self, x):
        x = self.conv(x)
        shape = tf.shape(x)
        h, w = shape[1], shape[2]
        x = tf.reshape(x, (shape[0], h * w * self.num_capsules, self.dim_capsules))
        return squash(x)


@register_keras_serializable(package="modelos_reducidos")
class DigitCaps(layers.Layer):
    """Capsulas de salida con dynamic routing (Sabour et al. 2017)."""

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
            initializer="glorot_uniform",
            trainable=True,
            name="routing_weights",
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # inputs: (batch, input_num_capsules, input_dim_capsules)
        inputs_expand = inputs[:, :, None, :, None]  # (batch, in_caps, 1, dim_in, 1)
        inputs_tiled = tf.tile(inputs_expand, [1, 1, self.num_capsules, 1, 1])  # (batch, in_caps, out_caps, dim_in, 1)
        W_tiled = tf.tile(self.W, [batch_size, 1, 1, 1, 1])  # (batch, in_caps, out_caps, dim_out, dim_in)

        # u_hat: prediccion de cada capsula de entrada sobre cada capsula de salida
        u_hat = tf.squeeze(tf.matmul(W_tiled, inputs_tiled), axis=-1)  # (batch, in_caps, out_caps, dim_out)

        b = tf.zeros((batch_size, self.input_num_capsules, self.num_capsules))

        for i in range(self.routing_iters):
            c = tf.nn.softmax(b, axis=2)
            s = tf.reduce_sum(c[..., None] * u_hat, axis=1)
            v = squash(s)
            if i < self.routing_iters - 1:
                agreement = tf.reduce_sum(u_hat * v[:, None, :, :], axis=-1)
                b = b + agreement

        return v  # (batch, num_capsules, dim_capsules)


@register_keras_serializable(package="modelos_reducidos")
class CapsuleLength(layers.Layer):
    """Devuelve la magnitud (norma) de cada capsula de salida -- se
    interpreta como la probabilidad de esa clase."""

    def call(self, inputs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1) + tf.keras.backend.epsilon())


@register_keras_serializable(package="modelos_reducidos")
def margin_loss(y_true, y_pred, m_plus=0.9, m_minus=0.1, lam=0.5):
    """Margin loss estandar de Sabour et al. 2017, adaptada a 2 clases
    (y_true en formato one-hot: [1,0] benigno, [0,1] maligno)."""
    L = y_true * tf.square(tf.maximum(0.0, m_plus - y_pred)) + \
        lam * (1 - y_true) * tf.square(tf.maximum(0.0, y_pred - m_minus))
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))


def build_capsnet(lr=1e-3, routing_iters=3, primary_num_capsules=32, primary_dim_capsules=8):
    inputs = keras.Input(shape=(*IMG_SIZE, 3))

    x = layers.Conv2D(64, 5, strides=2, activation="relu", padding="valid")(inputs)   # 110
    x = layers.Conv2D(128, 5, strides=2, activation="relu", padding="valid")(x)        # 53
    x = layers.Conv2D(256, 5, strides=2, activation="relu", padding="valid")(x)        # 25

    primary_caps = PrimaryCaps(
        num_capsules=primary_num_capsules, dim_capsules=primary_dim_capsules,
        kernel_size=9, strides=2
    )(x)

    digit_caps = DigitCaps(
        num_capsules=2, dim_capsules=16, routing_iters=routing_iters
    )(primary_caps)

    lengths = CapsuleLength()(digit_caps)

    model = keras.Model(inputs, lengths, name=f"CapsNet_r{routing_iters}_c{primary_num_capsules}x{primary_dim_capsules}")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr),
        loss=margin_loss,
        metrics=["accuracy"],
    )
    return model


MODEL_BUILDERS = {
    "efficientnetb1": build_efficientnetb1,
    "vit_base16": build_vit_base16,
    "vit_base32": build_vit_base32,
    "capsnet": build_capsnet,
}
