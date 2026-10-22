"""
verificar_vit_original.py

Checks whether the ORIGINAL ViT builder (modelos_reducidos.build_vit_base32,
used for the published v2 results) actually trains the ViT backbone.

It builds the model exactly as entrenar_split_final_v2.py did (train_base=True),
prints how many parameters the optimizer sees, runs one short training step on
random images and reports whether any backbone variable changed.
Run it in the same folder and environment used for the original experiments.

    python verificar_vit_original.py
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

import modelos_reducidos as mr

model = mr.build_vit_base32(train_base=True)
vit_layer = [l for l in model.layers if isinstance(l, mr.ViTBackbone)][0]
n_model = int(sum(np.prod(w.shape) for w in model.trainable_weights))
n_vit = int(sum(np.prod(w.shape) for w in vit_layer.vit.trainable_variables))
print(f"Keras version: {keras.__version__}")
print(f"Parameters the optimizer will update (model.trainable_weights): {n_model:,}")
print(f"Parameters inside the Hugging Face ViT backbone:               {n_vit:,}")

before = [v.numpy().copy() for v in vit_layer.vit.trainable_variables]
x = np.random.rand(8, 224, 224, 3).astype("float32")
y = np.array([1, 0] * 4, dtype="float32")
model.fit(x, y, epochs=1, batch_size=8, verbose=0)
changed = sum(not np.allclose(b, v.numpy()) for b, v in zip(before, vit_layer.vit.trainable_variables))
print(f"Backbone variables changed after one training step: {changed} of {len(before)}")
if changed == 0:
    print("\nRESULT: the ViT backbone was NOT trained (frozen). Only the dense head was trained.")
else:
    print("\nRESULT: the ViT backbone IS trained.")
