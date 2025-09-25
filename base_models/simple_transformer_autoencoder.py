#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 00:05:35 2025

@author: dev
"""


import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Dropout


def init_model(seq_length=50, d_model=64, num_heads=2, ff_dim=128, num_classes=2):
    """
    Initializes a simple Transformer encoder.
    Edit heads, feed-forward size, dropout as needed.
    """
    inputs = Input(shape=(seq_length, d_model))
    attn_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attn_output = Dropout(0.1)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ff_output = Dense(ff_dim, activation='relu')(out1)
    ff_output = Dense(d_model)(ff_output)
    ff_output = Dropout(0.1)(ff_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ff_output)

    pooled = tf.reduce_mean(out2, axis=1)
    outputs = Dense(num_classes, activation='softmax')(pooled)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, 'tensorflow'
