#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 00:04:46 2025

@author: dev
"""


from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense


def init_model(input_dim=32, encoding_dim=16):
    """
    Initializes a simple autoencoder.
    Edit layer sizes and activations as needed.
    """
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    model = Model(inputs=input_layer, outputs=decoded)
    model.compile(optimizer='adam', loss='mse')

    return model
