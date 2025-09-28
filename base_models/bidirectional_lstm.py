#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 00:05:14 2025

@author: dev
"""


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense


def init_model(vocab_size=10000, embedding_dim=128, input_length=100, num_classes=2):
    """
    Initializes a Bidirectional LSTM model.
    Edit units, dropout, and layers as needed.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                  input_length=input_length),
        Bidirectional(LSTM(64)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
