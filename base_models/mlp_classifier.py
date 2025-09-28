#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 00:03:53 2025

@author: dev
"""


from sklearn.neural_network import MLPClassifier


def init_model():
    """
    Initializes an MLP Classifier with default hyperparameters.
    Edit layers, activation, and solver as needed.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42
    )
    return model
