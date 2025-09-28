#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 00:03:37 2025

@author: dev
"""


from catboost import CatBoostClassifier


def init_model():
    """
    Initializes a CatBoost Classifier with default hyperparameters.
    Edit hyperparameters below as needed.
    """
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        random_state=42
    )
    return model
