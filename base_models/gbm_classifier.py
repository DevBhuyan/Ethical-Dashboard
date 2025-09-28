#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 23:58:12 2025

@author: dev
"""


from sklearn.ensemble import GradientBoostingClassifier


def init_model():
    """
    Initializes a Gradient Boosting Classifier with default hyperparameters.
    Edit hyperparameters below as needed.
    """
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    return model
