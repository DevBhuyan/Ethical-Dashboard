#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 00:03:19 2025

@author: dev
"""


import lightgbm as lgb


def init_model():
    """
    Initializes a LightGBM Classifier with default hyperparameters.
    Edit hyperparameters below as needed.
    """
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=-1,
        random_state=42
    )
    return model
