#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 00:04:09 2025

@author: dev
"""

from sklearn.tree import DecisionTreeClassifier


def init_model():
    """
    Initializes a Decision Tree Classifier with default hyperparameters.
    Edit hyperparameters below as needed.
    """
    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=None,
        random_state=42
    )
    return model
