#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 23:58:12 2025

@author: dev
"""


from sklearn.svm import SVC


def init_model():
    """
    Initializes a Support Vector Classifier (SVM) with default hyperparameters.
    Edit hyperparameters below as needed.
    """
    model = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42
    )
    return model
