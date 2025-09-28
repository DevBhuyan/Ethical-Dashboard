#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 00:01:44 2025

@author: dev
"""


from sklearn.neighbors import KNeighborsClassifier


def init_model():
    """
    Initializes a K-Nearest Neighbors Classifier with default hyperparameters.
    Edit hyperparameters below as needed.
    """
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',
        algorithm='auto'
    )
    return model
