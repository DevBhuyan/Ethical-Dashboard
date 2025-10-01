#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 22:58:05 2025

@author: dev
"""

from session_state_attrib import ss
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import numpy as np
import tensorflow as tf


def train(
        test_size=0.2,
        random_state=42,
        epochs=5,
        batch_size=32
):

    df = ss.selected_dataset
    model = ss.selected_model

    # Use 'Class' column as target
    y = df["Class"].values
    X = df.drop(columns=["Class"]).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    if hasattr(model, "fit") and not isinstance(model, tf.keras.Model):
        model.fit(X_train, y_train)

    elif isinstance(model, tf.keras.Model):
        # Ensure y is categorical if needed
        if len(np.unique(y)) > 2 and y_train.ndim == 1:
            y_train = tf.keras.utils.to_categorical(y_train)
            y_test = tf.keras.utils.to_categorical(y_test)

        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
    else:
        raise TypeError("Unsupported model type")

    ss.trained_model = model
    ss.X_test, ss.y_test = X_test, y_test


def eval_model():
    if not hasattr(ss, "trained_model"):
        raise ValueError("No trained model found. Train first!")

    model = ss.trained_model
    X_test, y_test = ss.X_test, ss.y_test

    # Prediction
    if hasattr(model, "predict") and not isinstance(model, tf.keras.Model):
        # sklearn/xgb
        y_pred = model.predict(X_test)

    elif isinstance(model, tf.keras.Model):
        y_prob = model.predict(X_test)

        # Handle binary or multi-class
        if y_prob.shape[1] > 1:
            y_pred = np.argmax(y_prob, axis=1)
            if y_test.ndim > 1:
                y_test = np.argmax(y_test, axis=1)
        else:
            y_pred = (y_prob > 0.5).astype(int).flatten()
            if y_test.ndim > 1:
                y_test = y_test.argmax(axis=1)
    else:
        raise TypeError("Unsupported model type")

    metrics = {
        "accuracy": accuracy_score(y_test,
                                   y_pred),
        "precision": precision_score(y_test,
                                     y_pred,
                                     average="weighted",
                                     zero_division=0),
        "recall": recall_score(y_test,
                               y_pred,
                               average="weighted",
                               zero_division=0),
        "f1_score": f1_score(y_test,
                             y_pred,
                             average="weighted",
                             zero_division=0)
    }

    ss.eval_metrics = metrics
