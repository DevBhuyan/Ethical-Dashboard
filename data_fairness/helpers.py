#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:27:50 2024

@author: dev
"""

import matplotlib.pyplot as plt
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.postprocessing import ThresholdOptimizer
import numpy as np
import pandas as pd


class bcolors:
    """
    A class containing ANSI escape codes for text color formatting.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def sigmoid(x: float) -> float:
    """
    Compute the sigmoid function of the input.

    Parameters:
    - x : float
        Input value.

    Returns:
    - float
        Sigmoid of the input value.
    """
    return 1 / (1 + np.exp(-x))


def gr0(y_test: pd.Series,
        group_indices: list) -> list:
    """
    Compute the indices of group 0.

    Parameters:
    - y_test : pd.Series
        True labels.
    - group_indices : list
        Indices of the group.

    Returns:
    - list
        Indices of group 0.
    """
    group_0th_indexing = []
    for idx, i in enumerate(y_test.index):
        if i in group_indices:
            group_0th_indexing.append(idx)

    return group_0th_indexing


def plot_wo_save(data: dict,
                 sensitive_feature: str,
                 metrics: dict,
                 pred: np.array,
                 silent: bool = False) -> tuple:
    """
    Plot the results without saving the plot.

    Parameters:
    - data : dict
        Dictionary containing dataset information.
    - sensitive_feature : str
        Name of the sensitive feature.
    - metrics : dict
        Dictionary containing fairness metrics functions.
    - pred : np.array
        Predicted labels.
    - silent : bool, optional
        Whether to suppress the plot display. Default is False.

    Returns:
    - tuple
        Tuple containing the matplotlib figure object and the computed results.
    """
    results = {}
    indices = []
    for group_value, group_label in data["category_maps"][sensitive_feature].items():

        group_metrics = {}
        group_indices = data["X_test"][data["X_test"][sensitive_feature]
                                       == group_value].index
        gr0_indices = gr0(data["y_test"], group_indices)
        indices.append(gr0_indices)
        for metric_name, metric_func in metrics.items():
            if metric_name not in ["accuracy_score_diff", "false_positive_rate_ratio", "false_negative_rate_ratio", "demographic_parity_difference", "equalized_odds_diff"]:
                group_metrics[metric_name] = metric_func(
                    data["y_test"].iloc[gr0_indices], pred[gr0_indices])
        results[group_label] = group_metrics

    results["INFO"] = "All values mentioned below are computed in percentage (%)"
    for metric_name in ["accuracy_score_diff", "false_positive_rate_ratio", "false_negative_rate_ratio", "demographic_parity_difference", "equalized_odds_diff"]:
        results[metric_name] = metrics[metric_name](
            data["y_test"], pred, indices)*100

    results["BIAS_MEASURE"] = sum([results["accuracy_score_diff"]*0.32,
                                  results["demographic_parity_difference"]*0.32,
                                  results["equalized_odds_diff"]*0.12,
                                  sigmoid(
                                      results["false_negative_rate_ratio"])*0.12,
                                  sigmoid(results["false_positive_rate_ratio"])*0.12])

    if results["BIAS_MEASURE"] == np.nan:
        raise Exception(results)

    # Plot the results
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))

    for i, metric in enumerate(["accuracy", "precision", "false_positive_rate", "false_negative_rate", "selection_rate", "count"]):
        values = []
        for group_label in data["category_maps"][sensitive_feature].values():
            values.append(results[group_label][metric])

        axes[i].bar(data["category_maps"]
                    [sensitive_feature].values(), values)
        axes[i].set_title(metric.capitalize())
        axes[i].set_ylabel(metric.capitalize())

    plt.tight_layout()
    if not silent:
        plt.show()

    return fig, results


def build_fair_model(accurate_model: object,
                     data: dict,
                     sensitive_feature_train: pd.Series,
                     sensitive_feature_test: pd.Series,
                     reduction: classmethod,
                     reduction_name: str,
                     threshold: bool = False) -> tuple:

    if threshold:
        fair_model = ThresholdOptimizer(estimator=accurate_model,
                                        constraints=reduction_name,
                                        objective='balanced_accuracy_score',
                                        prefit=True
                                        )

        fair_model.fit(data["X_train"], data["y_train"],
                       sensitive_features=sensitive_feature_train)

        print("Done!")

        fair_pred = fair_model.predict(
            data["X_test"], sensitive_features=sensitive_feature_test)

    else:
        fair_model = ExponentiatedGradient(accurate_model,
                                           constraints=reduction())

        fair_model.fit(data["X_train"], data["y_train"],
                       sensitive_features=sensitive_feature_train)

        print("Done!")

        fair_pred = fair_model.predict(
            data["X_test"])

    return fair_pred, fair_model
