#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:16:00 2024

@author: dev
"""

from sklearn.metrics import accuracy_score
from fairlearn.metrics import true_positive_rate, false_positive_rate, false_negative_rate, selection_rate
import pandas as pd
import numpy as np


def accuracy_score_difference(y_true: pd.Series,
                              y_pred: np.array,
                              group_indices_list: list) -> float:
    """
   Compute the absolute difference in accuracy scores between different groups.

   Parameters:
   - y_true : pd.Series
       True labels.
   - y_pred : np.array
       Predicted labels.
   - group_indices_list : list
       List of group indices.

   Returns:
   - float
       Absolute difference in accuracy scores.
   """
    accuracy_group_list = []
    for group_indices in group_indices_list:
        group_y_true = y_true.iloc[group_indices]
        group_y_pred = y_pred[group_indices]
        accuracy_group = accuracy_score(group_y_true, group_y_pred)
        accuracy_group_list.append(accuracy_group)

    max_accuracy = max(accuracy_group_list)
    min_accuracy = min(accuracy_group_list)

    return abs(max_accuracy - min_accuracy)


def false_positive_rate_ratio(y_true: pd.Series,
                              y_pred: np.array,
                              group_indices_list: list) -> float:
    """
    Compute the ratio of false_positive_rates between different groups.

    Parameters:
    - y_true : pd.Series
        True labels.
    - y_pred : np.array
        Predicted labels.
    - group_indices_list : list
        List of group indices.

    Returns:
    - float
        Ratio of false_positive_rates.
    """
    fpr_values = []
    for group_indices in group_indices_list:
        group_y_true = y_true.iloc[group_indices]
        group_y_pred = y_pred[group_indices]
        fpr = false_positive_rate(group_y_true, group_y_pred)
        fpr_values.append(fpr)

    max_fpr = max(fpr_values)
    min_fpr = min(fpr_values)

    if np.isnan(max_fpr / min_fpr) or np.isinf(max_fpr / min_fpr):
        return 1
    return min(max_fpr / min_fpr, 1)


def false_negative_rate_ratio(y_true: pd.Series,
                              y_pred: np.array,
                              group_indices_list: list) -> float:
    """
    Compute the ratio of false_negative_rates between different groups.

    Parameters:
    - y_true : pd.Series
        True labels.
    - y_pred : np.array
        Predicted labels.
    - group_indices_list : list
        List of group indices.

    Returns:
    - float
        Ratio of false_negative_rates.
    """
    fnr_values = []
    for group_indices in group_indices_list:
        group_y_true = y_true.iloc[group_indices]
        group_y_pred = y_pred[group_indices]
        fnr = false_negative_rate(group_y_true, group_y_pred)
        fnr_values.append(fnr)

    max_fnr = max(fnr_values)
    min_fnr = min(fnr_values)

    if np.isnan(max_fnr / min_fnr) or np.isinf(max_fnr / min_fnr):
        return 1
    return min(max_fnr / min_fnr, 1)


def demographic_parity_difference(y_true: pd.Series,
                                  y_pred: np.array,
                                  group_indices_list: list) -> float:
    """
    Compute the maximum absolute difference in selection rates between different groups.

    Parameters:
    - y_true : pd.Series
        True labels.
    - y_pred : np.array
        Predicted labels.
    - group_indices_list : list
        List of group indices.

    Returns:
    - float
        Maximum absolute difference in selection rates.
    """
    selection_rates = []
    for group_indices in group_indices_list:
        group_y_true = y_true.iloc[group_indices]
        group_y_pred = y_pred[group_indices]
        selection_rate_group = selection_rate(group_y_true, group_y_pred)
        selection_rates.append(selection_rate_group)

    overall_selection_rate = selection_rate(y_true, y_pred)
    max_diff = max(abs(sr - overall_selection_rate) for sr in selection_rates)

    return max_diff


def equalized_odds_difference(y_true: pd.Series,
                              y_pred: np.array,
                              group_indices_list: list) -> float:
    """
    Compute the sum of absolute differences in true positive and false_positive_rates between different groups.

    Parameters:
    - y_true : pd.Series
        True labels.
    - y_pred : np.array
        Predicted labels.
    - group_indices_list : list
        List of group indices.

    Returns:
    - float
        Sum of absolute differences in true positive and false_positive_rates.
    """
    tpr_values = []
    fpr_values = []
    for group_indices in group_indices_list:
        group_y_true = y_true.iloc[group_indices]
        group_y_pred = y_pred[group_indices]
        tpr = true_positive_rate(group_y_true, group_y_pred)
        fpr = false_positive_rate(group_y_true, group_y_pred)
        tpr_values.append(tpr)
        fpr_values.append(fpr)

    max_tpr = max(tpr_values)
    min_tpr = min(tpr_values)
    max_fpr = max(fpr_values)
    min_fpr = min(fpr_values)

    return abs(max_tpr - min_tpr) + abs(max_fpr - min_fpr)


def overall_accuracy_score_difference(y_true: pd.Series,
                                      y_pred: np.array,
                                      group_indices_list: list) -> float:
    """
    Compute the maximum pairwise difference in accuracy scores among all groups.

    Parameters:
    - y_true : pd.Series
        True labels.
    - y_pred : np.array
        Predicted labels.
    - group_indices_list : list
        List of group indices.

    Returns:
    - float
        Maximum pairwise difference in accuracy scores.
    """
    if len(group_indices_list) == 2:
        return accuracy_score_difference(y_true, y_pred, group_indices_list)
    accuracies = []
    for group_indices in group_indices_list:
        group_y_true = y_true.iloc[group_indices]
        group_y_pred = y_pred[group_indices]
        acc = accuracy_score(group_y_true, group_y_pred)
        accuracies.append(acc)

    pairwise_differences = [abs(acc1 - acc2) for i, acc1 in enumerate(accuracies)
                            for j, acc2 in enumerate(accuracies) if i != j]
    return max(pairwise_differences)


def overall_false_positive_rate_ratio(y_true: pd.Series,
                                      y_pred: np.array,
                                      group_indices_list: list) -> float:
    """
    Compute the maximum pairwise ratio of false_positive_rates among all groups.

    Parameters:
    - y_true : pd.Series
        True labels.
    - y_pred : np.array
        Predicted labels.
    - group_indices_list : list
        List of group indices.

    Returns:
    - float
        Maximum pairwise ratio of false_positive_rates.
    """
    if len(group_indices_list) == 2:
        return false_positive_rate_ratio(y_true, y_pred, group_indices_list)
    fpr_values = []
    for group_indices in group_indices_list:
        group_y_true = y_true.iloc[group_indices]
        group_y_pred = y_pred[group_indices]
        fpr = false_positive_rate(group_y_true, group_y_pred)
        fpr_values.append(fpr)

    pairwise_ratios = [fpr1 / fpr2 for i, fpr1 in enumerate(
        fpr_values) for j, fpr2 in enumerate(fpr_values) if i != j]
    if np.isnan(max(pairwise_ratios)) or np.isinf(max(pairwise_ratios)):
        return 1
    return min(max(pairwise_ratios), 1)


def overall_false_negative_rate_ratio(y_true: pd.Series,
                                      y_pred: np.array,
                                      group_indices_list: list) -> float:
    """
    Compute the maximum pairwise ratio of false_negative_rates among all groups.

    Parameters:
    - y_true : pd.Series
        True labels.
    - y_pred : np.array
        Predicted labels.
    - group_indices_list : list
        List of group indices.

    Returns:
    - float
        Maximum pairwise ratio of false_negative_rates.
    """
    if len(group_indices_list) == 2:
        return false_negative_rate_ratio(y_true, y_pred, group_indices_list)
    fnr_values = []
    for group_indices in group_indices_list:
        group_y_true = y_true.iloc[group_indices]
        group_y_pred = y_pred[group_indices]
        fnr = false_negative_rate(group_y_true, group_y_pred)
        fnr_values.append(fnr)

    pairwise_ratios = [fnr1 / fnr2 for i, fnr1 in enumerate(
        fnr_values) for j, fnr2 in enumerate(fnr_values) if i != j]
    if np.isnan(max(pairwise_ratios)) or np.isinf(max(pairwise_ratios)):
        return 1
    return min(max(pairwise_ratios), 1)


def overall_demographic_parity_difference(y_true: pd.Series,
                                          y_pred: np.array,
                                          group_indices_list: list) -> float:
    """
    Compute the maximum absolute difference in selection rates between each group and the overall selection rate.

    Parameters:
    - y_true : pd.Series
        True labels.
    - y_pred : np.array
        Predicted labels.
    - group_indices_list : list
        List of group indices.

    Returns:
    - float
        Maximum absolute difference in selection rates.
    """
    if len(group_indices_list) == 2:
        return demographic_parity_difference(y_true, y_pred, group_indices_list)
    selection_rates = []
    for group_indices in group_indices_list:
        group_y_true = y_true.iloc[group_indices]
        group_y_pred = y_pred[group_indices]
        selection_rate_group = selection_rate(group_y_true, group_y_pred)
        selection_rates.append(selection_rate_group)

    overall_selection_rate = selection_rate(y_true, y_pred)
    pairwise_differences = [abs(sr - overall_selection_rate)
                            for sr in selection_rates]
    return max(pairwise_differences)


def overall_equalized_odds_difference(y_true: pd.Series,
                                      y_pred: np.array,
                                      group_indices_list: list) -> float:
    """
    Compute the maximum pairwise difference in equalized odds among all groups.

    Parameters:
    - y_true : pd.Series
        True labels.
    - y_pred : np.array
        Predicted labels.
    - group_indices_list : list
        List of group indices.

    Returns:
    - float
        Maximum pairwise difference in equalized odds.
    """
    if len(group_indices_list) == 2:
        return equalized_odds_difference(y_true, y_pred, group_indices_list)
    tpr_values = []
    fpr_values = []
    for group_indices in group_indices_list:
        group_y_true = y_true.iloc[group_indices]
        group_y_pred = y_pred[group_indices]
        tpr = true_positive_rate(group_y_true, group_y_pred)
        fpr = false_positive_rate(group_y_true, group_y_pred)
        tpr_values.append(tpr)
        fpr_values.append(fpr)

    pairwise_differences = [abs(tpr1 - fpr1 - (tpr2 - fpr2)) for i, (tpr1, fpr1) in enumerate(zip(
        tpr_values, fpr_values)) for j, (tpr2, fpr2) in enumerate(zip(tpr_values, fpr_values)) if i != j]
    return max(pairwise_differences)
