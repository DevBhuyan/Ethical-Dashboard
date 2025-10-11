#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fairlearn.metrics import (
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate
)
from fairlearn.reductions import (
    DemographicParity,
    EqualizedOdds,
    TruePositiveRateParity,
    FalsePositiveRateParity
)
from .custom_metrics import (
    overall_equalized_odds_difference,
    overall_accuracy_score_difference,
    overall_demographic_parity_difference,
    overall_false_negative_rate_ratio,
    overall_false_positive_rate_ratio
)
from .train_rf import train_accurate_rf
from .train_dt import train_accurate_dt
from .train_lr import train_accurate_lr
from .train_nb import train_accurate_nb
from .train_svm import train_accurate_svm
from .EDA import get_data
from sklearn.metrics import accuracy_score, precision_score
import pickle
from .helpers import (
    plot_wo_save,
    build_fair_model,
    bcolors
)
import json
from datetime import datetime
from warnings import simplefilter
simplefilter("ignore")

START = datetime.now()
MODEL = "nb"
DATASET = "adult"


REDUCTION_METHODS = {
    "demographic_parity": DemographicParity,
    "equalized_odds": EqualizedOdds,
    "true_positive_rate_parity": TruePositiveRateParity,
    "false_positive_rate_parity": FalsePositiveRateParity
}

OPTIMAL_REDUCTIONS = {
    "adult": {
        "svm": {
            "sex": list(REDUCTION_METHODS.items())[1],
            "race": list(REDUCTION_METHODS.items())[0],
            "marital-status": list(REDUCTION_METHODS.items())[0]
        },
        "rf": {
            "sex": list(REDUCTION_METHODS.items())[0],
            "race": list(REDUCTION_METHODS.items())[0],
            "marital-status": list(REDUCTION_METHODS.items())[0]
        },
        "lr": {
            "sex": list(REDUCTION_METHODS.items())[0],
            "race": list(REDUCTION_METHODS.items())[1],
            "marital-status": list(REDUCTION_METHODS.items())[3]
        },
        "nb": {
            "sex": list(REDUCTION_METHODS.items())[0],
            "race": list(REDUCTION_METHODS.items())[0],
            "marital-status": list(REDUCTION_METHODS.items())[3]
        },
        "dt": {
            "sex": list(REDUCTION_METHODS.items())[3],
            "race": list(REDUCTION_METHODS.items())[0],
            "marital-status": list(REDUCTION_METHODS.items())[1]
        }
    },
    "german": {
        "svm": {
            "Sex": list(REDUCTION_METHODS.items())[3],
            "Job": list(REDUCTION_METHODS.items())[2],
            "Housing": list(REDUCTION_METHODS.items())[3]
        },
        "lr": {
            "Sex": list(REDUCTION_METHODS.items())[2],
            "Job": list(REDUCTION_METHODS.items())[1],
            "Housing": list(REDUCTION_METHODS.items())[2]
        },
        "nb": {
            "Sex": list(REDUCTION_METHODS.items())[0],
            "Job": list(REDUCTION_METHODS.items())[1],
            "Housing": list(REDUCTION_METHODS.items())[3]
        },
        "rf": {
            "Sex": list(REDUCTION_METHODS.items())[3],
            "Job": list(REDUCTION_METHODS.items())[2],
            "Housing": list(REDUCTION_METHODS.items())[0]
        },
        "dt": {
            "Sex": list(REDUCTION_METHODS.items())[3],
            "Job": list(REDUCTION_METHODS.items())[0],
            "Housing": list(REDUCTION_METHODS.items())[0]
        }
    }
}

MODEL_FUNCS = {
    "rf": train_accurate_rf,
    "dt": train_accurate_dt,
    "lr": train_accurate_lr,
    "nb": train_accurate_nb,
    "svm": train_accurate_svm
}

METRICS = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "false_positive_rate": false_positive_rate,
    "false_negative_rate": false_negative_rate,
    "selection_rate": selection_rate,
    "count": count,
    "accuracy_score_diff": overall_accuracy_score_difference,
    "false_positive_rate_ratio": overall_false_positive_rate_ratio,
    "false_negative_rate_ratio": overall_false_negative_rate_ratio,
    "demographic_parity_difference": overall_demographic_parity_difference,
    "equalized_odds_diff": overall_equalized_odds_difference,
}


def run_all_methods():
    file = open(f"./results/consolidated/{DATASET} {MODEL} model.txt", "w")

    print(f"{bcolors.BOLD}Dataset: {DATASET}; Model: {MODEL}\n{bcolors.ENDC}")
    file.write(f"Dataset: {DATASET}; Model: {MODEL}\n")

    data, sensitive_features = get_data(DATASET)
    accurate_model, _, y_pred = MODEL_FUNCS[MODEL](data)

    if len(sensitive_features) == 1:
        sensitive_features = [sensitive_features]

    # %% Plotting before debiasing
    for sensitive_feature in sensitive_features:
        print(f"{bcolors.BOLD}Sensitive Feature: {sensitive_feature}{bcolors.ENDC}")
        file.write(
            f"Sensitive Feature: {sensitive_feature}")
        fig, results = plot_wo_save(
            data,
            sensitive_feature,
            METRICS,
            y_pred,
            silent=True
        )

        file_name = f"./results/fairness_metrics_of_{data['name']}_dataset_based_on_{sensitive_feature}_using_{MODEL}_before.png"
        json_name = f"./results/fairness_metrics_of_{data['name']}_dataset_based_on_{sensitive_feature}_using_{MODEL}_before.json"
        fig.figure.savefig(file_name)
        with open(json_name, "w") as f:
            json.dump(results, f)
        print(
            f"{bcolors.OKGREEN}Bias measure: {results['BIAS_MEASURE']}\n{bcolors.ENDC}")
        file.write(
            f"\nBias measure: {results['BIAS_MEASURE']}\n")
        original_bias = results['BIAS_MEASURE']

        # %% Build fairer models for each sensitive feature
        for reduction_name, reduction_method in REDUCTION_METHODS.items():

            sensitive_feature_train = data["X_train"][sensitive_feature]
            sensitive_feature_test = data["X_test"][sensitive_feature]

            print(f"Building a fairer model using {reduction_name}")
            file.write(f"Building a fairer model using {reduction_name}")
            fair_pred, fair_model = build_fair_model(
                accurate_model, data, sensitive_feature_train, sensitive_feature_test, reduction=reduction_method, reduction_name=reduction_name)

            # %% Plot after debiasing
            fig, results = plot_wo_save(
                data, sensitive_feature, METRICS, fair_pred, silent=True)

            file_name = f"./results/fairness_metrics_of_{data['name']}_dataset_based_on_{sensitive_feature}_using_{MODEL}_after_{reduction_name}.png"
            json_name = f"./results/fairness_metrics_of_{data['name']}_dataset_based_on_{sensitive_feature}_using_{MODEL}_after_{reduction_name}.json"
            fig.figure.savefig(file_name)
            with open(json_name, "w") as f:
                json.dump(results, f)

            print(
                f"{bcolors.OKGREEN}Bias measure: {results['BIAS_MEASURE']}; Improvement of {original_bias - results['BIAS_MEASURE']}% \n{bcolors.ENDC}")
            file.write(
                f"\nBias measure: {results['BIAS_MEASURE']}; Improvement of {original_bias - results['BIAS_MEASURE']}% \n")

            path = f"./models/{data['name']}_{MODEL}_fair_based_on_{sensitive_feature}_using_{reduction_name}.pkl"
            with open(path, "wb") as f:
                pickle.dump(fair_model, f)

    file.write(f"\nTime Elapsed: {datetime.now()-START}")
    file.close()


def run_with_threshold():

    file = open(
        f"./results/consolidated/{DATASET} {MODEL} model with threshold.txt", "w")

    print(f"{bcolors.BOLD}Dataset: {DATASET}; Model: {MODEL}\n{bcolors.ENDC}")
    file.write(f"Dataset: {DATASET}; Model: {MODEL}\n")

    data, sensitive_features = get_data(DATASET)
    accurate_model, _, y_pred = MODEL_FUNCS[MODEL](data)

    if len(sensitive_features) == 1:
        sensitive_features = [sensitive_features]

    # %% Plotting before debiasing
    for sensitive_feature in sensitive_features:

        reduction_name, reduction_method = OPTIMAL_REDUCTIONS[DATASET][MODEL][sensitive_feature]

        print(f"{bcolors.BOLD}Sensitive Feature: {sensitive_feature}{bcolors.ENDC}")
        file.write(
            f"Sensitive Feature: {sensitive_feature}")
        fig, results = plot_wo_save(
            data, sensitive_feature, METRICS, y_pred, silent=True)

        file_name = f"./results/fairness_metrics_of_{data['name']}_dataset_based_on_{sensitive_feature}_using_{MODEL}_before.png"
        json_name = f"./results/fairness_metrics_of_{data['name']}_dataset_based_on_{sensitive_feature}_using_{MODEL}_before.json"
        fig.figure.savefig(file_name)
        with open(json_name, "w") as f:
            json.dump(results, f)
        print(
            f"{bcolors.OKGREEN}Bias measure: {results['BIAS_MEASURE']}\n{bcolors.ENDC}")
        file.write(
            f"\nBias measure: {results['BIAS_MEASURE']}\n")
        original_bias = results['BIAS_MEASURE']

        # %% Build fairer models for each sensitive feature

        sensitive_feature_train = data["X_train"][sensitive_feature]
        sensitive_feature_test = data["X_test"][sensitive_feature]

        print(
            f"Building a fairer model using {reduction_name} and thresholding")
        file.write(
            f"Building a fairer model using {reduction_name} and thresholding")
        fair_pred, fair_model = build_fair_model(
            accurate_model, data, sensitive_feature_train, sensitive_feature_test, reduction=reduction_method, reduction_name=reduction_name, threshold=True)

        # %% Plot after debiasing
        fig, results = plot_wo_save(
            data, sensitive_feature, METRICS, fair_pred, silent=True)

        file_name = f"./results/fairness_metrics_of_{data['name']}_dataset_based_on_{sensitive_feature}_using_{MODEL}_after_{reduction_name} and thresholding.png"
        json_name = f"./results/fairness_metrics_of_{data['name']}_dataset_based_on_{sensitive_feature}_using_{MODEL}_after_{reduction_name} and thresholding.json"
        fig.figure.savefig(file_name)
        with open(json_name, "w") as f:
            json.dump(results, f)

        print(
            f"{bcolors.OKGREEN}Bias measure: {results['BIAS_MEASURE']}; Improvement of {original_bias - results['BIAS_MEASURE']}% \n{bcolors.ENDC}")
        file.write(
            f"\nBias measure: {results['BIAS_MEASURE']}; Improvement of {original_bias - results['BIAS_MEASURE']}% \n")

        path = f"./models/{data['name']}_{MODEL}_fair_based_on_{sensitive_feature}_using_{reduction_name} and thresholding.pkl"
        with open(path, "wb") as f:
            pickle.dump(fair_model, f)

    file.write(f"\nTime Elapsed: {datetime.now()-START}")
    file.close()
    
    
if __name__ == "__main__":
	run_all_methods()
