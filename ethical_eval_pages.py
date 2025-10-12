#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 23:59:23 2025

@author: dev
"""


from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np
import shap
from data_fairness.main import (
    plot_wo_save,
    METRICS
)
from session_state_attrib import ss
import streamlit as st


def fairness_eval():

    y_pred = ss.trained_model.predict(ss.X_test)

    data = {
        "name": ss.selected_dataset_name,
        "X_train": ss.X_train,
        "X_test": ss.X_test,
        "y_train": ss.y_train,
        "y_test": ss.y_test,
        "category_maps": ss.category_maps
    }

    for sensitive_feature in ss.sensitive_attributes[ss.selected_dataset_name]:
        fig, results = plot_wo_save(
            data,
            sensitive_feature,
            METRICS,
            y_pred,
            silent=False
        )
        st.pyplot(fig)
        st.write(results)


def robustness_eval():
    pass


def privacy_eval():
    pass


def explainability_eval():
    """
    Evaluate explainability of a trained model using SHAP and present it in Streamlit.
    Works for regression, binary, and multi-class models.
    """

    st.header("🔍 Model Explainability Evaluation")

    model = ss.trained_model
    X_train = ss.X_train
    X_test = ss.X_test

    sample_size = 200
    if len(X_test) > sample_size:
        X_sample = X_test.sample(sample_size, random_state=42)
    else:
        X_sample = X_test.copy()

    # Choose SHAP explainer
    model_name = model.__class__.__name__.lower()
    if "xgb" in model_name or "lgbm" in model_name or "tree" in model_name or "forest" in model_name:
        explainer = shap.TreeExplainer(model)
    elif "linear" in model_name:
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.Explainer(model, X_train)

    st.info(f"Using **{explainer.__class__.__name__}** for SHAP explainability")

    shap_values = explainer(X_sample)

    # --- Global Plots ---
    st.subheader("🌍 Global Explainability")
    st.markdown(
        "These plots show which features most influence the model overall.")

    # Summary Plot (beeswarm)
    fig_summary, ax = plt.subplots()
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(fig_summary)
    plt.close(fig_summary)

    # Mean |SHAP| Importance Plot
    fig_bar, ax = plt.subplots()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    # --- Local Explanation ---
    st.subheader("🔬 Local Explanation")
    st.markdown(
        "Below is a local SHAP waterfall plot for a randomly selected sample. "
        "It shows which features pushed the prediction up or down."
    )

    idx = np.random.randint(0, len(X_sample))
    sample = X_sample.iloc[[idx]]

    if shap_values.values.ndim == 3:
        # Multi-class case
        pred = model.predict(sample)
        if pred.ndim == 1:
            class_idx = int(pred[0])
        else:
            class_idx = int(np.argmax(pred, axis=1)[0])

        single_expl = shap.Explanation(
            values=shap_values.values[idx, class_idx],
            base_values=shap_values.base_values[idx][class_idx]
            if shap_values.base_values.ndim > 1 else shap_values.base_values[idx],
            data=sample.values[0],
            feature_names=list(X_sample.columns)
        )
        title = f"Local Explanation for Sample #{idx} (Class {class_idx})"
    else:
        # Binary / regression
        single_expl = shap.Explanation(
            values=shap_values.values[idx],
            base_values=shap_values.base_values[idx]
            if hasattr(shap_values, "base_values") else 0,
            data=sample.values[0],
            feature_names=list(X_sample.columns)
        )
        title = f"Local Explanation for Sample #{idx}"

    fig_local, ax = plt.subplots()
    shap.plots.waterfall(single_expl, show=False)
    plt.title(title)
    plt.tight_layout()
    st.pyplot(fig_local)
    plt.close(fig_local)

    # --- Explainability Metrics ---
    st.subheader("📊 Quantitative Explainability Metrics")

    shap_matrix = np.array(shap_values.values)
    if shap_matrix.ndim == 3:
        shap_matrix = shap_matrix.mean(axis=1)

    dist = pairwise_distances(X_sample)
    np.fill_diagonal(dist, np.inf)
    nearest_idx = np.argmin(dist, axis=1)
    cos_sim = np.sum(shap_matrix * shap_matrix[nearest_idx], axis=1) / (
        np.linalg.norm(shap_matrix, axis=1) *
        np.linalg.norm(shap_matrix[nearest_idx], axis=1)
    )
    stability_score = np.nanmean(cos_sim)

    thresh = np.percentile(np.abs(shap_matrix), 75)
    sparsity = np.mean(np.sum(np.abs(shap_matrix) > thresh, axis=1))

    st.metric("Stability", f"{stability_score:.3f}",
              help="How consistent SHAP explanations are for similar samples.")
    st.metric("Sparsity", f"{sparsity:.2f}",
              help="Average number of dominant features per sample explanation.")

    with st.expander("🧾 Raw SHAP Objects"):
        st.write("**SHAP Explainer:**", explainer.__class__.__name__)
        st.write("**SHAP Values Shape:**", np.array(shap_values.values).shape)
        st.write("**Base Values:**", shap_values.base_values[:5])

    results = {
        "explainer": explainer,
        "shap_values": shap_values,
        "metrics": {
            "stability": float(stability_score),
            "sparsity": float(sparsity)
        }
    }

    return results
