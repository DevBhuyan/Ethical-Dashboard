#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 23:59:23 2025

@author: dev
"""


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import pandas as pd
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

    if data["y_test"].nunique() > 2:
        st.error("Fairness scores can only be computed for binary classification datasets. The functionality to compute scores for multiclass datasets is being built.")
        return

    for sensitive_feature in ss.sensitive_attributes[ss.selected_dataset_name]:

        st.subheader(f"Sensitive Feature: {sensitive_feature}")

        fig, results = plot_wo_save(
            data,
            sensitive_feature,
            METRICS,
            y_pred,
            silent=False
        )
        st.pyplot(fig)
        st.write(results)

        st.divider()


def privacy_eval():
    st.subheader("🔒 Privacy Evaluation")
    st.markdown("""
    This component simulates a **Membership Inference Attack (MIA)** baseline.
    It measures how distinguishable training samples are from unseen data using model outputs.
    """)

    model = ss.trained_model
    X_train = ss.X_train
    X_test = ss.X_test

    n_samples = min(300, len(X_train), len(X_test))
    X_member = X_train.sample(n_samples, random_state=42)
    X_nonmember = X_test.sample(n_samples, random_state=42)
    y_member = np.ones(n_samples)
    y_nonmember = np.zeros(n_samples)

    X_attack = pd.concat([X_member, X_nonmember])
    y_attack = np.concatenate([y_member, y_nonmember])

    # Compute probabilities or scores
    with st.spinner("Computing model confidence scores..."):
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_attack)
            if probs.shape[1] == 2:
                scores = probs[:, 1]
            else:
                scores = probs.max(axis=1)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_attack)
        else:
            # fallback: use raw predictions
            preds = model.predict(X_attack)
            scores = preds.astype(
                float) if preds.ndim == 1 else preds.max(axis=1)

    # Evaluate attack success
    auc = roc_auc_score(y_attack, scores)
    st.metric("Membership Inference AUC", f"{auc:.3f}",
              help="1.0 → full leakage, 0.5 → random guess. Lower = more private.")

    # Histogram plot of confidence scores
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(scores[y_attack == 1], bins=20, alpha=0.6, label="Train (Members)")
    ax.hist(scores[y_attack == 0], bins=20,
            alpha=0.6, label="Test (Non-members)")
    ax.set_title("Prediction Confidence Distribution")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # Interpretation
    if auc > 0.75:
        st.error(
            "⚠️ High privacy risk: Model outputs may leak training membership information.")
    elif auc > 0.6:
        st.warning("⚠️ Moderate privacy leakage risk detected.")
    else:
        st.success(
            "✅ Good: Model appears privacy-safe under baseline membership attack.")

    return {
        "membership_auc": float(auc),
        "n_samples": int(n_samples)
    }


def robustness_eval():
    st.subheader("🛡️ Robustness Evaluation")
    st.markdown("""
    This section measures how stable your model is against input perturbations.
    We add Gaussian noise to features and see how accuracy degrades.
    """)

    X_test = ss.X_test
    y_test = ss.y_test
    model = ss.trained_model

    # Subsample for interactivity speed
    sample_size = min(len(X_test), 300)
    X_sample = X_test.sample(sample_size, random_state=42)
    y_sample = y_test.loc[X_sample.index]

    # User control
    # --- Noise Level Selection ---
    st.subheader("🔹 Choose Noise Levels")

    option = st.radio(
        "Noise selection mode:",
        ["Preset", "Custom"],
        horizontal=True
    )

    if option == "Preset":
        noise_levels = st.multiselect(
            "Select Gaussian noise stddev values",
            options=[0.01, 0.05, 0.1, 0.2, 0.3],
            default=[0.01, 0.05, 0.1],
            help="You can select one or multiple noise levels to test robustness."
        )
    else:
        noise_min, noise_max = st.slider(
            "Noise range (stddev)",
            0.0, 0.5, (0.01, 0.1),
            help="Choose a range of noise values; multiple steps will be generated automatically."
        )
        steps = st.number_input("Number of intermediate steps", 1, 10, 3)
        noise_levels = np.linspace(
            noise_min, noise_max, steps).round(3).tolist()

    st.write(f"Selected noise levels: {noise_levels}")

    if not isinstance(noise_levels, (list, tuple, np.ndarray)):
        noise_levels = [noise_levels]

    # Base accuracy
    with st.spinner("Evaluating baseline accuracy..."):
        y_pred_clean = model.predict(X_sample)
        acc_clean = accuracy_score(y_sample, y_pred_clean)
    st.metric("Clean Data Accuracy", f"{acc_clean:.3f}")

    accuracies, drops = [], []
    progress = st.progress(0)
    for i, eps in enumerate(noise_levels):
        X_noisy = X_sample + np.random.normal(0, eps, X_sample.shape)
        acc_noisy = accuracy_score(y_sample, model.predict(X_noisy))
        accuracies.append(acc_noisy)
        drops.append(acc_clean - acc_noisy)
        progress.progress(int(((i+1)/len(noise_levels))*100))

    # Plot results
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(noise_levels, accuracies, marker="o")
    ax.set_xlabel("Noise Std. Dev.")
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness Curve: Accuracy vs. Noise")
    st.pyplot(fig)
    plt.close(fig)

    # Display summary table
    st.write("### Accuracy Degradation Summary")
    df = pd.DataFrame({
        "Noise Std": noise_levels,
        "Accuracy": accuracies,
        "Accuracy Drop": drops
    })
    st.dataframe(df.style.background_gradient(
        cmap="RdYlGn_r", subset=["Accuracy Drop"]))

    mean_drop = np.mean(drops)
    st.success(f"**Average Accuracy Drop:** {mean_drop:.3f}")

    return {
        "baseline_accuracy": float(acc_clean),
        "mean_accuracy_drop": float(mean_drop),
        "accuracy_by_noise": dict(zip(noise_levels, accuracies))
    }


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
