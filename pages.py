#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 01:13:51 2025

@author: Dhananjoy Bhuyan
"""


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import asyncio
import inspect
import streamlit as st
from session_state_attrib import ss
from load_datasets import (
    load_all_datasets,
    load_specific_dataset,
    load_dataset_from_st_upload
)
from load_models import (
    eager_load_all_models,
    lazy_load_model,
    load_model_from_st_upload
)
from dataset_viewer import (
    display_dataset,
    edit_dataset,
    data_card
)
from train_model import (
    train,
    eval_model
)


def data_home():

    st.subheader("To start with, let us choose a Dataset:")

    names = load_all_datasets(names_only=True)
    names.insert(0, "Select a dataset")
    choice = st.selectbox('datasets', names)

    if choice != "Select a dataset":
        ss.selected_dataset_name = choice
        ss.selected_dataset = load_specific_dataset(dset_name=choice)[choice]
        ss.page = "view_dataset"
        st.rerun()

    if st.button('Browse available Datasets'):
        ss.page = "browse_dataset"
        st.rerun()

    st.divider()

    upload = st.file_uploader(
        'Want to upload your custom dataset? Upload here',
        type="csv"
    )

    if upload:
        ss.selected_dataset_name = "Custom"
        ss.selected_dataset = load_dataset_from_st_upload(upload)
        ss.page = "view_dataset"
        st.rerun()


def view_dataset():

    display_dataset(ss.selected_dataset)

    _, col1, col2, col3, _ = st.columns(5)

    with col1:
        if st.button("Choose a different dataset",
                     use_container_width=True,
                     type='primary'):
            ss.page = "data_home"
            st.rerun()

    with col2:
        if st.button("Edit data",
                     use_container_width=True,
                     type='primary'):
            ss.page = "edit_data"
            st.rerun()

    with col3:
        if st.button("Proceed to model selection",
                     use_container_width=True,
                     type='primary'):
            ss.page = "model_home"
            st.rerun()


def edit_data():

    edited_df = edit_dataset(ss.selected_dataset)

    _, col1, col2, col3, _ = st.columns(5)

    with col1:
        if st.button("Choose a different dataset",
                     use_container_width=True,
                     type='primary'):
            ss.page = "data_home"
            st.rerun()

    with col2:
        if st.button("Confirm Changes",
                     use_container_width=True):
            ss.selected_dataset = edited_df
            ss.page = "view_dataset"
            st.rerun()

    with col3:
        if st.button("Proceed to model selection",
                     use_container_width=True,
                     type='primary'):
            ss.page = "model_home"
            st.rerun()


def model_home():

    col1, col2 = st.columns([1, 3])

    with col1:

        st.subheader("Input data")
        data_card(ss.selected_dataset)

    with col2:
        st.subheader("Select a Base Model to train on the data")

        names = eager_load_all_models(names_only=True)
        names.insert(0, "Select a model")
        choice = st.selectbox('datasets', names)

        if choice != "Select a model":
            ss.selected_model = lazy_load_model(choice)[choice]
            ss.page = "view_model"
            st.rerun()

        st.divider()

        upload = st.file_uploader(
            "OR Upload your custom model file",
            type=['pkl', 'h5', 'hdf5', 'safetensors']
        )

        if upload:
            ss.selected_model = load_model_from_st_upload(upload)
            ss.page = "view_model"
            st.rerun()


def view_model():
    """Display useful information about a loaded model in Streamlit."""

    model = ss.selected_model

    st.subheader("Model Summary")
    st.write(f"**Type:** {type(model)}")

    if isinstance(model, dict):
        st.info("Model appears to be raw tensors (from safetensors).")
        st.write(f"Number of tensors: {len(model)}")
        for name, tensor in list(model.items())[:5]:  # Show a few keys
            st.write(f"- {name}: shape {tuple(tensor.shape)}")
        if len(model) > 5:
            st.caption(f"... and {len(model) - 5} more tensors.")
        return

    if "keras" in str(type(model)).lower():
        st.success("Detected Keras model ✅")
        try:
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            st.text("\n".join(stringlist))
        except Exception:
            st.write("Unable to display Keras summary.")

    elif hasattr(model, "get_params"):
        st.success("Detected scikit-learn compatible model ✅")
        params = model.get_params()
        st.write("**Parameters:**")
        st.json(params)

    st.subheader("Available Methods")
    methods = [
        name
        for name, func in inspect.getmembers(
            model,
            predicate=inspect.ismethod
        )
        if not name.startswith("_")
    ]
    st.write(methods)

    # Check if trainable
    if hasattr(model, "fit") or hasattr(model, "train"):
        st.success("This model is trainable (has `fit` or `train`).")
        # Inspect signature
        candidate = model.fit if hasattr(model, "fit") else model.train
        sig = inspect.signature(candidate)
        st.write(f"Signature: `{candidate.__name__}{sig}`")
    else:
        st.warning("This model does not expose `fit` or `train` method.")

    _, col1, col2, _ = st.columns([2, 3, 3, 2])

    with col1:
        if st.button("Choose a different model",
                     use_container_width=True,
                     type='primary'):
            ss.page = "model_home"
            st.rerun()

    with col2:
        if st.button("Start Training",
                     use_container_width=True):
            with st.spinner("Training model....\nThis may take a while"):
                train()
            asyncio.run(toaster)
            ss.page = "training_results"
            st.rerun()


def training_results():
    st.title("📊 Model Evaluation Results")
    eval_model()

    if not hasattr(ss, "eval_metrics"):
        st.warning(
            "⚠️ No evaluation results found. Please train and evaluate a model first.")
        return

    metrics = ss.eval_metrics
    y_test = ss.y_test
    y_pred = ss.trained_model.predict(ss.X_test)

    # Handle Keras predictions (convert probs → labels)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
        if y_test.ndim > 1:
            y_test = np.argmax(y_test, axis=1)
    elif y_pred.ndim > 1:  # binary case from keras
        y_pred = (y_pred > 0.5).astype(int).flatten()
        if y_test.ndim > 1:
            y_test = np.argmax(y_test, axis=1)

    st.subheader("✅ Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    col2.metric("Precision", f"{metrics['precision']:.4f}")
    col3.metric("Recall", f"{metrics['recall']:.4f}")
    col4.metric("F1 Score", f"{metrics['f1_score']:.4f}")

    st.markdown("---")

    st.subheader("📑 Classification Report")
    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.background_gradient(
        cmap="Blues").format("{:.2f}"))

    st.markdown("---")

    st.subheader("🔍 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    st.pyplot(fig)


async def toaster():
    st.toast("Training Complete")


def debug_info():

    with st.sidebar:
        st.subheader("Current Session State")
        disp_dct = {}
        for k, v in ss.items():
            if isinstance(v, bool):
                disp_dct[k[:20]] = v
            else:
                try:
                    disp_dct[k[:20]] = v[:50]
                except:
                    disp_dct[k[:20]] = v

        st.write(disp_dct)
