#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 01:13:51 2025

@author: Dhananjoy Bhuyan
"""


from ethical_eval_pages import (
    fairness_eval,
    robustness_eval,
    privacy_eval,
    explainability_eval
)
import subprocess
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import asyncio
import streamlit as st
from session_state_attrib import (
    ss,
    save_configuration_as_previous,
    load_previous_configuration
)
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
    data_card,
    dataset_card
)
from train_model import (
    train,
    eval_model
)
from model_viewer import (
    model_details,
    model_card
)


def previous_config():

    if load_previous_configuration():

        with st.expander("Previously used Configuration", expanded=True):

            col1, col2, col3 = st.columns(3)

            with col1:
                data_card(ss.selected_dataset, freeze_dataset=True)

            with col2:
                model_card()

            with col3:
                training_card()


def data_home():

    if not load_previous_configuration():
        st.subheader("To start with, let us choose a Dataset:")

    else:
        st.subheader("Or select your own configuration")

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
        ss.selected_dataset_name = upload.name
        ss.selected_dataset = load_dataset_from_st_upload(upload)
        ss.page = "view_dataset"
        st.rerun()


def browse_dataset():

    cols = st.columns(5)

    all_datasets = load_all_datasets()

    st.divider()
    st.subheader("Available datasets")

    for idx, (dataset_name, df) in enumerate(all_datasets.items()):

        with cols[idx % 5]:

            with st.container(height=550):
                dataset_card(df,
                             dataset_name)

            if st.button(f"Select {dataset_name}",
                         type="primary",
                         width='stretch',
                         key=f"select_{dataset_name}"):

                ss.selected_dataset_name = dataset_name
                ss.selected_dataset = df
                ss.page = "view_dataset"
                st.rerun()


def view_dataset():

    display_dataset(ss.selected_dataset)

    if 'selected_model_name' in ss:
        _, col1, col2, col3, col4, _ = st.columns(6)
    else:
        _, col1, col2, col3, _ = st.columns(5)

    with col1:
        if st.button("Choose a different dataset",
                     width='stretch'):
            ss.page = "data_home"
            st.rerun()

    with col2:
        if st.button("Edit data",
                     width='stretch'):
            ss.page = "edit_data"
            st.rerun()

    with col3:
        if st.button("Proceed to model selection",
                     width='stretch',
                     type='primary'):
            ss.page = "model_home"
            st.rerun()

    if 'selected_model_name' in ss:
        with col4:
            if st.button(f"Train with {ss.selected_model_name}",
                         width='stretch',
                         type='primary'):
                ss.selected_model = lazy_load_model(
                    ss.selected_model_name
                )[
                    ss.selected_model_name
                ]
                with st.spinner("Training model....\nThis may take a while"):
                    train()
                asyncio.run(toaster())
                ss.page = "training_results"
                st.rerun()


def edit_data():

    edited_df = edit_dataset(ss.selected_dataset)

    if 'selected_model_name' in ss:
        _, col1, col2, col3, col4, _ = st.columns(6)
    else:
        _, col1, col2, col3, _ = st.columns(5)

    with col1:
        if st.button("Choose a different dataset",
                     width='stretch'):
            ss.page = "data_home"
            st.rerun()

    with col2:
        if st.button("Confirm Changes",
                     width='stretch'):
            ss.selected_dataset = edited_df
            ss.page = "view_dataset"
            st.rerun()

    with col3:
        if st.button("Proceed to model selection",
                     width='stretch',
                     type='primary'):
            ss.page = "model_home"
            st.rerun()

    if 'selected_model_name' in ss:
        with col4:
            if st.button(f"Train with {ss.selected_model_name}",
                         width='stretch',
                         type='primary'):
                ss.selected_model = lazy_load_model(
                    ss.selected_model_name
                )[
                    ss.selected_model_name
                ]
                with st.spinner("Training model....\nThis may take a while"):
                    train()
                asyncio.run(toaster())
                ss.page = "training_results"
                st.rerun()


def model_home():

    col1, col2 = st.columns([1, 3])

    with col1:

        st.subheader("Input data")
        with st.expander("", expanded=True):
            data_card(ss.selected_dataset, freeze_dataset=True)

        if st.button("Choose a different dataset",
                     width='stretch'):
            ss.page = "data_home"
            st.rerun()

    with col2:
        st.subheader("Select a Base Model to train on the data")

        names = eager_load_all_models(names_only=True)
        names.insert(0, "Select a model")
        choice = st.selectbox('datasets', names)

        if choice != "Select a model":
            ss.selected_model_name = choice
            ss.selected_model = lazy_load_model(choice)[choice]
            ss.page = "view_model"
            st.rerun()

        st.divider()

        upload = st.file_uploader(
            "OR Upload your custom model file",
            type=['pkl', 'h5', 'hdf5', 'safetensors']
        )

        if upload:
            ss.selected_model_name = upload.name
            ss.selected_model = load_model_from_st_upload(upload)
            ss.page = "view_model"
            st.rerun()


def view_model():

    model_details()

    _, col1, col2, _ = st.columns([2, 3, 3, 2])

    with col1:
        if st.button("Choose a different model",
                     width='stretch'):
            ss.page = "model_home"
            st.rerun()

    with col2:
        if st.button("Start Training",
                     width='stretch',
                     type='primary'):
            with st.spinner("Training model....\nThis may take a while"):
                train()
            asyncio.run(toaster())
            ss.page = "training_results"
            st.rerun()


def training_card():

    st.title("📊 Model Evaluation Results")

    if not hasattr(ss, "eval_metrics"):
        st.warning(
            "⚠️ No evaluation results found. Please train and evaluate a model first."
        )

        return

    save_configuration_as_previous()

    metrics = ss.eval_metrics

    st.subheader("✅ Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    col2.metric("Precision", f"{metrics['precision']:.4f}")
    col3.metric("Recall", f"{metrics['recall']:.4f}")
    col4.metric("F1 Score", f"{metrics['f1_score']:.4f}")

    col1, col2 = st.columns(2)

    with col2:
        if st.button("Proceed to Ethical Evaluation",
                     width='stretch',
                     type="primary"):
            with st.spinner("Training model....\nThis may take a while"):
                train()
            asyncio.run(toaster())
            ss.page = "ethical_eval"
            st.rerun()


def training_results():
    st.title("📊 Model Evaluation Results")
    eval_model()

    if not hasattr(ss, "eval_metrics"):
        st.warning(
            "⚠️ No evaluation results found. Please train and evaluate a model first."
        )

        return

    save_configuration_as_previous()

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

    _, col1, col2, _ = st.columns([2, 3, 3, 2])

    with col1:
        if st.button("Choose a different configuration",
                     width='stretch'):
            ss.page = "model_home"
            st.rerun()

    with col2:
        if st.button("Proceed to Ethical Evaluation",
                     width='stretch',
                     type='primary'):
            with st.spinner("Training model....\nThis may take a while"):
                train()
            asyncio.run(toaster())
            ss.page = "ethical_eval"
            st.rerun()


def ethical_eval():

    tab_names = [
        "Fairness",
        "Robustness",
        "Privacy",
        "Explainability"
    ]

    if st.button("Back to Home"):
        ss.page = "data_home"
        st.rerun()

    tabs = st.tabs(tab_names)

    ethical_pages = [
        fairness_eval,
        robustness_eval,
        privacy_eval,
        explainability_eval
    ]

    for idx, tab in enumerate(tabs):
        with tab:
            ethical_pages[idx]()


async def toaster():
    st.toast("Training Complete")


def run_command_live(command, stop_flag):
    """Run a command and yield lines of output in real time."""
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        if stop_flag["stop"]:
            process.terminate()
            yield "\n❌  Command cancelled by user."
            break
        yield line.rstrip()
    process.wait()


def dev_console(secret_key="1234"):
    st.markdown("### 🧠 Developer Console")

    if "console_unlocked" not in st.session_state:
        st.session_state.console_unlocked = False
    if not st.session_state.console_unlocked:
        key = st.text_input("🔐 Enter Developer Key", type="password")
        if key == secret_key:
            st.session_state.console_unlocked = True
            st.success("✅ Console unlocked")
        else:
            st.stop()

    st.caption("Type a one-line Linux command and press Enter.")
    command = st.text_input(
        ">", placeholder="e.g. pip install streamlit", key="cmd_input")

    if "stop_flag" not in st.session_state:
        st.session_state.stop_flag = {"stop": False}

    stop_button = st.button("🛑 Stop")

    if stop_button:
        st.session_state.stop_flag["stop"] = True

    if command:
        st.session_state.stop_flag["stop"] = False
        output_area = st.empty()
        output_lines = []

        for line in run_command_live(command, st.session_state.stop_flag):
            output_lines.append(line)
            output_area.code("\n".join(output_lines), language="bash")

        st.success("✅ Done")
        st.session_state.cmd_input = ""  # clear input


def debug_info():

    with st.sidebar:

        st.header("Debug Bar")

        if st.button("Dev Console"):
            ss.page = "dev_console"
            st.rerun()

        st.subheader("Current Session State")
        disp_dct = {}
        for k, v in ss.items():
            if isinstance(v, bool):
                disp_dct[k[:25]] = v
            else:
                try:
                    disp_dct[k[:25]] = v[:50]
                except:
                    disp_dct[k[:25]] = v

        try:
            st.write(disp_dct)
        except:
            for k, v in disp_dct.items():
                st.write({k: v})
