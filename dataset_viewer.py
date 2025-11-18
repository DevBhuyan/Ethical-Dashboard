#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 01:10:38 2025

@author: dev
"""


import streamlit as st
import pandas as pd
from load_datasets import (
    unique_count,
    infer_sensitive_attributes,
    load_dataset_from_st_upload,
    load_all_datasets,
    load_specific_dataset
)
from session_state_attrib import ss
from time import perf_counter_ns


DEFAULT_CONTAINER_HEIGHT = 400


def data_card(df,
              freeze_dataset: bool = False,
              interactive: bool = False,
              dataset_name: str = ""):

    names = load_all_datasets(names_only=True)

    if not freeze_dataset:
        choice = st.selectbox(
            label='Selected Dataset',
            options=names,
            index=names.index(ss.selected_dataset_name),
            key=str(perf_counter_ns())
        )

        if choice != ss.selected_dataset_name:
            ss.selected_dataset_name = choice
            ss.selected_dataset = load_specific_dataset(
                dset_name=choice
            )[choice]
            st.rerun()

    sensitive_attributes = infer_sensitive_attributes(df)

    if not dataset_name:
        st.write(f"Dataset name: **{ss.selected_dataset_name}**")
        ss.sensitive_attributes[ss.selected_dataset_name] = sensitive_attributes
    else:
        st.write(f"Dataset name: **{dataset_name}**")
        ss.sensitive_attributes[dataset_name] = sensitive_attributes

    st.info(
        f"Contains {df.shape[1] - 1} **features** and {df.shape[0]} **samples** | data points divided into {unique_count(df['Class'])} **Classes**"
    )
    if sensitive_attributes:
        if len(sensitive_attributes) > 1:
            st.success(
                f"Contains {len(sensitive_attributes)} sensitive attributes: {', '.join(sensitive_attributes)}"
            )
        else:
            st.success(
                f"Contains {len(sensitive_attributes)} sensitive attributes: {sensitive_attributes[0]}"
            )


def dataset_card(df: pd.DataFrame,
                 dataset_name: str):

    data_card(df,
              freeze_dataset=True,
              dataset_name=dataset_name)

    col_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [df[col].dtype for col in df.columns],
        "Missing Values": [df[col].isna().sum() for col in df.columns],
        "Unique Values": [df[col].nunique() for col in df.columns]
    })

    numeric_cols = df.select_dtypes(include='number').columns

    with st.expander("Column Information"):
        st.write("**Column Information:**")
        st.dataframe(col_info,
                     width='stretch',
                     hide_index=True)

    if len(numeric_cols) > 0:

        with st.expander("Numeric Summary"):
            st.write("**Numeric Summary:**")
            st.dataframe(df[numeric_cols].describe().T,
                         width='stretch')

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        with st.expander("Categorical Summary"):
            st.write("**Categorical Summary:**")
            cat_summary = pd.DataFrame({
                col: df[col].value_counts().head(5).to_dict()
                for col in categorical_cols
            }).T
            st.dataframe(cat_summary,
                         width='stretch')

    with st.expander("Dataset Preview"):
        st.write("**Preview of Data (first 10 rows):**")
        st.dataframe(df.head(10),
                     width='stretch')

    with st.expander("View Column wise data"):
        st.write("**Select columns to display:**")
        selected_cols = st.multiselect(
            "Columns",
            df.columns.tolist(),
            default=df.columns.tolist()
        )
        st.dataframe(df[selected_cols],
                     width='stretch')


def display_dataset(df: pd.DataFrame):
    """
    Displays a Pandas DataFrame nicely in Streamlit with:
    - Dataset info (shape, columns, types)
    - Quick statistics
    - Optional preview (top N rows)
    - Search/filter support
    """
    st.write("### Dataset Overview")

    data_card(df)

    col_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [df[col].dtype for col in df.columns],
        "Missing Values": [df[col].isna().sum() for col in df.columns],
        "Unique Values": [df[col].nunique() for col in df.columns]
    })

    col1, col2 = st.columns(2)

    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 0:

        with col1:
            with st.container(height=DEFAULT_CONTAINER_HEIGHT):
                st.write("**Column Information:**")
                st.dataframe(col_info,
                             width='stretch',
                             hide_index=True)

        with col2:
            with st.container(height=DEFAULT_CONTAINER_HEIGHT):
                st.write("**Numeric Summary:**")
                st.dataframe(df[numeric_cols].describe().T,
                             width='stretch')

    else:
        with st.container(height=DEFAULT_CONTAINER_HEIGHT):
            st.dataframe(col_info,
                         width='stretch')

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        with col1:
            with st.expander("Categorical Summary"):
                st.write("**Categorical Summary:**")
                cat_summary = pd.DataFrame({
                    col: df[col].value_counts().head(5).to_dict()
                    for col in categorical_cols
                }).T
                st.dataframe(cat_summary,
                             width='stretch')

    with col2:
        with st.expander("Dataset Preview"):
            st.write("**Preview of Data (first 10 rows):**")
            st.dataframe(df.head(10),
                         width='stretch')

    with st.expander("View Column wise data"):
        st.write("**Select columns to display:**")
        selected_cols = st.multiselect(
            "Columns",
            df.columns.tolist(),
            default=df.columns.tolist()
        )
        st.dataframe(df[selected_cols],
                     width='stretch')


def edit_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Allows users to edit a Pandas DataFrame in-place.
    Should be called on a dedicated 'Edit Dataset' page.

    Returns the edited DataFrame.
    """
    st.write("### Edit Dataset")

    st.info("You can edit values directly in the table below. "
            "Once you are done, the changes will be reflected in the returned DataFrame.")

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        width='stretch'
    )

    upload = st.file_uploader(
        "OR Edit the dataset in your local editor and re-upload here",
        type=['csv']
    )

    if upload:
        ss.selected_dataset = load_dataset_from_st_upload(upload)
        ss.page = "view_dataset"
        st.rerun()

    return edited_df
