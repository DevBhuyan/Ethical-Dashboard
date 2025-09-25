#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 01:10:38 2025

@author: dev
"""


import streamlit as st
import pandas as pd


def display_dataset(df: pd.DataFrame):
    """
    Displays a Pandas DataFrame nicely in Streamlit with:
    - Dataset info (shape, columns, types)
    - Quick statistics
    - Optional preview (top N rows)
    - Search/filter support
    """
    st.markdown("### Dataset Overview")

    st.write(f"**Number of rows:** {df.shape[0]}")
    st.write(f"**Number of columns:** {df.shape[1]}")

    col_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [df[col].dtype for col in df.columns],
        "Missing Values": [df[col].isna().sum() for col in df.columns],
        "Unique Values": [df[col].nunique() for col in df.columns]
    })
    st.markdown("**Column Information:**")
    st.dataframe(col_info,
                 use_container_width=True)

    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 0:
        st.markdown("**Numeric Summary:**")
        st.dataframe(df[numeric_cols].describe().T,
                     use_container_width=True)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        st.markdown("**Categorical Summary:**")
        cat_summary = pd.DataFrame({
            col: df[col].value_counts().head(5).to_dict()
            for col in categorical_cols
        }).T
        st.dataframe(cat_summary,
                     use_container_width=True)

    st.markdown("**Preview of Data (first 10 rows):**")
    st.dataframe(df.head(10),
                 use_container_width=True)

    st.markdown("**Select columns to display:**")
    selected_cols = st.multiselect(
        "Columns",
        df.columns.tolist(),
        default=df.columns.tolist()
    )
    st.dataframe(df[selected_cols],
                 use_container_width=True)


def edit_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Allows users to edit a Pandas DataFrame in-place.
    Should be called on a dedicated 'Edit Dataset' page.

    Returns the edited DataFrame.
    """
    st.markdown("### Edit Dataset")

    st.info("You can edit values directly in the table below. "
            "Once you are done, the changes will be reflected in the returned DataFrame.")

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True
    )

    if st.button("Save Changes"):
        st.success("Changes saved!")
        return edited_df

    return edited_df
