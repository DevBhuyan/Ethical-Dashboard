#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 00:30:13 2025

@author: dev and Dhananjoy Bhuyan
"""


from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from session_state_attrib import ss
from sklearn.impute import SimpleImputer


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess dataset:
    1. Label encode categorical features
    2. Impute missing values (numerical: mean, categorical: most frequent)
    Returns: preprocessed df
    """
    df = df.copy()

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    category_maps = {}

    for col in cat_cols:

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col].astype(str))

        classes = le.classes_
        encoded = le.transform(classes)
        mapping = dict(zip(encoded, classes))

        category_maps[col] = mapping

    for col in df.columns:
        if df[col].dtype in [float, int]:
            imputer = SimpleImputer(strategy='mean')
        else:  # categorical already encoded as int
            imputer = SimpleImputer(strategy='most_frequent')
        df[col] = imputer.fit_transform(df[[col]])

    ss.category_maps = category_maps

    return df


def load_all_datasets(names_only: bool = False):

    src = './datasets/'
    ds_store = pd.read_excel('./Dataset Repository.xlsx',
                             sheet_name="Sheet1")

    classification_dsets = ds_store[ds_store['Type'] == 'Classification']
    classification_dsets.drop(columns=['Sl. No', 'Type'], inplace=True)

    if names_only:
        dsets = [
            ds_name
            for ds_name, file in zip(
                classification_dsets['Name'].values,
                classification_dsets['Location'].values
            )
            if os.path.exists(src + file)
        ]

    else:
        dsets = {
            ds_name: pd.read_csv(src + file)
            for ds_name, file in zip(
                classification_dsets['Name'].values,
                classification_dsets['Location'].values
            )
            if os.path.exists(src + file)
        }

    return dsets


def load_specific_dataset(dset_name: str = "",
                          dset_path: str = ""):

    src = './datasets/'
    ds_store = pd.read_excel('./Dataset Repository.xlsx',
                             sheet_name="Sheet1")

    classification_dsets = ds_store[ds_store['Type'] == 'Classification']
    classification_dsets.drop(columns=['Sl. No', 'Type'], inplace=True)

    if dset_path and dset_name:

        return {
            dset_name: preprocess_data(pd.read_csv(src + dset_path))
        }

    elif dset_name:
        dset_path = src + classification_dsets[
            classification_dsets['Name'] == dset_name
        ]['Location'].values[0]

        return {
            dset_name: preprocess_data(pd.read_csv(dset_path))
        }

    elif dset_path:
        dset_name = src + classification_dsets[
            classification_dsets['Location'] == dset_name
        ]['Name'].values[0]

        return {
            dset_name: preprocess_data(pd.read_csv(dset_path))
        }

    else:
        raise ValueError("Either of dset_name or dset_path must be provided")


def infer_sensitive_attributes(df: pd.DataFrame):

    with open('./sensitive_keywords.txt') as f:
        sensitive_kws = [
            i.strip().lower()
            for i in f.readlines()
        ]

    sensitive_attrs = []
    for col in df.columns:
        if col.lower() in sensitive_kws:
            sensitive_attrs.append(col)

    return sensitive_attrs


def unique_count(series: pd.Series) -> int:
    return series.nunique(dropna=False)


def convert_upload_to_df(upload):
    """Convert Streamlit uploaded file to a Pandas DataFrame (CSV only)."""
    try:
        df = pd.read_csv(upload)
    except Exception as e:
        raise ValueError(f"Error reading uploaded CSV: {e}")
    return df


def validate_dataset(df: pd.DataFrame):
    """
    Validate dataset based on custom rules:
    1. Must be classification dataset
    2. Must have a categorical column named 'Class' such that
       unique_count(Class) < len(df) // 2
    3. May optionally have sensitive attributes
    """
    # Rule 1 & 2a: Must contain "Class" column
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a column named 'Class'.")

    # Rule 2b: Classification criterion
    class_unique = df["Class"].nunique()
    if class_unique >= len(df) // 2:
        raise ValueError(
            f"Invalid 'Class' column: has {class_unique} unique values, "
            f"which is >= half of dataset size ({len(df)//2})."
        )

    # Ensure 'Class' is numeric labels
    le = None
    if df["Class"].dtype == "object":
        le = LabelEncoder()
        df["Class"] = le.fit_transform(df["Class"].astype(str))

    # Rule 3: Sensitive attributes check (optional)
    sensitive_attrs = [col for col in df.columns if "sensitive" in col.lower()]
    if sensitive_attrs:
        print(f"⚠️ Detected sensitive attributes: {sensitive_attrs}")

    return df, le


def load_dataset_from_st_upload(upload):

    df = convert_upload_to_df(upload)
    df = preprocess_data(df)
    validate_dataset(df)

    return df
