#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 00:30:13 2025

@author: dev and Dhananjoy Bhuyan
"""


from streamlit.runtime.scriptrunner import get_script_run_ctx
import numpy as np
import logging
import traceback
from warnings import warn
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from session_state_attrib import ss
from sklearn.impute import SimpleImputer


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
            ds_name: preprocess_data(
                pd.read_csv(src + file), ds_name)
            for ds_name, file in zip(
                classification_dsets['Name'].values,
                classification_dsets['Location'].values
            )
            if os.path.exists(src + file)
        }

    return dsets


def infer_sensitive_attributes(df: pd.DataFrame):

    with open('./sensitive_keywords.txt') as f:
        sensitive_kws = [
            i.strip().lower()
            for i in f.readlines()
        ]

    sensitive_attrs = []
    sensitive_attrs = []
    for col in df.columns:
        for kw in sensitive_kws:
            if kw.lower() in col.lower():  # substring match
                sensitive_attrs.append(col)
                break  # stop checking once matched

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
    df = preprocess_data(df, upload.name)
    validate_dataset(df)

    return df


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_problematic_columns(df: pd.DataFrame, sample_n: int = 200):
    """
    Quick inspector to find columns likely to fail Arrow conversion.
    Returns a dict: {col: summary}.
    """
    issues = {}
    for col in df.columns:
        ser = df[col]
        dtype = ser.dtype
        summary = {"dtype": str(dtype)}
        # presence of pandas NA sentinel
        if ser.isna().any():
            # pd.NA is pandas' sentinel; detect extension-nullable types by dtype name too
            summary["has_na"] = True
        else:
            summary["has_na"] = False

        if pd.api.types.is_extension_array_dtype(dtype):
            summary["extension_dtype"] = True
        else:
            summary["extension_dtype"] = False

        if pd.api.types.is_object_dtype(dtype):
            # sample types to detect heterogeneity
            sample = ser.dropna().head(sample_n).map(type).unique().tolist()
            summary["sample_types"] = [t.__name__ for t in sample]
            if len(sample) > 1:
                summary["heterogeneous_object"] = True
            else:
                summary["heterogeneous_object"] = False

            # check for numpy scalar objects (e.g. dtype objects in cells)
            sample_vals = ser.dropna().head(50).tolist()
            contains_numpy_scalars = any(
                hasattr(v, "dtype") and not isinstance(v, str) for v in sample_vals)
            summary["contains_numpy_scalars"] = contains_numpy_scalars

        issues[col] = summary
    return issues


def make_arrow_friendly(df: pd.DataFrame, convert_ints_to_float: bool = True) -> pd.DataFrame:
    """
    Convert DataFrame columns to types that pyarrow/Streamlit reliably accept.
    - Convert pandas extension dtypes (Int64, Float64, boolean) -> numpy-backed types
    - Replace pd.NA with np.nan (numeric) or None (object)
    - Normalize mixed-object columns: try numeric coercion, otherwise stringify safely
    - Keep category dtype as-is (Arrow supports dictionary), but ensure codes are consistent
    - Convert timezone-aware datetimes to tz-naive UTC if needed, or coerce to datetime64[ns]
    Returns the converted DataFrame (copy).
    """
    df = df.copy(deep=False)  # shallow copy, we'll reassign columns
    changes = {}

    for col in df.columns:
        try:
            ser = df[col]
            dtype = ser.dtype
            orig_dtype = str(dtype)

            # 1) Extension dtypes (nullable Int64/Float64/boolean)
            if pd.api.types.is_extension_array_dtype(dtype):
                # nullable integer (Int64) -> float64 (if missing values) or int64 if safe and requested
                if pd.api.types.is_integer_dtype(dtype):
                    if convert_ints_to_float or ser.isna().any():
                        df[col] = pd.to_numeric(
                            ser, errors='coerce').astype('float64')
                        changes[col] = f"{orig_dtype} -> float64 (nullable ints to float)"
                    else:
                        # try int64 (will error if NA present)
                        df[col] = ser.astype('int64')
                        changes[col] = f"{orig_dtype} -> int64"
                # nullable float (Float64) -> numpy float64
                elif pd.api.types.is_float_dtype(dtype):
                    df[col] = pd.to_numeric(
                        ser, errors='coerce').astype('float64')
                    changes[col] = f"{orig_dtype} -> float64"
                # nullable boolean -> object / bool
                elif pd.api.types.is_bool_dtype(dtype):
                    if ser.isna().any():
                        df[col] = ser.astype(object).where(ser.notna(), None)
                        changes[col] = f"{orig_dtype} -> object with None for missing bools"
                    else:
                        df[col] = ser.astype(bool)
                        changes[col] = f"{orig_dtype} -> bool"
                else:
                    # fallback: convert to object and replace pd.NA -> None
                    df[col] = ser.astype(object).where(ser.notna(), None)
                    changes[col] = f"{orig_dtype} -> object fallback"

            # 2) Categorical: Arrow supports dictionary encodings, keep but ensure no pd.NA sentinel
            elif pd.api.types.is_categorical_dtype(dtype):
                # replace missing categories with None
                df[col] = ser.where(ser.notna(), None)
                changes[col] = f"{orig_dtype} -> category (missing->None)"

            # 3) Datetime: ensure datetime64[ns]; coerce invalid -> NaT
            elif pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
                df[col] = pd.to_datetime(ser, errors='coerce')
                changes[col] = f"{orig_dtype} -> datetime64[ns] (coerced)"

            # 4) Object dtype: mixed types or numpy scalars inside cells often break Arrow
            elif pd.api.types.is_object_dtype(dtype):
                # replace pd.NA with None first
                tmp = ser.where(ser.notna(), None)

                # attempt to detect purely-numeric objects and coerce
                try:
                    coerced = pd.to_numeric(tmp, errors='raise')
                    df[col] = coerced
                    changes[col] = f"{orig_dtype} object -> numeric (coerced)"
                except Exception:
                    # not purely numeric: check heterogeneity
                    sample_types = tmp.dropna().head(200).map(type).unique().tolist()
                    # if heterogeneous (ints, floats, numpy scalars), stringify safe
                    # but preserve strings as-is
                    if any(hasattr(v, "dtype") and not isinstance(v, str) for v in tmp.dropna().head(50)):
                        df[col] = tmp.apply(
                            lambda x: None if x is None else str(x))
                        changes[col] = f"{orig_dtype} heterogeneous numpy scalars -> stringified"
                    elif len(sample_types) > 1:
                        # fallback to string to guarantee Arrow conversion
                        df[col] = tmp.apply(
                            lambda x: None if x is None else str(x))
                        changes[col] = f"{orig_dtype} heterogeneous types -> stringified"
                    else:
                        # homogeneous objects (likely strings) — leave as-is but ensure None for missing
                        df[col] = tmp
                        # no change recorded

            # 5) Numeric numpy dtypes and others: ensure native numpy dtypes and replace pandas NA
            else:
                if ser.isna().any():
                    if pd.api.types.is_numeric_dtype(dtype):
                        df[col] = ser.where(ser.notna(), np.nan).astype(dtype)
                        changes[col] = f"{orig_dtype} -> numeric native with np.nan"
                    else:
                        df[col] = ser.where(ser.notna(), None)
                        changes[col] = f"{orig_dtype} -> object-like fill None for missing"
                # else leave as-is

        except Exception as e:
            # If conversion for this column fails, fallback: stringify safely to preserve data
            logger.exception(
                "Failed converting column %s (dtype=%s): %s", col, dtype, e)
            try:
                df[col] = df[col].apply(
                    lambda x: None if pd.isna(x) else str(x))
                changes[col] = f"{dtype} -> stringified on failure"
            except Exception as e2:
                logger.exception("Failed to stringify column %s: %s", col, e2)

    if changes:
        logger.info("make_arrow_friendly applied conversions:\n%s",
                    "\n".join(f"{k}: {v}" for k, v in changes.items()))
    else:
        logger.debug("make_arrow_friendly made no conversions")

    return df


def preprocess_data(df: pd.DataFrame,
                    dataset_name: str,
                    one_hot_encode: bool = False) -> pd.DataFrame:
    df = df.copy()

    protected_col = "Class"   # <-- special handling rule

    # convert boolean columns safely, exclude Class
    bool_cols = df.select_dtypes(
        include='bool').columns.difference([protected_col])
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # categorical columns (object/category)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    category_maps = {}
    dummies_list = []
    scalers = {}

    for col in cat_cols:

        unique_count = df[col].nunique(dropna=True)

        # ----- Always label-encode CLASS, regardless of settings -----
        if col == protected_col:
            le = LabelEncoder()
            ser = df[col].astype(str).fillna("<<MISSING>>")
            df[col] = le.fit_transform(ser)

            classes = le.classes_
            encoded = le.transform(classes)
            mapping = dict(zip(encoded, classes))
            category_maps[col] = mapping
            continue

        # Skip large cardinality categorical variables
        if unique_count > 100:
            warn(
                f"Skipping LabelEncoding for column '{col}' in dataset '{dataset_name}' "
                f"because it has {unique_count} unique values (> 100)."
            )

            # HINT: Added feature to drop such columns with huge unique counts which could be names
            df.drop(columns=[col], inplace=True)
            continue

        # ----- One-hot encoding only if flag enabled -----
        if one_hot_encode and unique_count <= 10:
            ser = df[col].astype(object).where(df[col].notna(), "<<MISSING>>")
            dummies = pd.get_dummies(ser, prefix=col, dtype=float)
            dummies_list.append(dummies)
            df.drop(columns=[col], inplace=True)
            continue

        # ----- Otherwise, label encode -----
        le = LabelEncoder()
        ser = df[col].astype(str).fillna("<<MISSING>>")
        df[col] = le.fit_transform(ser)

        classes = le.classes_
        encoded = le.transform(classes)
        mapping = dict(zip(encoded, classes))
        category_maps[col] = mapping

    # Add dummy columns if any
    if dummies_list:
        try:
            all_dummies = pd.concat(dummies_list, axis=1)
            all_dummies.index = df.index
            df = pd.concat([df, all_dummies], axis=1)
        except Exception:
            warn("Failed to concat dummy columns. Attempting per-column fallback.")
            for d in dummies_list:
                df = pd.concat([df, d], axis=1)

    # ----- FLOAT SCALING (EXCLUDE CLASS) -----
    float_cols = [
        c for c in df.select_dtypes(include=[np.floating]).columns
        if c != protected_col
    ]

    for col in float_cols:
        try:
            col_vals = df[col]
            col_min = col_vals.min(skipna=True)
            col_max = col_vals.max(skipna=True)

            if pd.isna(col_min) and pd.isna(col_max):
                scalers[col] = {"min": None, "max": None}
                continue

            if col_max == col_min:
                df[col] = col_vals.apply(
                    lambda x: np.nan if pd.isna(x) else 0.0)
                scalers[col] = {"min": float(col_min), "max": float(col_max)}
                continue

            df[col] = (col_vals - col_min) / (col_max - col_min)
            scalers[col] = {"min": float(col_min), "max": float(col_max)}

        except Exception:
            warn(
                f"Failed to scale float column '{col}' in dataset '{dataset_name}'.")
            traceback.print_exc()

    # ----- IMPUTATION (EXCLUDE CLASS) -----
    for col in df.columns:
        if col == protected_col:
            continue

        if pd.api.types.is_numeric_dtype(df[col].dtype):
            imputer = SimpleImputer(strategy='mean')
        else:
            imputer = SimpleImputer(strategy='most_frequent')

        try:
            df[col] = imputer.fit_transform(df[[col]]).ravel()
        except Exception:
            warn(f"Couldn't impute {col} column of {dataset_name}.")
            traceback.print_exc()

    print(
        f"Adding the following category mapping to session state for {dataset_name}: \n"
        f"{category_maps}\n"
    )

    if get_script_run_ctx():
        ss.category_maps[dataset_name] = category_maps

    return df


def load_specific_dataset(dset_name: str = "", dset_path: str = ""):
    src = './datasets/'
    ds_store = pd.read_excel('./Dataset Repository.xlsx',
                             sheet_name="Sheet1")

    classification_dsets = ds_store[ds_store['Type'] == 'Classification']
    classification_dsets.drop(columns=['Sl. No', 'Type'], inplace=True)

    if dset_path and dset_name:
        return {
            dset_name: preprocess_data(pd.read_csv(src + dset_path), dset_name)
        }

    elif dset_name:
        # case-insensitive match (already implemented)
        dset_path = src + classification_dsets[
            classification_dsets['Name'].str.lower() == dset_name.lower()
        ]['Location'].values[0]

        return {
            dset_name: preprocess_data(pd.read_csv(dset_path), dset_name)
        }

    elif dset_path:
        # BUG FIX: previously you compared Location with dset_name.lower() — that was wrong
        dset_name = classification_dsets[
            classification_dsets['Location'].str.lower() == dset_path.lower()
        ]['Name'].values[0]

        return {
            dset_name: preprocess_data(pd.read_csv(src + dset_path), dset_name)
        }

    else:
        raise ValueError("Either of dset_name or dset_path must be provided")
