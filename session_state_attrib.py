#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 00:43:36 2025

@author: dev
"""


import numpy as np
import pandas as pd
import os
from streamlit import session_state as ss
import json
import pickle


def init():

    with open('./ss_attrib.json') as f:
        ss_attrib = json.load(f)

    for attrib, value in ss_attrib.items():
        if attrib not in ss:
            ss[attrib] = value


def make_json_serializable(obj):
    """Recursively convert numpy and non-JSON-safe types."""
    if isinstance(obj, dict):
        return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(i) for i in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_dataframe_or_array(obj, path):
    """Save DataFrame, Series, or NumPy array automatically, preserving type."""
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path + '.csv', index=False)
        with open(path + '.type', 'w') as f:
            f.write('dataframe')
    elif isinstance(obj, pd.Series):
        obj.to_csv(path + '.csv', index=False, header=True)
        with open(path + '.type', 'w') as f:
            f.write('series')
    elif isinstance(obj, np.ndarray):
        np.save(path + '.npy', obj)
        with open(path + '.type', 'w') as f:
            f.write('ndarray')
    else:
        raise TypeError(f"Unsupported type for saving: {type(obj)}")


def load_dataframe_or_array(path_base):
    """Load previously saved DataFrame, Series, or NumPy array preserving original type."""
    type_path = path_base + '.type'
    if not os.path.exists(type_path):
        return None

    with open(type_path, 'r') as f:
        obj_type = f.read().strip()

    if obj_type == 'dataframe' and os.path.exists(path_base + '.csv'):
        return pd.read_csv(path_base + '.csv')
    elif obj_type == 'series' and os.path.exists(path_base + '.csv'):
        df = pd.read_csv(path_base + '.csv')
        # Restore as Series (first column)
        if df.shape[1] == 1:
            return df.iloc[:, 0]
        else:
            return pd.Series(df.to_dict(orient='records'))
    elif obj_type == 'ndarray' and os.path.exists(path_base + '.npy'):
        return np.load(path_base + '.npy', allow_pickle=True)
    else:
        return None


def save_configuration_as_previous():
    save_dir = './previous_config/'
    os.makedirs(save_dir, exist_ok=True)

    dct = {
        "dataset": ss.selected_dataset_name,
        "model": ss.selected_model_name,
        "metrics": make_json_serializable(ss.eval_metrics) if hasattr(ss, 'eval_metrics') else None,
        "category_maps": make_json_serializable(ss.category_maps) if hasattr(ss, 'category_maps') else None,
    }

    # Save metadata
    with open(os.path.join(save_dir, 'selection.json'), 'w') as f:
        json.dump(dct, f, indent=4)

    # Save main dataset
    ss.selected_dataset.to_csv(os.path.join(
        save_dir, ss.selected_dataset_name), index=False)

    # Save model
    with open(os.path.join(save_dir, ss.selected_model_name + '.pkl'), 'wb') as f:
        pickle.dump(ss.trained_model, f)

    # Save splits
    for split_name in ['X_train', 'X_test', 'y_train', 'y_test']:
        if hasattr(ss, split_name):
            save_dataframe_or_array(
                getattr(ss, split_name), os.path.join(save_dir, split_name))


def load_previous_configuration():
    load_dir = './previous_config/'
    if not os.path.exists(load_dir):
        return False

    with open(os.path.join(load_dir, 'selection.json')) as f:
        dct = json.load(f)

    ss.selected_dataset_name = dct["dataset"]
    ss.selected_model_name = dct["model"]
    ss.eval_metrics = dct.get("metrics", None)
    ss.category_maps = dct.get("category_maps", None)

    # Load main dataset
    ss.selected_dataset = pd.read_csv(
        os.path.join(load_dir, ss.selected_dataset_name))

    # Load model
    with open(os.path.join(load_dir, ss.selected_model_name + '.pkl'), 'rb') as f:
        ss.trained_model = pickle.load(f)
        ss.selected_model = ss.trained_model

    # Load splits
    for split_name in ['X_train', 'X_test', 'y_train', 'y_test']:
        data = load_dataframe_or_array(os.path.join(load_dir, split_name))
        if data is not None:
            ss[split_name] = data

    return True
