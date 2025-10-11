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


def save_configuration_as_previous():
    save_dir = './previous_config/'
    os.makedirs(save_dir, exist_ok=True)

    dct = {
        "dataset": ss.selected_dataset_name,
        "model": ss.selected_model_name,
        "metrics": ss.eval_metrics if hasattr(ss, 'eval_metrics') else None,
        "category_maps": ss.category_maps if hasattr(ss, 'category_maps') else None
    }

    # Save metadata (selection + metrics + category maps)
    with open(os.path.join(save_dir, 'selection.json'), 'w') as f:
        json.dump(make_json_serializable(dct), f, indent=4)

    # Save dataset
    ss.selected_dataset.to_csv(os.path.join(
        save_dir, ss.selected_dataset_name), index=False)

    # Save model
    with open(os.path.join(save_dir, ss.selected_model_name + '.pkl'), 'wb') as f:
        pickle.dump(ss.trained_model, f)


def load_previous_configuration():
    load_dir = './previous_config/'

    try:
        # Load metadata
        with open(os.path.join(load_dir, 'selection.json'), 'r') as f:
            dct = json.load(f)

        ss.selected_dataset_name = dct["dataset"]
        ss.selected_model_name = dct["model"]

        # Load optional fields
        ss.eval_metrics = dct.get("metrics", None)
        ss.category_maps = dct.get("category_maps", None)

        # Load dataset
        ss.selected_dataset = pd.read_csv(
            os.path.join(load_dir, ss.selected_dataset_name))

        # Load model
        with open(os.path.join(load_dir, ss.selected_model_name + '.pkl'), 'rb') as f:
            ss.trained_model = pickle.load(f)
            ss.selected_model = ss.trained_model

        return True

    except:
        return False
