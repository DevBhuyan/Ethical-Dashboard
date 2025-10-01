#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 00:47:54 2025

@author: dev and Dhananjoy Bhuyan
"""

from safetensors import safe_open
import tensorflow as tf
import sklearn
import tempfile
import inspect
import joblib
import pickle
import os
from importlib.util import (
    spec_from_file_location,
    module_from_spec
)


def eager_load_all_models(names_only: bool = False):

    src = './base_models/'

    base_model_files = [i
                        for i in os.listdir(src)
                        if not i.startswith('_')]

    if names_only:
        return [model[:-3]
                for model in base_model_files]

    model_builders = {}
    for model_file in base_model_files:

        model_name = model_file[:-3]
        file_path = src + model_file

        spec = spec_from_file_location(model_name, file_path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)

        model_builders[model_name] = module.init_model()

    return model_builders


def lazy_load_model(model_name: str):

    src = './base_models/'

    file_path = src + model_name + '.py'

    spec = spec_from_file_location(model_name, file_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    return {
        model_name: module.init_model()
    }


def convert_upload_to_model(upload):
    """Infer file type from extension and load model into a trainable object."""
    ext = os.path.splitext(upload.name)[-1].lower()

    # Write upload to a temp file for frameworks that need file paths
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(upload.read())
        tmp_path = tmp.name

    try:
        if ext == ".pkl":
            try:
                model = joblib.load(tmp_path)
            except Exception:
                with open(tmp_path, "rb") as f:
                    model = pickle.load(f)

        elif ext in [".h5", ".hdf5"]:
            model = tf.keras.models.load_model(tmp_path)

        elif ext == ".safetensors":
            # safetensors usually contains raw tensors, not an assembled model
            # Users typically reconstruct via Hugging Face APIs
            tensors = {}
            with safe_open(tmp_path, framework="tf", device="cpu") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
            model = tensors  # Placeholder, caller must wrap into HF model

        else:
            raise ValueError(f"Unsupported model format: {ext}")

    finally:
        os.remove(tmp_path)

    return model


def validate_model(model):
    """
    Ensure model has a trainable method (fit or train) with signature (X_train, y_train, [X_val, y_val]).
    """
    candidate = None
    if hasattr(model, "fit"):
        candidate = model.fit
    elif hasattr(model, "train"):
        candidate = model.train

    if candidate is None:
        raise ValueError("Model must implement a `fit` or `train` method.")

    sig = inspect.signature(candidate)
    params = list(sig.parameters.keys())

    # Must have at least X_train, y_train
    if len(params) < 2:
        raise ValueError(
            f"Trainable method must accept at least two args (X_train, y_train), found {params}"
        )

    return True


def load_model_from_st_upload(upload):

    model = convert_upload_to_model(upload)
    validate_model(model)

    return model
