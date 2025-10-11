#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 01:34:40 2025

@author: dev
"""


import math
import inspect
from session_state_attrib import (
    ss
)
import streamlit as st


def model_card():

    model = ss.selected_model

    st.subheader("Model Summary")
    st.write(model)

    if isinstance(model, dict):
        st.write(f"Number of tensors: {len(model)}")
        for name, tensor in list(model.items())[:5]:
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

        # Filter out None and NaN values
        filtered_params = {
            k: v for k, v in params.items()
            if not (isinstance(v, float) and math.isnan(v)) and v
        }

        st.write("**Parameters:**")
        st.json(filtered_params)


def model_details():

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
