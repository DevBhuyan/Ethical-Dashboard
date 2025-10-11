#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 23:59:23 2025

@author: dev
"""


from data_fairness.main import (
    plot_wo_save,
    METRICS
)
from session_state_attrib import ss
import streamlit as st


def fairness_eval():

    y_pred = ss.trained_model.predict(ss.X_test)

    data = {
        "name": ss.selected_dataset_name,
        "X_train": ss.X_train,
        "X_test": ss.X_test,
        "y_train": ss.y_train,
        "y_test": ss.y_test,
        "category_maps": ss.category_maps
    }

    for sensitive_feature in ss.sensitive_attributes[ss.selected_dataset_name]:
        fig, results = plot_wo_save(
            data,
            sensitive_feature,
            METRICS,
            y_pred,
            silent=False
        )
        st.pyplot(fig)
        st.write(results)


def robustness_eval():
    pass


def privacy_eval():
    pass


def explainability_eval():
    pass
