#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 01:13:51 2025

@author: Dhananjoy Bhuyan
"""


import streamlit as st  # for UI/UX
from streamlit import session_state as ss  # persistance of values
from load_datasets import load_all_datasets, load_specific_dataset  # loaaaaad!!!
from load_models import eager_load_all_models, lazy_load_model  # looooaaaad again!!
from dataset_viewer import display_dataset
import pandas as pd


if 'showing_all' not in ss:
    ss.showing_all = False
if 'current_data' not in ss:
    ss.current_data = False
if 'custom_csv' not in ss:
    ss.custom_csv = False


def show_data():  # page 1

    st.markdown("""
                
                <h3>Choose a Dataset:</h3>
                
                """, unsafe_allow_html=True)  # upper stuff......hehe
    names = load_all_datasets(True)  # get the names.
    choice = st.selectbox('datasets', names)  # take input.
    if st.button('Show all datasets'):
        ss.showing_all = True

        st.rerun()
    elif ss.custom_csv:
        ss.showing_all = False
    else:
        ss.showing_all = False
        ss.current_data = load_specific_dataset(choice)[choice]

    file = st.file_uploader(
        'Want to upload your own data? Upload here:', type="csv")

    if file and not ss.custom_csv:
        ss.current_data = pd.read_csv(file)
        ss.custom_csv = True
        ss.showing_all = False
        st.rerun()
    else:
        ss.custom_csv = False

    if ss.showing_all:
        for name, dset in load_all_datasets().items():
            st.subheader(name)
            display_dataset(dset)

    else:
        display_dataset(ss.current_data)

    if st.button('Continue'):
        pass
