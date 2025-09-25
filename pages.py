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


if 'page' not in ss:
    ss.page = 'show_data'


def show_data():  # page 1
    st.markdown("""
                <h4>Choose a Dataset:</h4>
                """, unsafe_allow_html=True)  # header
    names = load_all_datasets(True)  # get the names.
    choice = st.selectbox('datasets', names)  # take input.
    if st.button('Show all datasets'):
        for name, dset in load_all_datasets().items():
            st.subheader(name)
            display_dataset(dset)
    else:
        display_dataset(load_specific_dataset(choice)[choice])

    if st.button('Upload my own data'):
        pass


if __name__ == "__main__":
    if ss.page == "show_data":
        show_data()
