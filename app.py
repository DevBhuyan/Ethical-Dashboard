#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 22:59:35 2025

@author: dev and Dhananjoy Bhuyan
"""


import streamlit as st
from pages import (
    data_home,
    debug_info,
    view_dataset,
    edit_data,
    model_home,
    view_model,
    training_results
)
from session_state_attrib import (
    init,
    ss
)


st.set_page_config(
    page_title="Ethical AI - Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:devvjiit.bhuyan@gmail.com',
        'Report a bug': 'mailto:devvjiit.bhuyan@gmail.com',
        'About': "This dashboard tracks and visualizes AI ethics metrics, fairness, and model accountability."
    }
)


init()


def main():
    st.markdown(
        """
    <style>
    /* Remove padding/margins around the main container */
    .css-18e3th9 {padding: 0rem 1rem 0rem 1rem;}
    /* Hide hamburger menu and footer (extra safety) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
        unsafe_allow_html=True
    )

    st.title("Ethical AI Dashboard")

    if ss.page == "data_home":
        data_home()

    elif ss.page == "view_dataset":
        view_dataset()

    elif ss.page == "edit_data":
        edit_data()

    elif ss.page == "model_home":
        model_home()

    elif ss.page == "view_model":
        view_model()

    elif ss.page == "training_results":
        training_results()

    else:
        st.subheader("You have reached a dead-end")
        st.error(f"There is no page named {ss.page}")
        st.snow()

    debug_info()


if __name__ == "__main__":
    main()
