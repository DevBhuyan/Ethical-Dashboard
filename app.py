#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 22:59:35 2025

@author: dev
"""


import streamlit as st


st.set_page_config(
    page_title="Ethical AI Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://example.com/help',
        'Report a bug': 'mailto:devvjiit.bhuyan@gmail.com',
        'About': "This dashboard tracks and visualizes AI ethics metrics, fairness, and model accountability."
    }
)


st.markdown(
    """
    <style>
    /* Remove padding/margins around the main container */
    .css-18e3th9 {padding: 0rem 1rem 0rem 1rem;}
    /* Hide hamburger menu and footer (extra safety) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    /* Optional: make all buttons and inputs look standard */
    button, input, select, textarea {
        border-radius: 4px !important;
        border: 1px solid #ccc !important;
        font-family: sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# st.title("Ethical AI Dashboard")
