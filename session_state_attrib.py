#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 00:43:36 2025

@author: dev
"""


from streamlit import session_state as ss
import json


def init():

    with open('./ss_attrib.json') as f:
        ss_attrib = json.load(f)

    for attrib, value in ss_attrib.items():
        if attrib not in ss:
            ss[attrib] = value
