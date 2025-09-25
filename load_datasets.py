#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 00:30:13 2025

@author: dev
"""


import pandas as pd


def load_all_datasets(names_only: bool = False) -> dict | list:

    src = './datasets/'
    ds_store = pd.read_excel('./Dataset Repository.xlsx',
                             sheet_name="Sheet1")

    classification_dsets = ds_store[ds_store['Type'] == 'Classification']
    classification_dsets.drop(columns=['Sl. No', 'Type'], inplace=True)

    if names_only:
        dsets = [ds_name
                 for ds_name in classification_dsets['Name'].values]

    else:
        dsets = {
            ds_name: pd.read_csv(src + file)
            for ds_name, file in zip(
                classification_dsets['Name'].values,
                classification_dsets['Location'].values
            )
        }

    return dsets


def load_specific_dataset(dset_name: str = "",
                          dset_path: str = ""):

    src = './datasets/'
    ds_store = pd.read_excel('./Dataset Repository.xlsx',
                             sheet_name="Sheet1")

    classification_dsets = ds_store[ds_store['Type'] == 'Classification']
    classification_dsets.drop(columns=['Sl. No', 'Type'], inplace=True)

    if dset_path and dset_name:

        return {
            dset_name: pd.read_csv(src + dset_path)
        }

    elif dset_name:
        dset_path = src + classification_dsets[
            classification_dsets['Name'] == dset_name
        ]['Location'].values[0]

        return {
            dset_name: pd.read_csv(dset_path)
        }

    elif dset_path:
        dset_name = src + classification_dsets[
            classification_dsets['Location'] == dset_name
        ]['Name'].values[0]

        return {
            dset_name: pd.read_csv(dset_path)
        }
