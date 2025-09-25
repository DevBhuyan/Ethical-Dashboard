#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 00:47:54 2025

@author: dev
"""

import os
from importlib.util import (
    spec_from_file_location,
    module_from_spec
)


base_model_files = os.listdir('./base_models')


def eager_load_all_models():

    src = './base_models/'

    model_builders = {}
    for model_file in base_model_files:
        model_name = model_file[:-3]
        file_path = src + model_file

        spec = spec_from_file_location(model_name, file_path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)

        model_builders[model_name] = module.init_model

    return model_builders


def lazy_load_model(model_name: str):

    src = './base_models/'

    file_path = src + model_name + '.py'

    spec = spec_from_file_location(model_name, file_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    return {
        model_name: module.init_model
    }
