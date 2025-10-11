#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:36:15 2024

@author: dev
"""

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 
  
# metadata 
print(adult.metadata) 
  
# variable information 
print(adult.variables) 
