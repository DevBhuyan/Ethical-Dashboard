#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from .EDA import get_data
import os
import pickle
from warnings import simplefilter
simplefilter("ignore")


def train_accurate_rf(data):
    X_train = data["X_train"]
    X_test = data["X_test"] 
    y_train = data["y_train"]
    y_test = data["y_test"]
    
    path = f"./models/{data['name']}_rf_accurate.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            rf = pickle.load(f)
    else:
        if data["name"] == 'adult':
            rf = RandomForestClassifier(
                        n_estimators=600,
                        max_depth=15,
                        random_state=42
                    )
            
        if data["name"] == 'german':
            rf = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    )
            
        rf.fit(X_train, y_train)
        
        with open(path, "wb") as f:
            pickle.dump(rf, f)
    
    y_pred = rf.predict(X_test)    
    
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Random Forest accuracy: {acc*100:.2f}%")
    
    return rf, acc, y_pred


if __name__ == "__main__":
    data = get_data('german')
    model, acc, y_pred = train_accurate_rf(data)
