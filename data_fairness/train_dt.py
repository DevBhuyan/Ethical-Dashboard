#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from .EDA import get_data
import os
import pickle


def train_accurate_dt(data):
    X_train = data["X_train"]
    X_test = data["X_test"] 
    y_train = data["y_train"]
    y_test = data["y_test"]
    
    # Find optimal params first
    
    path = f"./models/{data['name']}_dt_accurate.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            dt = pickle.load(f)
    else:
        if data["name"] == 'adult':
            dt = DecisionTreeClassifier(
                        criterion="entropy",
                        max_features=6,
                        min_samples_split=9,
                        max_depth=10,
                        random_state=42
                    )
            
        if data["name"] == 'german':
            dt = DecisionTreeClassifier(
                        max_features=3,
                        max_depth=20,
                        random_state=42
                    )
            
        dt.fit(X_train, y_train)
        
        with open(path, "wb") as f:
            pickle.dump(dt, f)
    
    y_pred = dt.predict(X_test)    
    
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Decision Tree accuracy: {acc*100:.2f}%")
    
    return dt, acc, y_pred


if __name__ == "__main__":
    data = get_data('german')
    model, acc, y_pred = train_accurate_dt(data)
