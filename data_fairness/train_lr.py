#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from .EDA import get_data
import os
import pickle


def train_accurate_lr(data):
    X_train = data["X_train"]
    X_test = data["X_test"] 
    y_train = data["y_train"]
    y_test = data["y_test"]
    
    # Find optimal params first
    
    path = f"./models/{data['name']}_lr_accurate.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            lr = pickle.load(f)
    else:
        if data["name"] == 'adult':
            lr = LogisticRegression(
                        max_iter=500,
                        random_state=42
                    )
            
        if data["name"] == 'german':
            lr = LogisticRegression(
                        random_state=42
                    )
            
        lr.fit(X_train, y_train)
        
        with open(path, "wb") as f:
            pickle.dump(lr, f)
    
    y_pred = lr.predict(X_test)    
    
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Logistic Regression accuracy: {acc*100:.2f}%")
    
    return lr, acc, y_pred


if __name__ == "__main__":
    data = get_data('german')
    model, acc, y_pred = train_accurate_lr(data)


