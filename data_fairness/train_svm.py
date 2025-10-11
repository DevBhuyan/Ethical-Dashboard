#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from .EDA import get_data
import os
import pickle
from warnings import simplefilter
simplefilter("ignore")


def train_accurate_svm(data):
    X_train = data["X_train"]
    X_test = data["X_test"] 
    y_train = data["y_train"]
    y_test = data["y_test"]
    
    # Find optimal params first
    
    path = f"./models/{data['name']}_svm_accurate.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            svm = pickle.load(f)
    else:
        if data["name"] == 'adult':
            svm = SVC(
                        kernel='poly',
                        degree=1,
                        max_iter=800,
                        probability=True,
                        random_state=42
                    )
            
        if data["name"] == 'german':
            svm = SVC(
                        kernel='poly',
                        degree=1,
                        max_iter=200,
                        probability=True,
                        random_state=42
                    )
            
        svm.fit(X_train, y_train)
        
        with open(path, "wb") as f:
            pickle.dump(svm, f)
    
    y_pred = svm.predict(X_test)    
    
    acc = accuracy_score(y_test, y_pred)
    
    print(f"SVM accuracy: {acc*100:.2f}%")
    
    return svm, acc, y_pred


if __name__ == "__main__":
    data = get_data('german')
    model, acc, y_pred = train_accurate_svm(data)

