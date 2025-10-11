#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from .EDA import get_data
import os
import pickle


def train_accurate_nb(data):
    X_train = data["X_train"]
    X_test = data["X_test"] 
    y_train = data["y_train"]
    y_test = data["y_test"]
    
    # Find optimal params first [N/A]
    
    path = f"./models/{data['name']}_nb_accurate.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            nb = pickle.load(f)
    else:
        nb = GaussianNB(
                    )
            
        nb.fit(X_train, y_train)
        
        with open(path, "wb") as f:
            pickle.dump(nb, f)
    
    y_pred = nb.predict(X_test)    
    
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Naive Bayes accuracy: {acc*100:.2f}%")
    
    return nb, acc, y_pred


if __name__ == "__main__":
    data = get_data('german')
    model, acc, y_pred = train_accurate_nb(data)


