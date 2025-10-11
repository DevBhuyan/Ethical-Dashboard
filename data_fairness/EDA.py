#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def show_eda(df):
    """
    Perform exploratory data analysis (EDA) on the given DataFrame.

    Parameters:
    - df : pandas DataFrame
        The dataset to analyze.

    Output:
    - Prints the shape of the DataFrame.
    - Prints summary statistics (count, mean, std, min, 25%, 50%, 75%, max) for each numerical column using df.describe().
    - Prints the count of unique values for each categorical or discrete column using value_counts().
    """
    df = df.dropna()
    print(df.shape)
    print()
    print(df.describe())
    print()
    for col in df.columns:
        print(df[col].value_counts())


def get_data(dataset, path=None):
    """
    Load and preprocess the dataset for a given dataset name.

    Parameters:
    - dataset : str
        The name of the dataset to load and preprocess. Allowed values: 'adult' or 'german'.
    - path : str, optional
        The file path to load the dataset from. If None, default paths will be used.

    Returns:
    - data : dict
        A dictionary containing the preprocessed dataset and metadata, including:
        - "name" : str
            The name of the dataset.
        - "X_train" : pandas DataFrame
            Features for training.
        - "X_test" : pandas DataFrame
            Features for testing.
        - "y_train" : pandas Series
            Labels for training.
        - "y_test" : pandas Series
            Labels for testing.
        - "category_maps" : dict
            A mapping of original category labels to encoded values.
    - sensitive_features : list
        A list of sensitive features present in the dataset, used for fairness analysis.
    """
    if dataset == 'adult':
        column_names = [
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "Class"
        ]
        if path is None:
            path = "./datasets/adult.data"
        df = pd.read_csv(path, names=column_names, skipinitialspace=True)
        df = df.dropna()

    elif dataset == 'german':
        if path is None:
            path = './datasets/german_credit_data.csv'
        df = pd.read_csv(path, index_col=False)
        df = df.dropna()

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    le = LabelEncoder()
    category_maps = {}

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

        classes = le.classes_
        encoded = le.transform(classes)
        mapping = dict(zip(encoded, classes))

        category_maps[col] = mapping

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    if dataset == 'adult':
        sensitive_features = ["sex",
                              "race",
                              # "workclass",
                              "marital-status",
                              # "native-country"
                              ]

    if dataset == 'german':
        sensitive_features = ["Sex",
                              "Job",
                              "Housing"]

    return {
        "name": dataset,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "category_maps": category_maps
    }, sensitive_features


if __name__ == "__main__":

    df_list = {
        "adult": pd.read_csv('./datasets/uci_adult_dataset.csv'),
        "german_cred": pd.read_csv('./datasets/german_credit_data.csv')
    }

    for name, df in df_list.items():
        print(f"\n{name}\n")
        show_eda(df)
