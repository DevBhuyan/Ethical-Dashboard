from sklearn.ensemble import RandomForestClassifier


def init_model():
    '''
    Initializes a Random Forest Classifier with default hyperparameters.
    Edit the hyperparameters below as needed.
    '''
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    return model
