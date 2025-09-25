from xgboost import XGBClassifier


def init_model():
    '''
    Initializes an XGBoost Classifier with default hyperparameters.
    Edit the hyperparameters below as needed.
    '''
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    return model
