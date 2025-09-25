from sklearn.linear_model import LogisticRegression


def init_model():
    '''
    Initializes a Logistic Regression model with default hyperparameters.
    Edit the hyperparameters below as needed.
    '''
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=100
    )
    return model
