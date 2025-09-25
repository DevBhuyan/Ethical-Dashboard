from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


def init_model(input_shape=(None, 10), num_classes=2):
    '''
    Initializes a simple MLP model with default hyperparameters.
    Edit layers, activations, dropout, etc. as needed.
    '''
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape[1:]),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
