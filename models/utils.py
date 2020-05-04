"""
Tutorial material from:
https://github.com/philtabor/Youtube-Code-Repository/
"""


from tensorflow import keras
from tensorflow.keras.optimizers import Adam


def build_dqn(learning_rate, action_count, input_dims, fc1, fc2):

    model = keras.Sequential([
        keras.layers.Dense(fc1, activation='relu', input_shape=input_dims),
        keras.layers.Dense(fc2, activation='relu', input_shape=input_dims),
        keras.layers.Dense(action_count, activation=None)
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error'
    )

    return model
