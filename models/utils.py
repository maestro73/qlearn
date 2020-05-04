# -*- coding: utf-8 -*-
"""
Tutorial material from:
https://github.com/philtabor/Youtube-Code-Repository/
"""
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


def build_dqn(learning_rate, action_count, batch_size):

    model = keras.Sequential()

    model.add(
        keras.layers.Dense(
            batch_size,
            activation='relu',
            batch_size=batch_size
        )
    )

    model.add(
        keras.layers.Dense(
            batch_size,
            activation='relu',
            batch_size=batch_size
        )
    )

    model.add(keras.layers.Dense(action_count, activation=None))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error'
    )

    return model
