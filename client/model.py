# import tensorflow as tf

# def create_model():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(8,)),
        
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.3),

#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.3),

#         tf.keras.layers.Dense(16, activation='relu'),

#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )

#     return model

# import tensorflow as tf

# def create_model():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(8,)),

#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.4),

#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.3),

#         tf.keras.layers.Dense(32, activation='relu'),

#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )

#     return model

import tensorflow as tf


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(32, activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model