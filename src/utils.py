import os
import sys
import tensorflow as tf

from src.logger import logging
from src.exception import CustomException




def create_model():
    logging.info('initiating to build neurons')
    Image_shape = (256, 256, 3)
    base_model = tf.keras.applications.EfficientNetB6(weights="imagenet", include_top=False, input_shape=Image_shape)
    base_model.trainable = False
    # optimizer = tf.keras.optimizers.Adam()
    mae = tf.keras.metrics.MeanAbsoluteError()
    # precision = tf.keras.metrics.Precision()

    

    model_inputs = tf.keras.Input(shape=(256, 256, 3))
    x = base_model(model_inputs, training=False)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)

    # let's add a fully-connected layer
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # start passing that fully connected block output to all the different model heads
    y1 = tf.keras.layers.Dense(32, activation='relu')(x)
    y1 = tf.keras.layers.Dropout(0.2)(y1)
    y1 = tf.keras.layers.Dense(16, activation='relu')(y1)
    y1 = tf.keras.layers.Dropout(0.2)(y1)
    
    # y2 = tf.keras.layers.Dense(32,activation='sigmoid')(x)
    # y2 = tf.keras.layers.Dropout(8)(y2)
    # y2 = tf.keras.layers.Dense(16,activation='sigmoid')(y2)
    # y2 = tf.keras.layers.Dropout(4)(y2)


    # y3 = tf.keras.layers.Dense(32,activation='relu')(x)
    # y3 = tf.keras.layers.Dropout(8)(y3)
    # y3 = tf.keras.layers.Dense(16,activation='relu')(y2)
    # y3 = tf.keras.layers.Dropout(4)(y3)
    #     # Predictions for each task
    y1 = tf.keras.layers.Dense(units=3, activation="linear", name='bmi')(y1)
    # y2 = tf.keras.layers.Dense(units=3,activation="sigmoid",name='sex')(y2)
    # y3 = tf.keras.layers.Dense(units=3,activation="linear",name='age')(y3)

    custom_model = tf.keras.Model(inputs=model_inputs,outputs=y1)
    
    # loss_functions = {
    # 'bmi': 'mean_squared_error',
    # 'age': 'mean_squared_error',
    # 'sex': 'hinge'
    # }       

    custom_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5),
                         loss='mse',
                         metrics=mae)
    print(custom_model.summary())
    logging.info('Model created')
    return custom_model
