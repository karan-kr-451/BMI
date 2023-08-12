import os
import sys
import tensorflow as tf
import numpy as np
from src.utils import create_model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('D:/project/BMI/artifact', 'Model.h5')


class ModelTrainer:
    def __init__(self):
        self.TrainerConfig = ModelTrainerConfig()

    def initiate_training(self, x_train, y_train, x_val, y_val, batch_size=32, num_epochs=35, steps_per_epoch=20):
        logging.info('initiate Training')
        try:
            model_dir = './'
            es = EarlyStopping(patience=5)
            ckp = ModelCheckpoint(model_dir, save_best_only=True, save_weights_only=True, verbose=1)
            callbacks = [es, ckp]

            y_train = tf.stack(y_train)
            y_val = tf.stack(y_val)

            # Convert data to NumPy arrays
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_val = np.array(x_val)
            y_val = np.array(y_val)

            # Create the TensorFlow model
            model = create_model()

            # Training loop
            model.fit(x=x_train, y=y_train,
                      epochs=num_epochs, callbacks=callbacks, batch_size=batch_size,
                      steps_per_epoch=steps_per_epoch, validation_data=(x_val, y_val))

            # Optionally, evaluate the model on the validation dataset
            val_loss, val_mae = model.evaluate(x_val, y_val, batch_size=batch_size)

            print("Validation Loss:", val_loss)
            print("Validation MAE:", val_mae)

            # After the training loop, you can save the trained model if needed
            os.makedirs(os.path.dirname(self.TrainerConfig.model_path), exist_ok=True)
            model.save(self.TrainerConfig.model_path)
            self.TrainerConfig.model_path = self.TrainerConfig.model_path.replace('\\', '/')
            logging.info('Model Created')

            return self.TrainerConfig.model_path

        except Exception as e:
            logging.info('Exception occurred at Training Stage')
            raise CustomException(e, sys)
