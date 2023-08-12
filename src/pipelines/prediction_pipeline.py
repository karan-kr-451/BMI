import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from src.exception import CustomException


class PredictPipeline:
    def __init__(self):
        self.cache = {}

    def preprocess_image(self, img, target_size=(256, 256)):
        img = cv2.resize(img, target_size)
        image_array = img_to_array(img)
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def predict(self, img):
        try:
            
            if tuple(img.shape) in self.cache:
                return self.cache[tuple(img.shape)]

            model_path = os.path.join('artifact', 'Model.h5')

            # Preprocess the image
            img = self.preprocess_image(img)

            # Load the model
            model = load_model(model_path)

            # Make the prediction
            pred = model.predict(img)

            # Cache the result
            self.cache[tuple(img.shape)] = pred

            return pred

        except Exception as e:
            raise CustomException(e, sys)
