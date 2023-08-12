import os
import sys
import tensorflow as tf
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    transformer = DataTransformation()
    obj = DataIngestion()
    train_model = ModelTrainer()

    train_path, test_path, valid_path = obj.initiate_data_ingestion()

    train_dataset = DataTransformation.read_tfrecord_folder(train_path)
    valid_dataset = DataTransformation.read_tfrecord_folder(valid_path)

    image_batch, bmi_batch = next(iter(train_dataset))
    x_train, y_train = DataTransformation.input_gen(image_batch.numpy(), bmi_batch.numpy())

    image_batch, bmi_batch = next(iter(valid_dataset))
    x_val, y_val = DataTransformation.input_gen(image_batch.numpy(), bmi_batch.numpy())

    model = train_model.initiate_training(x_train, y_train, x_val, y_val)
