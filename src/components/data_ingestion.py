import os
import sys
import shutil
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass


## intitialize the Data Ingestion configuration

@dataclass
class DataIngestionconfig:
    train_data_path = os.path.join('D:/project/BMI/artifact', 'Train')
    test_data_path = os.path.join('D:/project/BMI/artifact', 'Test')
    valid_data_path = os.path.join('D:/project/BMI/artifact', 'Valid')


## create the data ingestion class

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:

            Train = os.path.join('D:/project/BMI/notebooks/data/Train')
            Test = os.path.join('D:/project/BMI/notebooks/data/Test')

            def get_filenames(directory_path):
                filenames = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)]
                return filenames

            train_filenames = get_filenames(Train)
            test_filenames = get_filenames(Test)

            # Divide the train_filenames list into train and validation sets
            valid_filenames = train_filenames[50:]
            train_filenames = train_filenames[:50]

            logging.info('Split Data into Train and Validation')

            os.makedirs(self.ingestion_config.train_data_path, exist_ok=True)
            for filename in train_filenames:
                shutil.copy(filename, self.ingestion_config.train_data_path)

            os.makedirs(self.ingestion_config.test_data_path, exist_ok=True)
            for filename in test_filenames:
                shutil.copy(filename, self.ingestion_config.test_data_path)

            os.makedirs(self.ingestion_config.valid_data_path, exist_ok=True)
            for filename in valid_filenames:
                shutil.copy(filename, self.ingestion_config.valid_data_path)

            logging.info('DATA INGESTION SUCCESSFUL')

            self.ingestion_config.train_data_path = self.ingestion_config.train_data_path.replace('\\', '/')
            self.ingestion_config.test_data_path = self.ingestion_config.test_data_path.replace('\\', '/')
            self.ingestion_config.valid_data_path = self.ingestion_config.valid_data_path.replace('\\', '/')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.valid_data_path

            )

        except Exception as e:
            logging.info('Exception occurred at Data Ingestion Stage')
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_path, test_path, valid_path = obj.initiate_data_ingestion()
    print(train_path, test_path, valid_path)
