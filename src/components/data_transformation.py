import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

from src.logger import logging
from src.exception import CustomException




# ## Data Transformation class
class DataTransformation:

    def input_image_data(self, image_path, image_size=(256, 256)):

        try:
            image = load_img(image_path, target_size=image_size)
            image = image.astype('float') / 255.0
            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            return image_array

        except Exception as e:
            raise CustomException(f"Error processing image: {str(e)}")


class DataTransformation:
    try:

        logging.info('initializing Data as Dataset')

        def _parse_image_function(example_proto):
            image_feature_description = {
                'image': tf.io.FixedLenFeature([], tf.string),
                'bmi': tf.io.FixedLenFeature([], tf.float32),
                # 'sex': tf.io.FixedLenFeature([], tf.int64),
                # 'age': tf.io.FixedLenFeature([], tf.float32)
            }

            features = tf.io.parse_single_example(example_proto, image_feature_description)
            image = tf.io.decode_raw(features['image'], tf.uint8)
            image.set_shape([3 * 256 * 256])
            image = tf.reshape(image, [256, 256, 3])

            bmi = tf.cast(features['bmi'], tf.float32)
            # sex = tf.cast(features['sex'], tf.int64)
            # age = tf.cast(features['age'], tf.int32)
            return image, bmi

        def _unparse_image_function(example_proto):

            image_feature_description = {
                'image': tf.io.FixedLenFeature([], tf.string)
            }

            features = tf.io.parse_single_example(example_proto, image_feature_description)
            image = tf.io.decode_raw(features['image'], tf.uint8)
            image.set_shape([3 * 256 * 256])
            image = tf.reshape(image, [256, 256, 3])
            return image

        # @staticmethod
        def read_tfrecord_folder(folder_path, batch_size=32, shuffle_buffer_size=1024, repeat=True, labeled=True):
            AUTO = tf.data.AUTOTUNE
            REPLICAS = 50
            tfrecord_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                              file.endswith('.tfrecord')]
            dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=AUTO)
            dataset = dataset.cache()

            if shuffle_buffer_size > 0:
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_size * 24)
                opt = tf.data.Options()
                opt.experimental_deterministic = True
                dataset = dataset.with_options(opt)

            if repeat:
                dataset = dataset.repeat()

            if labeled:
                dataset = dataset.map(DataTransformation._parse_image_function, num_parallel_calls=AUTO)

            else:
                dataset = dataset.map(DataTransformation._unparse_image_function, num_parallel_calls=AUTO)

            dataset = dataset.batch(batch_size * REPLICAS)
            dataset = dataset.prefetch(buffer_size=AUTO)
            logging.info('Dataset generated')

            return dataset

        def input_gen(image_batch, bmi_batch):
            logging.info('Extracting Inputs and Output as x and y')
            x = []
            y = []
            for n in range(len(image_batch)):
                input_arr = tf.keras.preprocessing.image.img_to_array(image_batch[n])
                output_arr = [bmi_batch[n]]
                x.append(input_arr)
                y.append(output_arr)

            logging.info('Inputs and Outputs generated')
            return x, y
    except Exception as e:
        logging.info('Exception occurred at Transformation Stage')
        raise CustomException(e, sys)


if __name__ == '__main__':
    train_folder = 'D:/project/BMI/notebooks/data/Train'  # Use forward slashes or raw string
    print(train_folder)
    batch_size = 32  # Set your desired batch size

    train_dataset = DataTransformation.read_tfrecord_folder(train_folder)

  
    # Example usage of input_gen
    image_batch, bmi_batch = next(iter(train_dataset))
    x, y = DataTransformation.input_gen(image_batch.numpy(), bmi_batch.numpy())
    print(x)
    print(y)
    print(train_folder)
