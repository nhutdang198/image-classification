### import neccessary libraries
import tensorflow as tf
# from tensorflow import keras
# from tensorflow import io
# import matplotlib.pyplot as pyplot
# import numpy
# from PIL import Image
import os
# import json
import pathlib
from colorama import Fore
# from datetime import datetime
from helper.can_decode_image import can_decode_image
from helper.decode_image import decode_image

### transform feature(Xn) and output(Y) to tensor
def transform_to_tensor(image_path: str, label: str, image_size: int) -> any:
  image = decode_image(image_path, image_size)
  return image, label


# def preprocess_image_dataset(features, labels, image_size: int = 224, batch_size: int=32, training_split_size: float = 0.9) -> any:
#   # Shuffle the data and split it into training (80%) and validation (20%) sets
#   dataset_size = len(features)
#   indices = tf.range(start=0, limit=dataset_size, dtype=tf.int32)
#   shuffled_indices = tf.random.shuffle(indices)


#   # Calculate split sizes
#   train_size = int(training_split_size * dataset_size)
#   train_indices = shuffled_indices[:train_size]
#   val_indices = shuffled_indices[train_size:]

#   ### train dataset 
#   train_image_paths = tf.gather(features, train_indices)
#   train_labels = tf.gather(labels, train_indices)

#   ### validation dataset
#   val_image_paths = tf.gather(features, val_indices)
#   val_labels = tf.gather(labels, val_indices)

#   train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
#   train_dataset = train_dataset.map(lambda x, y:  transform_to_tensor(x, y, image_size=image_size), num_parallel_calls=tf.data.AUTOTUNE)
#   train_dataset = train_dataset.batch(batch_size)
#   train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
#   train_dataset_size = len(train_image_paths)

#   validation_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_labels))
#   validation_dataset = validation_dataset.map(lambda x, y: transform_to_tensor(x, y, image_size), num_parallel_calls=tf.data.AUTOTUNE)
#   validation_dataset = validation_dataset.batch(batch_size)
#   validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
#   validation_dataset_size = len(val_image_paths)
  
#   return train_dataset, validation_dataset, train_dataset_size, validation_dataset_size

def preprocess_image_dataset(features, labels, image_size: int = 224, batch_size: int=32) -> any:
  # Shuffle the data and split it into training (80%) and validation (20%) sets
  # dataset_size = len(features)
  # indices = tf.range(start=0, limit=dataset_size, dtype=tf.int32)
  # shuffled_indices = tf.random.shuffle(indices)


  # # Calculate split sizes
  # train_size = int(training_split_size * dataset_size)
  # train_indices = shuffled_indices[:train_size]
  # val_indices = shuffled_indices[train_size:]

  # ### train dataset 
  # train_image_paths = tf.gather(features, train_indices)
  # train_labels = tf.gather(labels, train_indices)

  # ### validation dataset
  # val_image_paths = tf.gather(features, val_indices)
  # val_labels = tf.gather(labels, val_indices)

  train_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
  train_dataset = train_dataset.map(lambda x, y:  transform_to_tensor(x, y, image_size=image_size), num_parallel_calls=tf.data.AUTOTUNE)
  train_dataset = train_dataset.batch(batch_size)
  train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
  train_dataset_size = len(features)

  # validation_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_labels))
  # validation_dataset = validation_dataset.map(lambda x, y: transform_to_tensor(x, y, image_size), num_parallel_calls=tf.data.AUTOTUNE)
  # validation_dataset = validation_dataset.batch(batch_size)
  # validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
  # validation_dataset_size = len(val_image_paths)
  
  return train_dataset, train_dataset_size
  