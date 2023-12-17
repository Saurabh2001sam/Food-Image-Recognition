# Importing major Libraries 
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

import os
import glob
import pandas as pd

# Specify the root directory where your food image dataset is located
root_dir = './train'
# root_dir = './valid'

def give_dataset(path):
  
  # Initialize lists to store file paths and corresponding food category labels
  file_paths = []
  labels = []
  # Traverse the directory structure to collect file paths and labels
  for folder_name in os.listdir(path):
      if os.path.isdir(os.path.join(path, folder_name)):
          # Inside each subfolder (representing a food category)
          folder_path = os.path.join(path, folder_name)
          image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
          file_paths.extend(image_files)
          labels.extend([folder_name] * len(image_files))

  # Create a DataFrame to store the dataset
  data = {'file_path': file_paths, 'food_category': labels}
  df = pd.DataFrame(data)

  # Shuffle the dataset
  df = df.sample(frac=1).reset_index(drop=True)

  # Save the dataset to a CSV file
  df.to_csv('food_dataset.csv', index=False)

  # Load the dataset
  dataset = pd.read_csv('food_dataset.csv')
  dataset = dataset.sample(frac=1)
  return dataset

  # Split data into training and test sets
  train_data, test_data = train_test_split(
      dataset, test_size=0.2, random_state=42)

train_data = give_dataset(root_dir)
test_data = give_dataset('./valid')


# Data preprocessing and augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# Define the base model
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Add custom classification layers
num_classes = 10

# Add a global average pooling layer
x = GlobalAveragePooling2D()(base_model.output)

# Add a fully connected layer with 1024 units and ReLU activation
x = Dense(1024, activation='relu')(x)

# Add a dropout layer with 0.2 dropout rate
x = tf.keras.layers.Dropout(0.2)(x)

# Add the output layer
predictions = Dense(num_classes, activation='softmax')(x)


# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
train_generator = datagen.flow_from_dataframe(
    dataframe=train_data,
    x_col='file_path',
    y_col='food_category',
    target_size=(224, 224),
    batch_size=10,
    class_mode='categorical'
)

test_generator = datagen.flow_from_dataframe(
    dataframe=test_data,
    x_col='file_path',
    y_col='food_category',
    target_size=(224, 224),
    batch_size=10,
    class_mode='categorical'
)

# model.fit(train_generator, validation_data=test_generator, epochs=10)


def stop_training_on_bad_results(model, train_generator, validation_data, epochs=10, model_save_path='best_model.keras'):
  """Trains a model and saves the best performing model after the early stopping callback is triggered.

  Args:
    model: A compiled Keras model.
    train_generator: A Keras DataGenerator object for training data.
    validation_data: A Keras DataGenerator object for validation data.
    epochs: The number of epochs to train for.
    model_save_path: The path to save the best performing model.

  Returns:
    A history object containing training and validation metrics.
  """

  # Create an early stopping callback to stop training when validation loss
  # stops improving.
  early_stopping = EarlyStopping(monitor='val_loss', patience=3)

  # Create a checkpoint callback to save the best performing model.
  model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True)

  # Train the model.
  history = model.fit(train_generator,
                      validation_data=validation_data,
                      epochs=epochs,
                      callbacks=[early_stopping, model_checkpoint])

  return history


history = stop_training_on_bad_results(model, train_generator, validation_data= test_generator, epochs=20, model_save_path='best_model.keras')
