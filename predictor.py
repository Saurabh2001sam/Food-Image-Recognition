# def gr(str):
#     return "hello my name suman"

# Importing major Libraries 
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

# # Specify the root directory where your food image dataset is located
# root_dir = './train_images'  

# # Initialize lists to store file paths and corresponding food category labels
# file_paths = []
# labels = []

# # Traverse the directory structure to collect file paths and labels
# for folder_name in os.listdir(root_dir):
#     if os.path.isdir(os.path.join(root_dir, folder_name)):
#         # Inside each subfolder (representing a food category)
#         folder_path = os.path.join(root_dir, folder_name)
#         image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
#         file_paths.extend(image_files)
#         labels.extend([folder_name] * len(image_files))

# # Create a DataFrame to store the dataset
# data = {'file_path': file_paths, 'food_category': labels}
# df = pd.DataFrame(data)

# # Shuffle the dataset
# df = df.sample(frac=1).reset_index(drop=True)

# # Save the dataset to a CSV file
# df.to_csv('food_dataset.csv', index=False)

# # Load the dataset
# dataset = pd.read_csv('food_dataset.csv')

# # Split data into training and test sets
# train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# # Data preprocessing and augmentation
# datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# # Create a base model (MobileNetV2) with pre-trained weights
# base_model = MobileNetV2(weights='imagenet', include_top=False)

# # Add custom classification layers
# num_classes = 12
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
# predictions = Dense(num_classes, activation='softmax')(x)  # 'num_classes' is the number of food categories

# # Create the final model
# model = Model(inputs=base_model.input, outputs=predictions)

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# train_generator = datagen.flow_from_dataframe(
#     dataframe=train_data,
#     x_col='file_path',
#     y_col='food_category',
#     target_size=(224, 224),
#     batch_size=12,
#     class_mode='categorical'
# )

# test_generator = datagen.flow_from_dataframe(
#     dataframe=test_data,
#     x_col='file_path',
#     y_col='food_category',
#     target_size=(224, 224),
#     batch_size=12,
#     class_mode='categorical'
# )

# model.fit(train_generator, validation_data=test_generator, epochs=10)

# # Make predictions on a new image
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# # Load and preprocess the image
# Load the dataset
dataset = pd.read_csv('food_dataset.csv')

# Split data into training and test sets



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

train_generator = datagen.flow_from_dataframe(
    dataframe=dataset,
    x_col='file_path',
    y_col='food_category',
    target_size=(224, 224),
    batch_size=10,
    class_mode='categorical'
)

from PIL import Image
import io
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# For loading the model
model = tf.keras.models.load_model('best_model.keras')

def predict_cal(file):
    image_data = io.BytesIO(file.read())
    image = Image.open(image_data)
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    class_label = train_generator.class_indices
    for key, value in class_label.items():
        if value == predicted_class:
            predicted_class = key
            break
    calories = {
        'tiramisu': 'Tiramisu - 240 calories per 100g',
        'sushi': 'Sushi - 45 calories per 100g',
        'ramen': 'Ramen - 68 calories per 100g',
        'french_toast': 'French Toast - 268 calories per 100g',
        'falafel': 'Falafel - 333 calories per 100g',
        'edamame': 'Edamame - 122 calories per 100g',
        'cannoli': 'Cannoli - 344 calories per 100g',
        'bibimbap': 'Bibimbap - 250 calories per 100g',
        'apple_pie': 'Apple Pie - 237 calories per 100g',
        'ice_cream': 'Ice Cream - 207 calories per 100g'
    }

    return calories[predicted_class]
    # return predicted_class

