import os
import random
import numpy as np
from io import BytesIO

# Plotting and dealing with images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf

# Interactive widgets
from ipywidgets import widgets

BASE_DIR = r'C:\Users\toran\Documents\GitHub\ML_PadiCare\data'

train_dir = os.path.join(BASE_DIR, 'train')
test_dir = os.path.join(BASE_DIR, 'test')
validation_dir = os.path.join(BASE_DIR, 'val')
# Directory with training 
train_leafblight_dir = os.path.join(train_dir, 'bacterial_leaf_blight')
train_panicleblight_dir = os.path.join(train_dir, 'bacterial_panicle_blight')
train_brownspot_dir = os.path.join(train_dir, 'brown_spot')
train_sheathblight_dir = os.path.join(train_dir, 'rice_sheath_blight')
train_normal_dir = os.path.join(train_dir, 'normal')
# Directory with validation 
validation_leafblight_dir = os.path.join(validation_dir, 'bacterial_leaf_blight')
validation_panicleblight_dir = os.path.join(validation_dir, 'bacterial_panicle_blight')
validation_brownspot_dir = os.path.join(validation_dir, 'brown_spot')
validation_sheathblight_dir = os.path.join(validation_dir, 'rice_sheath_blight')
validation_normal_dir = os.path.join(validation_dir, 'normal')
# Directory with test 
test_leafblight_dir = os.path.join(test_dir, 'bacterial_leaf_blight')
test_panicleblight_dir = os.path.join(test_dir, 'bacterial_panicle_blight')
test_brownspot_dir = os.path.join(test_dir, 'brown_spot')
test_sheathblight_dir = os.path.join(test_dir, 'rice_sheath_blight')
test_normal_dir = os.path.join(test_dir, 'normal')
print(f"Contents of base directory: {os.listdir(BASE_DIR)}")
print(f"\nContents of train directory: {train_dir}")
print(f"\nContents of validation directory: {validation_dir}")
print(f"\nContents of test directory: {test_dir}")

train_leafblight_fnames = os.listdir(train_leafblight_dir)
train_panicleblight_fnames = os.listdir(train_panicleblight_dir )
train_brownspot_fnames = os.listdir(train_brownspot_dir)
train_sheathblight_fnames = os.listdir(train_sheathblight_dir )
train_normal_fnames = os.listdir(train_normal_dir)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])

model = tf.keras.models.Sequential([
    data_augmentation,
    tf.keras.Input(shape=(150, 150, 3)),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])