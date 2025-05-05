# Butterfly Species Classification Model

A deep learning model built using TensorFlow and Keras to classify butterfly species from images.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Details](#model-details)
- [Classes](#classes)
- [Example Code](#example-code)
- [Notes](#notes)

## Overview
This project uses a custom-trained CNN model to identify butterfly species from images. The model is trained on a dataset of images of various butterfly species and can be used to classify new images.

## Requirements
- Python 3.10+
- TensorFlow 2+
- Keras
- NumPy
- Pillow

## Usage
```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("path/to/model.keras")

# Load an image
image = Image.open("path/to/image.jpg")

# Convert the image to a NumPy array
image_numpy = np.array(image)

# Make predictions
pred = model.predict(np.expand_dims(image_numpy, axis=0))

# Get the predicted class
argmax = np.argmax(pred)
