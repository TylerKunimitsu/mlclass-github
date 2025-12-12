# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Intro to Computer Vision
# In this notebook you will use the [quickdraw](https://quickdraw.withgoogle.com/data) dataset to build a simple image recongnition neural net.
#
# We will be using matplotlib for this assignment so start with installing it via pip.

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt


# %% [markdown]
# # Step 1: Data
#
# Quickdraw supports A LOT of images. Rather than training on all 50 million, we will start with just a few. 
# Your model should train on at least 5 categories of drawings. 
# This will be more fun if they look similar to each other. 
#
# Go to [quickdraw](https://quickdraw.withgoogle.com/data) and select at least 5 categories to use for your model. These are stored in a Google Cloud storage bucket called quickdraw_dataset.
#
# Load each of your categories by these steps:
#
# 1. ` pip install gsutil`
#
# 2.  `gsutil -m cp 'gs://quickdraw_dataset/full/numpy_bitmap/<your category>.npy' .`
#
#
# Do step 2 once per category. 
#
# *You will need to do this both locally and in your gcloud vm. (they are too big to push to github)

# %% [markdown]
# ## Load Your Data
#
# Your data will be in a .npy format (numpy array). We will load it differently this time: for example
#
# `data_airplanes = np.load("airplane.npy")`

# %%
# Helper function to show images from your dataset
def show_image(data):
    index = random.randint(0, len(data)-1)
    img2d = data[index].reshape(28, 28)
    plt.figure()
    plt.imshow(img2d)
    plt.colorbar()
    plt.grid(False)
    plt.show()


# %%
# Load your first cateogry of images
data_line = np.load("line.npy")

# %%
# Now use the function above to show an image from your data
show_image(data_line)

# %%
# Now load your second category of images
data_triangle = np.load("triangle.npy")

# %%
# visualize your second category of images
show_image(data_triangle)

# %%
# Now load your third category of images
data_square = np.load("square.npy")

# %%
# visualize your third category of images
show_image(data_square)

# %%
# Now load your 4th category of images
data_circle = np.load("circle.npy")

# %%
# visualize your 4th category of images
show_image(data_circle)

# %%
# Now load your 5th category of images
data_star = np.load("star.npy")

# %%
# visualize your 5th category of images
show_image(data_star)

# %% [markdown]
# ## Preprocess your data
#
# Now it's time to define X and y. We want our X and y data to include ALL of the categories you loaded above.
#
# You can combine np.arrays together using something like `np.vstack`.
#

# %%
# define X by combining all loaded data from above
X = np.vstack((data_line, data_circle, data_triangle, data_square, data_star))
print(len(X[0]))

# %%
# verify the X was defined correctly
assert X.shape[1] == 784
assert X.shape[0] >= 550000
print(len(data_line))
print(len(data_circle))
print(len(data_triangle))
print(len(data_square))
print(len(data_star))

# %% [markdown]
# Now it's time to define y. Recall that y is an array of "correct labels" for our data. For example, ['airplane', 'airplane', 'cat', 'bird', ....]
#
#
# In the above step you created an array of images stored in X of 5 different categories. Now, create an array of labels to match those images. 

# %%
# define y by creating an array of labels that match X
encoder=LabelEncoder()

array1 = np.full(143549, "line", dtype=object)
array2 = np.full(122876, "circle", dtype=object)
array3 = np.full(123170, "triangle", dtype=object)
array4= np.full(125145, "square", dtype=object)
array5 = np.full(137619, "star", dtype=object)
y = np.concatenate((array1, array2, array3, array4, array5))
y = encoder.fit_transform((y))
print(encoder.transform(["line", "circle", "triangle", "square", "star"]))

# %%
# verify that y is the same length as X
assert len(y) == len(X)

# %% [markdown]
# ## Split your data
#
# Split your data is 80 training/ 20 testing as usual. 

# %%
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# %% [markdown]
# # Define your model
#
# Now define your neural network using tensorflow. 

# %%
# Define your model with the correct input shape and appropriate layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(len(X[0]), input_shape=[(len(X[0]))]),
    tf.keras.layers.Dense(67, activation="relu"),
    tf.keras.layers.Dense(5, activation="relu")
])

# %%
# Compile your model
lr=0.0001

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    metrics=["accuracy"]
)

# %% [markdown]
# ## Train your model locally
#
# Train your model. 

# %%
# Fit your model 
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)


# %% [markdown]
# # Train your model on gcloud
#
# Your model likely is a bit slow... move it to the Gcloud gpu. Since we can't run notebooks in gcloud, you can convert it to a python file first by running:
#
# `jupytext --to py quickdrawClassifier.ipynb`
#
# Then push the generated quickdrawClassifier.py file to github and then pull it down to your gcloud laptop to run.
