# Using Keras to process our training data and train our model
from keras.preprocessing import image
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D

# To save and load model history
from joblib import dump, load

# To plot our data and get appropriate GRID_SIZE for subplots
import matplotlib.pyplot as plt
from math import ceil, sqrt

# To grab environment variables and perform checks
import os
from os.path import exists

# Stops TF optimization warnings from displaying
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Sets models target size, learning rate, number of passes of our training data (epochs) and batch size
TARGET_SIZE = (150, 150)
LEARNING_RATE = 1e-4
EPOCHS = 25
BATCH_SIZE = 32

# Dynamically sets n * n grid size to plot our sample augmented images generated
GRID_SIZE = ceil(sqrt(BATCH_SIZE))

# Path to dataset, to model, and to save/load history of model
DATASET_PATH = './data'
MODEL_PATH = 'asl_model'
HISTORY_PATH = MODEL_PATH + '/history.joblib'