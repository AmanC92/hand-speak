# Loading our model, manipulate images, and get pretrained VGG16.
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16

# Compiling our model with additional layers, and transforming data to categorical
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras import utils

# To actually divide our training and testing data
from sklearn.model_selection import train_test_split

# To resize our images to the given target size from a provided image path
import cv2

# To handle our image data and create/remove nesting to image arrays
import numpy as np

# To plot our loss/accuracy data as well as images
import matplotlib.pyplot as plt

# To save and load model history
from joblib import dump, load

# To grab environment variables and perform checks
import os
from os.path import exists

# Stops TF optimization warnings from displaying
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Setting hyper parameters, target size however is dependent on what VGG16 model was trained on
TARGET_SIZE = (32, 32)
TEST_SPLIT = 0.2
LEARNING_RATE = 1e-4
EPOCHS = 25
BATCH_SIZE = 32
DROPOUT = 0.6

# Model parameters and metrics. Input shape is reliant on VGG16 shape.
INPUT_SHAPE = (32, 32, 3)
OPTIMIZER = 'Adam'
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']
CATEGORIES = 30

# Path to dataset, train & test path, to model, and to save/load history of model
DATASET_PATH = './dataset'
TRAIN_PATH = DATASET_PATH + '/asl_train'
TEST_PATH = DATASET_PATH + '/asl_test'
MODEL_PATH = 'asl_model'
HISTORY_PATH = MODEL_PATH + '/history.joblib'


# Trains our model by using parameters set above and a pretrained VGG16 as the starting point.
def train_model():
    symbols, labels = get_image_data(TRAIN_PATH)

    X_train, X_test, Y_train, Y_test = train_test_split(symbols, labels, test_size=TEST_SPLIT, stratify=labels)

    # Categorizing the labels
    Y_train = utils.to_categorical(Y_train)
    Y_test = utils.to_categorical(Y_test)

    # Normalizing our data to be between 0.0 and 1.0 for faster computations & consistency with VGG16
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Setting up VGG16 to be the starting point to our additional layers
    classifier_vgg16 = VGG16(input_shape=INPUT_SHAPE,
                             include_top=False,
                             weights='imagenet')

    # To not retrain already trained layers in VGG16 for when we add our layers on afterwards and train.
    for layer in classifier_vgg16.layers:
        layer.trainable = False

    # Add the additional layers for our model and have the last layer be to the size of the
    # available categories for our model.
    classifier = Flatten()(classifier_vgg16.output)
    classifier = Dense(units=512, activation='relu')(classifier)
    classifier = Dropout(DROPOUT)(classifier)
    classifier = Dense(units=CATEGORIES, activation='softmax')(classifier)

    # Setting up our model to start from VGG16 input and then go to the additional layers we have set up
    model = Model(inputs=classifier_vgg16.input,
                  outputs=classifier)

    # Compile model with parameters set at start
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=METRICS)

    # Train our model
    history = model.fit(X_train,
                        Y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_test, Y_test)
                        ).history

    # Saves model as well as the history associated with it.
    model.save(MODEL_PATH)
    dump(history, HISTORY_PATH)


# Hot encodes all image data in our dataset
def get_image_data(folder):
    symbols = sorted(os.listdir(folder))
    X = []
    Y = []

    # Iterates over all possible images in our data and adds them to the corresponding symbol and index
    for enum, label in enumerate(symbols):
        for img in os.listdir(folder + "/" + label):
            if img != '.DS_Store':
                path = folder + "/" + label + "/" + img
                resized_img = cv2.resize(cv2.imread(path), TARGET_SIZE)
                X.append(resized_img)
                Y.append(enum)

    # Converts to a numpy array before returning
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


# Returns the predicted category for a given image path/data.
# Alternatively also can print the category and show the image supplied if needed.
def predict(image_path='', img_data=None, print_data=True):
    # Loading in our previously trained model using joblib
    if img_data is None:
        img_data = []
    saved_model = load_model(MODEL_PATH)

    # Load in image and set target size to what model was trained on.
    # Image loaded is either from file path or from a given image array.
    if image_path:
        image_data = image.load_img(image_path, target_size=TARGET_SIZE)
    else:
        image_data = img_data

    # Convert to a numpy array, rescales to what we trained our model on and adds additional level of nesting
    image_array = np.array(image_data)
    image_array = image_array / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    # Gets prediction of passed image
    prediction = saved_model.predict(image_batch)

    # Gets all symbol categories from evaluation path
    symbols = sorted(os.listdir(TRAIN_PATH))

    # Iterates over array and checks to see which category the model predicted
    # and then rewrites prediction to the corresponding symbol
    print(prediction[0])
    for i in range(len(prediction[0])):
        if prediction[0][i] >= 0.5:
            prediction = symbols[i]
            break
        elif i == len(prediction[0]) - 1:
            prediction = 'n/a'

    # If we want to show our data results, then we will print
    # the category as well as show the input image that was used.
    if print_data:
        plt.imshow(image_data)
        plt.show()
        print(prediction)

    return prediction


# Utility function that plots Validation & Training Loss/Accuracy from our model
def plot_history(file_path):
    # Load history of our model from designated file path
    model_history = load(file_path)
    print(model_history)

    # Plots the Validation & Training Loss
    plt.plot(range(1, EPOCHS + 1), model_history['loss'], 'g', label='Training loss')
    plt.plot(range(1, EPOCHS + 1), model_history['val_loss'], 'b', label='Validation loss')
    plt.title('Training and Validation loss'), plt.xlabel('Epochs'), plt.ylabel('Loss'), plt.legend()
    plt.show()

    # Plots the Validation & Training Accuracy
    plt.plot(range(1, EPOCHS + 1), model_history['accuracy'], 'g', label='Training accuracy')
    plt.plot(range(1, EPOCHS + 1), model_history['val_accuracy'], 'b', label='Validation accuracy')
    plt.title('Training and Validation accuracy'), plt.xlabel('Epochs'), plt.ylabel('Accuracy'), plt.legend()
    plt.show()


# Trains and saves model if not done already
if not exists(MODEL_PATH):
    train_model()

if __name__ == '__main__':
    # Plots Validation & Training Loss/Accuracy
    plot_history(HISTORY_PATH)

    # Uses our utility function to predict the category from our model given an image path/data.
    # predict('./dataset/asl_test/A')
