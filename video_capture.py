# Using Keras to load our model
from keras.models import load_model

# To remove TF warnings & have system sleep while VideoCapture is booting
import os
import time

# To increase dimensions of np arrays
import numpy as np

# To capture frames from web camera and detect faces from frames
import cv2

import matplotlib.pyplot as plt

# Stops TF optimization warnings from displaying
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Path to dataset, to model, and to save/load history of model
MODEL_PATH = 'asl_model'

# Target size our model was trained on
TARGET_SIZE = (32, 32)

# Load in our previously trained mask model
model = load_model(MODEL_PATH)

DATASET_PATH = './dataset'
TRAIN_PATH = DATASET_PATH + '/asl_train'


# Grabs video capture of default web camera
vs = cv2.VideoCapture(0)
# Waiting while video feed is initializing
print("Initializing video feed please wait...")
time.sleep(5.0)

# Gets all symbol categories from evaluation path
symbols = sorted(os.listdir(TRAIN_PATH))


def prediction(image):
    # Convert to a numpy array, rescales to what we trained our model on and adds additional level of nesting
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    # Gets prediction of passed image
    prediction = model.predict(image_batch)

    # Iterates over array and checks to see which category the model predicted
    # and then rewrites prediction to the corresponding symbol
    print(prediction[0])
    res = 'n/a'
    for i in range(len(prediction[0])):

        score = 0
        if prediction[0][i] >= 0.25 and prediction[0][i] > score:
            score = prediction[0][i]
            res = symbols[i]

    plt.imshow(image)
    plt.show()

    # If we want to show our data results, then we will print
    # the category as well as show the input image that was used.

    return res


while True:
    # Capture the current frame and disregard the return value
    _, frame = vs.read()

    x = 100
    w = 400
    y = 100
    h = 400

    # Converting frame from BGR to RGB
    crop_img = frame[y:y + h, x:x + w]
    rgb_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_img, TARGET_SIZE)

    # Display the resulting frame with the provided title
    cv2.imshow("Hand Speak", frame)
    cv2.imshow("cropped", crop_img)

    # Wait 1ms between frame captures
    key = cv2.waitKey(100)

    # Make prediction from frame if key is 'q'
    if key == ord('p'):
        print(prediction(resized_img))

    # if the key `q` or escape was pressed, break from the loop
    if key == ord('q') or key == 27:
        break

# Closing all windows - clean up
cv2.destroyAllWindows()