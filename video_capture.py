# Using Keras to load our model
from keras.models import load_model

# To remove TF warnings & have system sleep while VideoCapture is booting
import os
import time

# To increase dimensions of np arrays
import numpy as np

# To capture frames from web camera and detect faces from frames
import cv2

# Stops TF optimization warnings from displaying
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Path to dataset, to model, and to save/load history of model
MODEL_PATH = 'asl_model'

# Target size our model was trained on
TARGET_SIZE = (32, 32)

# Load in our previously trained mask model
model = load_model(MODEL_PATH)

# Path to dataset & train path
DATASET_PATH = './dataset'
TRAIN_PATH = DATASET_PATH + '/asl_train'

# Colours for our frames and text
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Gets all symbol categories from evaluation path
symbols = sorted(os.listdir(TRAIN_PATH))

# Dimensions for our cropped frame
x, y, w, h = 100, 100, 400, 400

# Text to be outputted
pred_text = ''

# Grabs video capture of default web camera
vs = cv2.VideoCapture(0)
# Waiting while video feed is initializing
print("Initializing video feed please wait...")
time.sleep(5.0)


def prediction(image):
    # Convert to a numpy array, rescales to what we trained our model on and adds additional level of nesting
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    # Gets prediction of passed image
    predict = model.predict(image_batch)

    # Iterates over array and checks to see which category the model predicted
    # and then rewrites prediction to the corresponding symbol
    res = 'n/a'
    for i in range(len(predict[0])):
        score = 0

        if predict[0][i] >= 0.25 and predict[0][i] > score:
            # noinspection PyUnusedLocal
            score = predict[0][i]
            res = symbols[i]

    return res


while True:
    # Capture the current frame and disregard the return value
    _, frame = vs.read()

    # Converting frame from BGR to RGB
    crop_img = frame[y:y + h, x:x + w]
    rgb_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_img, TARGET_SIZE)

    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = y - 2
    cv2.rectangle(frame, (0, 0), (width, height), BLACK, -1)
    cv2.putText(frame, pred_text.title(), (x + int(width/3), int(height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, WHITE, 2)

    cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, cv2.LINE_4)

    # Display the resulting frame with the provided title
    cv2.imshow("Hand Speak", frame)

    # Wait 1ms between frame captures
    key = cv2.waitKey(1)

    # Make prediction from frame if key is 'q'
    if key == ord('p'):
        pred = prediction(resized_img).lower()
        print(pred)

        if pred == 'del':
            pred_text = pred_text.rsplit(' ', 1)[0]
        elif pred == 'space':
            pred_text += ' '
        else:
            pred_text += pred

    # if the key `q` or escape was pressed, break from the loop
    if key == ord('q') or key == 27:
        break

# Closing all windows - clean up
cv2.destroyAllWindows()