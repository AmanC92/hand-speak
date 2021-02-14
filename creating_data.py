import os
import time

# To capture frames from web camera and detect faces from frames
import cv2

# For transforming key to uppercase
from string import ascii_uppercase


def make_dir(train, test):
    for c in ascii_uppercase:
        train_dir = train + "/" + c
        test_dir = test + "/" + c

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)


def get_indexes(path):
    labels = os.listdir(path)
    dictionary = {}

    for label in labels:
        length = len(os.listdir(path + "/" + label))
        dictionary[label] = length

    return dictionary


DATASET_PATH = '/Users/aman/PycharmProjects/hand-speak/dataset'
TRAIN_PATH = DATASET_PATH + '/asl_train'
TEST_PATH = DATASET_PATH + '/asl_test'
CUSTOM = 'space'

GREEN = (0, 255, 0)
x, y, w, h = 100, 100, 400, 400

make_dir(TRAIN_PATH, TEST_PATH)
index_dict = get_indexes(TRAIN_PATH)

# Grabs video capture of default web camera
vs = cv2.VideoCapture(0)
# Waiting while video feed is initializing
print("Initializing video feed please wait...")
time.sleep(5.0)


while True:
    # Capture the current frame and disregard the return value
    _, frame = vs.read()

    # Converting frame from BGR to RGB
    crop_img = frame[y:y + h, x:x + w]
    cv2.imshow("Picture Selection", crop_img)

    # Display the resulting frame with the provided title
    cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, cv2.LINE_4)
    cv2.imshow("Main image", frame)

    # Wait 1ms between frame captures
    key = cv2.waitKey(10)

    # if the escape key was pressed, break from the loop
    directory = TRAIN_PATH
    if ord('a') <= key <= ord('z'):
        print(chr(key).upper())
        cv2.imwrite(directory + "/" + chr(key).upper() + "/" + str(index_dict[chr(key).upper()]) + ".jpg", crop_img)
        index_dict[chr(key).upper()] += 1
    # If space key is entered create custom data in custom folder
    elif key == 32:
        print(CUSTOM)
        cv2.imwrite(directory + "/" + CUSTOM + "/" + str(index_dict[CUSTOM]) + ".jpg", crop_img)
        index_dict[CUSTOM] += 1
    elif key == 27:
        break

# Closing all windows - clean up
cv2.destroyAllWindows()