import os
import cv2
import numpy as np
from imutils import paths
import imutils

data_set = "/Users/admin/Development/Dev/ComputerVision/SmileDetection/dataset/SMILEsmileD/SMILEs/positives"
output = "output"
# initialize the list of data and labels
data = []
labels = []


def convert_data():
    X = []
    # loop over the input images
    for imagePath in sorted(list(paths.list_images(data_set))):
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = imutils.resize(image, width=28)
        T = np.zeros([28, 28, 1])
        T[:, :, 0] = image

        # extract the class label from the image path and update the label list
        label = imagePath.split(os.path.sep)[-3]
        label = "smiling" if label == "positives" else "not_smiling"
        # labels.append(label)
        X.append((image, label))

    for _ in range(10):
        np.random.shuffle(X)

    train_data, test_data = X[:10000], X[3000:]

    np.save('/Users/admin/Development/Dev/ComputerVision/SmileDetection/dataset/data/' +
            'train.npy', train_data)
    np.save('/Users/admin/Development/Dev/ComputerVision/SmileDetection/dataset/data/' +
            'test.npy', test_data)
