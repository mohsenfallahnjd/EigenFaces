import cv2
import os
import glob
import numpy as np
import time
img_dir = "./Face_DataSet/"  # Enter Directory of all images
data_path = os.path.join(img_dir, '*g')
files = glob.glob(data_path)


def createDataMatrix(images):
    print("Creating data matrix", end=" ... ")
    numImages = len(images)
    sz = images[0].shape
    data = np.zeros((numImages, sz[0] * sz[1]), dtype=np.float32)
    for i in range(0, numImages):
        image = images[i].flatten()
        data[i, :] = image

    print("DONE")
    return data


def createNewFace(*args):
    # Start with the mean image
    output = averageFace

    # Add the eigen faces with the weights
    for i in range(0, NUM_EIGEN_FACES):
        sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars")
        weight = sliderValues[i] - int(MAX_SLIDER_VALUE/2)
        output = np.add(output, (eigenFaces[i] * weight))

    # Display Result at 2x size
    output = cv2.resize(output, (0, 0), fx=2, fy=2)
    cv2.imshow("Result", output/255)


if __name__ == '__main__':

    # Number of EigenFaces
    NUM_EIGEN_FACES = 30
    # Maximum weight
    MAX_SLIDER_VALUE = 255

    images = []
    for f1 in files:
        img = cv2.imread(f1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
        imgvector = np.reshape(img, -1)
        images.append(img)

    data = createDataMatrix(images)
    mean, eigenVectors = cv2.PCACompute(
        data, mean=None, maxComponents=NUM_EIGEN_FACES)

    # Size of images
    sz = images[0].shape

    averageFace = mean.reshape(sz)

    eigenFaces = []

    for eigenVector in eigenVectors:
        eigenFace = eigenVector.reshape(sz)
        eigenFaces.append(eigenFace)

    # Create window for displaying Mean Face
    cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)

    # Display result at 2x size
    output = cv2.resize(averageFace, (0, 0), fx=2, fy=2)
    cv2.imshow("Result", output)

    # Create Window for trackbars
    cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)

    sliderValues = []

    # Create Trackbars
    for i in range(0, NUM_EIGEN_FACES):
        sliderValues.append(int(MAX_SLIDER_VALUE/2))
        cv2.createTrackbar("Weight" + str(i), "Trackbars",
                           int(MAX_SLIDER_VALUE/2), MAX_SLIDER_VALUE, createNewFace)

    cv2.waitKey(0)
    cv2.destroyAllWindows()