# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from skimage import feature
from imutils import paths
import pickle
import cv2
import os
import numpy as np

def quantify_image(image):
	# compute the histogram of oriented gradients feature vector for
	# the input image
	features = feature.hog(image, orientations=9,
		pixels_per_cell=(10, 10), cells_per_block=(2, 2),
		transform_sqrt=True, block_norm="L1")

	# return the feature vector
	return features

def load_split(path):
	# grab the list of images in the input directory, then initialize
	# the list of data (i.e., images) and class labels
	imagePaths = list(paths.list_images(path))
	data = []
	labels = []

	# loop over the image paths
	for imagePath in imagePaths:
		# extract the class label from the filename
		label = imagePath.split(os.path.sep)[-2]

		# load the input image, convert it to grayscale, and resize
		# it to 200x200 pixels, ignoring aspect ratio
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (200, 200))

		# threshold the image such that the drawing appears as white
		# on a black background
		image = cv2.threshold(image, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

		# quantify the image
		features = quantify_image(image)

		# update the data and labels lists, respectively
		data.append(features)
		labels.append(label)

	# return the data and labels
	return (np.array(data), np.array(labels))

classifiers = ["SVC", "LinearSVC", "RandomForest", "DecisionTree"]

def load_classifier(name):
    filename = str(name) + '.pkl'
    with open(filename, "rb") as clf_infile:
        clf = pickle.load(clf_infile)
    return clf

dataset= 'data/spiral'

# define the path to the training and testing directories
trainingPath = os.path.sep.join([dataset, "training"])
testingPath = os.path.sep.join([dataset, "testing"])

# loading the training and testing data
print("[INFO] loading data...")
(trainX, trainY) = load_split(trainingPath)
(testX, testY) = load_split(testingPath)
# encode the labels as integers
le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

for classifier in classifiers:
    clf = load_classifier(classifier)
    testingPaths = list(paths.list_images(testingPath))
    idxs = np.arange(0, len(testingPaths))
    idxs = np.random.choice(idxs, size=(25,), replace=False)
    images = []

    # loop over the testing samples
    for i in idxs:
    	# load the testing image, clone it, and resize it
    	image = cv2.imread(testingPaths[i])
    	output = image.copy()
    	output = cv2.resize(output, (128, 128))

    	# pre-process the image in the same manner we did earlier
    	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    	image = cv2.resize(image, (200, 200))
    	image = cv2.threshold(image, 0, 255,
    		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    	# quantify the image and make predictions based on the extracted
    	# features using the last trained Random Forest
    	features = quantify_image(image)
    	preds = clf.predict([features])
    	label = le.inverse_transform(preds)[0]