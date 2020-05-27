# import the necessary packages
from skimage import feature
from imutils import paths
import pickle
import cv2
import json
import numpy as np

def quantify_image(image):
	# compute the histogram of oriented gradients feature vector for
	# the input image
	features = feature.hog(image, orientations=9,
		pixels_per_cell=(10, 10), cells_per_block=(2, 2),
		transform_sqrt=True, block_norm="L1")

	# return the feature vector
	return features

def load_classifier(name):
    filename = str(name) + '.pkl'
    with open(filename, "rb") as clf_infile:
        clf = pickle.load(clf_infile)
    return clf

def prepare_incoming_image(image):
    # quantify the image and make predictions based on the extracted
    # features using the last trained Random Forest
    features = quantify_image(image)
    return features

def get_prediction_by_classifier(classifier, features):
    clf = load_classifier(classifier)
    features = prepare_incoming_image(features)
    preds = clf.predict([features])
    return preds

def prepare_prediction_result(classifier, pred):
    res = dict()
    res["model"] = str(classifier)
    res["prediction"] = 'PD possitive' if pred[0] == 1 else 'PD negative'
    return res

def get(features):
    features = np.array(features)
    result = []
    classifiers = ["GaussianNB", "SVC", "LinearSVC", "RandomForest", "DecisionTree"]
    for classifier in classifiers:
        pred = get_prediction_by_classifier(classifier, features)
        result.append(prepare_prediction_result(classifier, pred))
        
    return json.dumps(result)
