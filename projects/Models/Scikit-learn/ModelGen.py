# USASGE
# python ModelGen.py

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import pickle
import cv2
import os

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

def dump_classifier(clf, svr):
    filename = str(svr) + '.pkl'
    with open(filename, "wb") as clf_outfile:
        pickle.dump(clf, clf_outfile)



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

def prec_recall (clf, svr):
    """
    Make performance analys of current classifier and return precision, recall and accuracy
    :param clf: classifier
    :param svr: classifier name for save purposes
    :return precision: avg precision of currrent classifier
    :return recall: avg recall of currrent classifier
    :return accuracy: avg accuracy of currrent classifier
    """
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
      
    ### fit the classifier using training set, and test on test set
    clf.fit(trainX, trainY)
    predictions = clf.predict(testX)
    for prediction, truth in zip(predictions, testY):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        elif prediction == 1 and truth == 1:
            true_positives += 1
        else:
            print("Warning: Found a predicted label not == 0 or 1.")
            print("All predictions should take value 0 or 1.")
            print("Evaluating performance for processed predictions:")
            break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
    except:
        print("Got a divide by zero when trying out:", clf)
        print("Precision or recall may be undefined due to a lack of true positive predicitons.")
    dump_classifier(clf, svr)
    return (precision, recall, accuracy)

# initialize our trials dictionary
trials = {}

# create few different ML models and test it with different parameters
# Compare performance
from sklearn.naive_bayes import GaussianNB ## NAIVE BAYES
from sklearn.svm import SVC ## SUPPORT VECTOR MACHINE
from sklearn.svm import LinearSVC # LINEAR SUPPORT VECTOR MACHINE
from sklearn.ensemble import RandomForestClassifier ## RANDOM FOREST
from sklearn import tree ## DECISION TREE

Parameters_Grid = {
    "GaussianNB" : [
        GaussianNB(),
        {
            
        }
    ],
    "SVC" : [
        SVC(),
        {
            'C': [1, 10, 100, 1000], 
            'gamma': [0.01, 0.001, 0.0001], 
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
        }
    ],
    "LinearSVC" : [
        LinearSVC(),
        {
            'C': [1, 10, 100, 1000], 
            'loss': ['hinge', 'squared_hinge'], 
        }
    ],
    "RandomForest" : [
        RandomForestClassifier(),
        {
            'n_estimators': [1,2,3,5,8,9,10,11,15,20,21],
            'criterion': ['gini','entropy'],
            'max_features': ["auto","sqrt","log2"],
            'max_leaf_nodes': [2,3,4,5,6,7,8,9,10]
        }
    ],
    "DecisionTree" : [
        tree.DecisionTreeClassifier(),
        {
            'criterion': ['gini','entropy'],
            'splitter':  ['best','random'],
            'max_features': ["auto","sqrt","log2"],
            'max_leaf_nodes': [2,3,4,5,6,7,8,9,10]
        }
    ]
}

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
for svr in Parameters_Grid:
    clf = GridSearchCV(Parameters_Grid[svr][0],Parameters_Grid[svr][1])
    print("\n","-"*34)
    print(svr)
    print("-"*80)
    print("Tuned classifier:")
    print("-"*80)
    print(clf.estimator)
    print("-"*80)

#Performance test
for svr in Parameters_Grid:
    ctl = {
        'GaussianNB': GaussianNB(priors=None),
        'SVC': SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                        max_iter=-1, probability=False, random_state=None, shrinking=True,
                        tol=0.001, verbose=False),
        'LinearSVC': LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                            intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                            multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                            verbose=0),
        'RandomForest': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                max_depth=None, max_features='auto', max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                                oob_score=False, random_state=None, verbose=0,
                                warm_start=False),
        'DecisionTree': tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                                max_features=None, max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                                splitter='best')
    }[svr]
    #Performance results
    print("\n","-"*34)
    print(svr)
    precision, recall, accuracy = prec_recall(clf, svr)
    print("Accuracy:\t", accuracy)
    print("Precision:\t", precision)
    print("Recall:\t\t", recall)
    print("_"*80)


# # randomly select a few images and then initialize the output images
# # for the montage
# testingPaths = list(paths.list_images(testingPath))
# idxs = np.arange(0, len(testingPaths))
# idxs = np.random.choice(idxs, size=(25,), replace=False)
# images = []

# # loop over the testing samples
# for i in idxs:
# 	# load the testing image, clone it, and resize it
# 	image = cv2.imread(testingPaths[i])
# 	output = image.copy()
# 	output = cv2.resize(output, (128, 128))

# 	# pre-process the image in the same manner we did earlier
# 	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	image = cv2.resize(image, (200, 200))
# 	image = cv2.threshold(image, 0, 255,
# 		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# 	# quantify the image and make predictions based on the extracted
# 	# features using the last trained Random Forest
# 	features = quantify_image(image)
# 	preds = model.predict([features])
# 	label = le.inverse_transform(preds)[0]

# 	# draw the colored class label on the output image and add it to
# 	# the set of output images
# 	color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
# 	cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
# 		color, 2)
# 	images.append(output)

# # create a montage using 128x128 "tiles" with 5 rows and 5 columns
# montage = build_montages(images, (128, 128), (5, 5))[0]

# # show the output montage
# cv2.imshow("Output", montage)
# cv2.waitKey(0)