import sys
sys.path.append("/Users/Prerna/Desktop/Prerna/NTU/Courses-Year4-Sem1/NLP/SemEval15/code");


import extractBigrams #importing file for extracting types from input
from collections import OrderedDict # OrderedDict to sort feature names in dictionary struct

#filename = "gridsearch.tsv";
filename = "../rawdata/train/twitter-train-cleansed-B_rmnotav.tsv";
startColIdx = 3;

typesDict = extractBigrams.extractBigramTypes(filename, startColIdx);

#The line below does NOT work!
#OrderedDict(sorted(typesDict.items(), key=lambda t: t[0]));

print "Number of types extracted = ", len(typesDict);

import extractBigramFeatureVecX; #importing file for extracting feature vector X from training data
print "Feature Vector X Extraction Started";
Xfeatures = extractBigramFeatureVecX.extractBigramFeatureVecX(filename, startColIdx, typesDict);
print "Feature Vector X of size ", len(Xfeatures), " extracted";

labelIdx = 2;
import handleClassLabels;
print "Class Label Vector Y Extraction Started";
YLabels = handleClassLabels.extractClassLabels(filename, labelIdx);
print "Class Label Vector Y of size ", len(YLabels), " extracted";

#Setting up scaler for standardisation
from sklearn import preprocessing
scaler = preprocessing.StandardScaler();

import numpy as np
#from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

C_range = 10. ** np.arange(-3, 4);

param_grid = dict(C=C_range)

# Training SVM
from sklearn import svm
import evaluate;

print "Declaring SVM"
clf = svm.LinearSVC(class_weight='auto', penalty='l1', dual=0); # linearsvc2

print "standardising training data"
Xfeatures = scaler.fit_transform(Xfeatures, YLabels);
#print "splitting training and test data"
#X_train, X_test, y_train, y_test = train_test_split(Xfeatures, YLabels, test_size=0.2, random_state=0)

print "Doing GridSearch"
grid = GridSearchCV(clf, param_grid=param_grid, scoring='f1')

grid.fit(Xfeatures, YLabels)

print("The best classifier is: ", grid.best_estimator_)

