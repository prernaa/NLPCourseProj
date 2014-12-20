import sys
sys.path.append("/Users/Prerna/Desktop/Prerna/NTU/Courses-Year4-Sem1/NLP/SemEval15/code");

import extractTrigrams #importing file for extracting bigram types from input
import extractTypes    #importing file for extracting unigram types from input
from collections import OrderedDict # OrderedDict to sort feature names in dictionary struct

#filename = "addedTest.tsv";
filename = "../rawdata/train/twitter-train-cleansed-B_rmnotav_ADDEDtest.tsv";
startColIdx = 3;

typesDictTrigrams = extractTrigrams.extractTrigrams(filename, startColIdx);
typesDictUnigrams = extractTypes.extractTypes(filename, startColIdx);

print "Number of trigram types extracted = ", len(typesDictTrigrams);
print "Number of unigram types extracted = ", len(typesDictUnigrams);

import extractTrigramFeatureVecX; #importing file for extracting trigram feature vector from training data
import extractFeatureVecX; #importing file for extracting unigram feature vector from training data

print "Feature Vector Extraction Started";
XfeaturesUnigrams = extractFeatureVecX.extractFeatureVecX(filename, startColIdx, typesDictUnigrams);
XfeaturesTrigrams = extractTrigramFeatureVecX.extractTrigramFeatureVecX(filename, startColIdx, typesDictTrigrams);

#combining the two feature vectors below
Xfeatures=[];
Ndata=len(XfeaturesUnigrams)#number of sentences in training data 
lUTypes=len(XfeaturesUnigrams[0]); #number of unigram types
lTTypes=len(XfeaturesTrigrams[0]); #number of trigram types
for i in range (0,NData):
    XFeatures.append(XFeaturesUnigrams[i]);
    for j in range (0,lTTypes):
        XFeatures[i].append(XfeaturesTrigrams[i][j]);#combining of feature vectors finished

#print "Feature Vector of size ", len(Xfeatures), " extracted";

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
