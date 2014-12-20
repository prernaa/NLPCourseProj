import sys
sys.path.append("/Users/Prerna/Desktop/Prerna/NTU/Courses-Year4-Sem1/NLP/SemEval15/code");


import extractBigrams #importing file for extracting types from input
from collections import OrderedDict # OrderedDict to sort feature names in dictionary struct

#filename = "addedTest.tsv";
filename = "../rawdata/train/twitter-train-cleansed-B_rmnotav_new.tsv";
#filename = "../rawdata/train/twitter-train-cleansed-B_rmnotav_ADDEDtest.tsv";
startColIdx = 3;

typesDict = extractBigrams.extractBigramTypes(filename, startColIdx);

#The line below does NOT work!
#OrderedDict(sorted(typesDict.items(), key=lambda t: t[0]));

print "Number of types extracted = ", len(typesDict);

#To print the types and their total number of occurrences... uncomment for speed
#for key in typesDict:
   # print key+" : ",typesDict[key]," ";

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


# Training SVM
from sklearn import svm
from sklearn import linear_model
print "Declaring SVM"
#clf = svm.LinearSVC(); # linearsvc1
clf = svm.LinearSVC(C=100.0, class_weight='auto', penalty='l1', dual=0); # linearsvc2
#clf = svm.SVC(cache_size = 1000, class_weight='auto', kernel = 'poly'); # Predicts all as POSITIVE :((
#clf = linear_model.SGDClassifier();  # not tried yet
print "standardising training data"
Xfeatures = scaler.fit_transform(Xfeatures, YLabels);
print "Fitting Data To SVM"
clf.fit(Xfeatures, YLabels);
print "SVM trained"


#Saving Trained Classifier
from sklearn.externals import joblib
print "Saving SVM"
fileToSave = "BigramSVMClassifier_C100.0.joblib.pkl";
_ = joblib.dump(clf, fileToSave, compress=9);
print "Classifier SAVED!";


# Extract feature vector from test data

#testFilename = "validation.tsv"
testFilename = "../rawdata/test/input/twitter-dev-input-B_rmnotav_new.tsv"

#testFilename = "svc.tsv"
#testFilename = "../rawdata/test/AB_SemEval2013_task2_test_fixed/input/sms-test-input-B.tsv"
testStartColIdx = 3
print "XTEST Feature Vector Extraction Started";
XTestFeatures = extractBigramFeatureVecX.extractBigramFeatureVecX(testFilename, testStartColIdx, typesDict);
print "XTEST Feature Vector of size ", len(XTestFeatures), " extracted";
print "standardising test data"
XTestFeatures = scaler.transform(XTestFeatures);
#Using Trained SVM to classify data
print "Predicting labels for test data"
predictedLabels = clf.predict(XTestFeatures);

#Writing predicted labels to file
writeToFile = "BigramSVMIgnoreUNK-B_C100.0txt"
handleClassLabels.labelsToFile(predictedLabels, writeToFile);


print "SUCCESS!";
