import sys
sys.path.append("/Users/Prerna/Desktop/Prerna/NTU/Courses-Year4-Sem1/NLP/SemEval15/code");


import extractTypes #importing file for extracting types from input
from collections import OrderedDict # OrderedDict to sort feature names in dictionary struct

#filename = "test_input.tsv";
#filename = "../rawdata/train/twitter-train-cleansed-B_rmnotav_new.tsv";
filename = "../rawdata/train/twitter-train-cleansed-B_rmnotav_ADDEDtest_new.tsv";
startColIdx = 3;

typesDict = extractTypes.extractTypes(filename, startColIdx);

#The line below does NOT work!
#OrderedDict(sorted(typesDict.items(), key=lambda t: t[0]));

print "Number of types extracted = ", len(typesDict);

#To print the types and their total number of occurrences... uncomment for speed
#for key in typesDict:
#    print key+" : ",typesDict[key]," ";

import extractFeatureVecX; #importing file for extracting feature vector X from training data
print "Feature Vector X Extraction Started";
Xfeatures = extractFeatureVecX.extractFeatureVecX(filename, startColIdx, typesDict);
print "Feature Vector X of size ", len(Xfeatures), " extracted";

import addPOStagsFeature;
#print "Adding NounNum to Xfeatures ", len(Xfeatures);
#Xfeatures = addPOStagsFeature.addNounNum(filename, Xfeatures);


#print "Adding VerbNum to Xfeatures ", len(Xfeatures);
#Xfeatures = addPOStagsFeature.addVerbNum(filename, Xfeatures#);

#print "Adding AdjAdvNum to Xfeatures ", len(Xfeatures);
#Xfeatures = addPOStagsFeature.addAdjAdvNum(filename, Xfeatures);

#print "Adding NounAdjRatio to Xfeatures ", len(Xfeatures);
#Xfeatures = addPOStagsFeature.addNounAdjRatio(filename, Xfeatures);

#print "Adding VerbAdvRatio to Xfeatures ", len(Xfeatures);
#Xfeatures = addPOStagsFeature.addVerbAdvRatio(filename, Xfeatures);

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
print "Declaring SVM"
clf = svm.LinearSVC(C=0.01, class_weight='auto', penalty='l1', dual=0); # linearsvc2
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
fileToSave = "UnigramSVMClassifier.joblib.pkl";
_ = joblib.dump(clf, fileToSave, compress=9);
print "Classifier SAVED!";



#### CAN BE INEFFICIENT! CAN MAKE PREDICTIONS LINE BY LINE, IF WE FACE ISSUES
# Extract feature vector from test data
testFilename = "../rawdata/test/AB_SemEval2013_task2_test_fixed/input/twitter-test-input-B_500.tsv"
#testFilename = "../rawdata/test/AB_SemEval2013_task2_test_fixed/input/sms-test-input-B.tsv"
testStartColIdx = 3
print "XTEST Feature Vector Extraction Started";
XTestFeatures = extractFeatureVecX.extractFeatureVecX(testFilename, testStartColIdx, typesDict);
print "XTEST Feature Vector of size ", len(XTestFeatures), " extracted";

#print "Adding NounNum to XTestfeatures ", len(XTestFeatures);
#XTestFeatures = addPOStagsFeature.addNounNum(filename, XTestFeatures);


#print "Adding VerbNum to XTestfeatures ", len(XTestFeatures);
#XTestFeatures = addPOStagsFeature.addVerbNum(filename, XTestFeatures);

#print "Adding AdjAdvNum to XTestfeatures ", len(XTestfeatures);
#XTestFeatures = addPOStagsFeature.addAdjAdvNum(filename, XTestFeatures);

#print "Adding NounAdjRatio to XTestfeatures ", len(XTestFeatures);
#XTestFeatures = addPOStagsFeature.addNounAdjRatio(filename, XTestFeatures);

#print "Adding VerbAdvRatio to XTestfeatures ", len(XTestFeatures);
#XTestFeatures = addPOStagsFeature.addVerbAdvRatio(filename, XTestFeatures);

print "standardising test data"
XTestFeatures = scaler.transform(XTestFeatures);
#Using Trained SVM to classify data
print "Predicting labels for test data"
predictedLabels = clf.predict(XTestFeatures);

#Writing predicted labels to file
writeToFile = "UnigramSVMTest_C0.01.txt"
handleClassLabels.labelsToFile(predictedLabels, writeToFile);


print "SUCCESS!";
