import sys
sys.path.append("/Users/Prerna/Desktop/Prerna/NTU/Courses-Year4-Sem1/NLP/SemEval15/code");

import extractBigrams #importing file for extracting bigram types from input
import extractTypes    #importing file for extracting unigram types from input
from collections import OrderedDict # OrderedDict to sort feature names in dictionary struct

#filename = "addedTest.tsv";
filename = "../rawdata/train/twitter-train-cleansed-B_rmnotav_ADDEDtest_new.tsv";
startColIdx = 3;

typesDictBigrams = extractBigrams.extractBigramTypes(filename, startColIdx);
typesDictUnigrams = extractTypes.extractTypes(filename, startColIdx);

print "Number of bigram types extracted = ", len(typesDictBigrams);
print "Number of unigram types extracted = ", len(typesDictUnigrams);

import extractBigramFeatureVecX; #importing file for extracting bigram feature vector X from training data
import extractFeatureVecX; #importing file for extracting unigram feature vector from training data
print "Feature Vector Extraction Started";

XFeaturesUnigrams = extractFeatureVecX.extractFeatureVecX(filename, startColIdx, typesDictUnigrams);
XFeaturesBigrams = extractBigramFeatureVecX.extractBigramFeatureVecX(filename, startColIdx, typesDictBigrams);

#combining the two feature vectors below
XFeatures=[];
Ndata=len(XFeaturesUnigrams)#number of sentences in training data 
lUTypes=len(XFeaturesUnigrams[0]); #number of unigram types
lBTypes=len(XFeaturesBigrams[0]); #number of bigram types

for i in range (0,Ndata):
    XFeatures.append(XFeaturesUnigrams[i]);
    for j in range (0,lBTypes):
        XFeatures[i].append(XFeaturesBigrams[i][j]); #combining of feature vectors finished

#print "Feature Vector of size ", len(Xfeatures), " extracted";

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
clf = svm.LinearSVC(C=1000.0, class_weight='auto', penalty='l1', dual=0); # linearsvc2
#clf = svm.SVC(cache_size = 1000, class_weight='auto', kernel = 'poly'); # Predicts all as POSITIVE :((
#clf = linear_model.SGDClassifier();  # not tried yet
print "standardising training data"
XFeatures = scaler.fit_transform(XFeatures, YLabels);
print "Fitting Data To SVM"
clf.fit(XFeatures, YLabels);
print "SVM trained"


# Saving Trained Classifier
from sklearn.externals import joblib
print "Saving SVM"
fileToSave = "UnigramBigramSVMClassifier.joblib.pkl";
_ = joblib.dump(clf, fileToSave, compress=9);
print "Classifier SAVED!";



#### CAN BE INEFFICIENT! CAN MAKE PREDICTIONS LINE BY LINE, IF WE FACE ISSUES
# Extract feature vector from test data
testFilename = "../rawdata/test/AB_SemEval2013_task2_test_fixed/input/twitter-test-input-B_500.tsv"
#testFilename = "../rawdata/test/AB_SemEval2013_task2_test_fixed/input/sms-test-input-B.tsv"
testStartColIdx = 3

print "XTEST Feature Vector Extraction Started";
XTestUnigramFeatures = extractFeatureVecX.extractFeatureVecX(testFilename, testStartColIdx, typesDictUnigrams);
XTestBigramFeatures = extractFeatureVecX.extractFeatureVecX(testFilename, testStartColIdx, typesDictBigrams);

#combining the two feature vectors below
XTestFeatures=[]
Ndata=len(XTestUnigramFeatures)#number of sentences in training data 
lUTypes=len(XTestUnigramFeatures[0]); #number of unigram types
lBTypes=len(XTestBigramFeatures[0]); #number of unigram types

for i in range (0,Ndata):
    XTestFeatures.append(XTestUnigramFeatures[i]);
    for j in range (0,lBTypes):
        XTestFeatures[i].append(XTestBigramFeatures[i][j]);#combining of feature vectors finished

#print "XTEST Feature Vector of size ", len(XTestFeatures), " extracted";
#Using Trained SVM to classify data
predictedLabels = clf.predict(XTestFeatures);

#Writing predicted labels to file
writeToFile = "UnigramBigramSVMTest_C1000.0.txt"
handleClassLabels.labelsToFile(predictedLabels, writeToFile);


print "SUCCESS!";
