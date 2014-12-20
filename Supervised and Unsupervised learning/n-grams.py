from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
scaler = preprocessing.StandardScaler();
from sklearn import svm
from scipy.sparse import coo_matrix, hstack

def combineFeatures(f1,f2):
    f1 = coo_matrix(f1)
    f2 = coo_matrix(f2)
    return hstack([f1,f2]).toarray()

inputPath = "./data/twitter-train-cleansed-B_rmnotav.tsv"
inputIdx = 3
labelIdx = 2

testPath = "./data/twitter-dev-input-B_rmnotav.tsv"
testIdx = 3

outputPath = "./data/output_alltrain.txt"

f = open(inputPath, 'r');

corpus = []
labels = []

print "getting corpus"

##def getPOSfeatures(twlst):
##    return DictVectorizer()    

for line in f:
    lst = line.split()
    twtList = lst[inputIdx:]
    twt = " ".join(twtList)
    corpus.append(twt)
    labels.append(lst[labelIdx])
f.close

##print corpus
cv = CountVectorizer(analyzer='word', max_features=30000, ngram_range=(1,3))
cv2 = CountVectorizer(analyzer='char_wb', max_features=50000, ngram_range=(3,6))
print "extracting features from corpus"
sparsefeatureVec = cv.fit_transform(corpus).toarray()
charNgramsFeatures = cv2.fit_transform(corpus).toarray()

sparsefeatureVec = combineFeatures(sparsefeatureVec, charNgramsFeatures)

featureNamesWordNgrams = cv.get_feature_names();
featureNamesCharNgrams = cv2.get_feature_names();
print len(featureNamesWordNgrams)
print len(featureNamesCharNgrams)
print len(sparsefeatureVec[0])



##for w in cv.get_feature_names():
##    print w

print "declaring svm"
clf = svm.LinearSVC(C=0.01, class_weight='auto', penalty='l1', dual=0); # linearsvc2
print "scaling features"
sparsefeatureVec = scaler.fit_transform(sparsefeatureVec, labels);
print "training svm"
clf.fit(sparsefeatureVec, labels);

# Saving Trained Classifier
from sklearn.externals import joblib
print "Saving SVM"
fileToSave = "bestclassifier.joblib.pkl";
_ = joblib.dump(clf, fileToSave, compress=9);
print "Classifier SAVED!";


ft = open(testPath, 'r');
tcorpus = []
print "getting test data"
for line in ft:
    lst = line.split()
    twtList = lst[testIdx:]
    twt = " ".join(twtList)
    tcorpus.append(twt)
ft.close

print "extracting features from test data"
testfeatureVec = cv.transform(tcorpus).toarray()
testcharNgramsFeatures = cv2.transform(tcorpus).toarray()
testfeatureVec = combineFeatures(testfeatureVec, testcharNgramsFeatures)
print "scaling test features"
testfeatureVec = scaler.transform(testfeatureVec)
print "predicting labels for test data"
predicted = clf.predict(testfeatureVec)


fw = open(outputPath, 'w');
for l in predicted:
    fw.write(l)
    fw.write('\n')
fw.close()



