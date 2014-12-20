# Code for extracting class labels Y from training data
# @Author - Prerna Chikersal

import string

# neutral = 0
# positive = 1
# negative = 2

def extractClassLabels(filename, labelIdx):
    Y=[];
    f = open(filename, 'r');
    for line in f:
        tokens = line.split();
        label = tokens[labelIdx];
        if label == "neutral":
            Y.append(0);
        elif label == "positive":
            Y.append(1);
        elif label == "negative":
            Y.append(2);
        else: print "ERROR: UNKNOWN LABEL WHILE READING TRAINING DATA!"
        
    f.close();
    return Y;

def labelsToFile(labels, filename):
    fw = open(filename,'w')
    for label in labels:
        if label == 0:
            fw.write('neutral\n')
        elif label == 1:
            fw.write('positive\n')
        elif label == 2:
            fw.write('negative\n')
        else: print "ERROR: UNKNOWN LABEL WHILE WRITING PREDICTIONS!"
    fw.close();
