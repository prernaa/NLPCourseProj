# Code for extracting feature vector X from training data
# @Author - Prerna Chikersal

import string

#must return a 2D array
def extractFeatureVecX(filename, startColIdx, typesDict):
    X = []; # Feature Vector X (will be 2D)

    f = open(filename, 'r');
    lineCount = 1;
    for line in f:
        #Uncomment lineCount printer for speed
        #print lineCount;

        sampleX = [];

        # code to pre-process tokens begins
        tokens = line.split();
        words = [];
        for tnum in range(startColIdx, len(tokens)):
            word = tokens[tnum].translate(string.maketrans("",""), string.punctuation);
            if word!="" and word!=" ":
                words.append(word);
        # code to pre-process tokens begins

        # code to extract sampleX starts
        for key in typesDict:
            keyCount = words.count(key);
            sampleX.append(float(keyCount));
        # code to extract sampleX ends
        
        X.append(sampleX);
        lineCount = lineCount + 1;

    f.close();
    return X;
