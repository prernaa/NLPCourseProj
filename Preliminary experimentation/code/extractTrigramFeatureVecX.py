# Code for extracting feature vector X (unigrams and trigrams combined) from training data

import string

#must return a 2D array
def extractTrigramFeatureVecX(filename, startColIdx, typesDict):
    X = []; # Feature Vector X (will be 2D)

    f = open(filename, 'r');
    lineCount = 1;
    for line in f:
        #Uncomment lineCount printer for speed
        #print lineCount;

        sampleX = [];

        # code to pre-process tokens begins
        tokens = line.split();
        trigrams = [];
        types=[];
        for tnum in range(startColIdx, len(tokens)):
            word = tokens[tnum].translate(string.maketrans("",""), string.punctuation);
            if word!="" and word!=" ":
                types.append(word);
        for i in range(0,len(types)-2):
            trigrams.append(types[i]+" "+types[i+1]+" "+types[i+2]);

        # code to extract sampleX starts
        for key in typesDict:
            keyCount = trigrams.count(key);
            sampleX.append(float(keyCount));
        # code to extract sampleX ends
        
        X.append(sampleX);
        lineCount = lineCount + 1;

    f.close();
    return X;
