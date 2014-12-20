# Code for extracting bigram feature vector X from training data
# @Author - Sriporna Mukherjee

import string

def find_bigrams(input_list):
  bigram_list = []
  for i in range(len(input_list)-1):
    bigram_word1 = input_list[i].translate(string.maketrans("",""), string.punctuation);
    bigram_word2 = input_list[i+1].translate(string.maketrans("",""), string.punctuation);
    if(bigram_word1 != "" and bigram_word1 != " " and bigram_word2 != "" and bigram_word2 != " "):
      bigram_list.append((bigram_word1, bigram_word2))
  return bigram_list

#must return a 2D array
def extractBigramFeatureVecX(filename, startColIdx, typesDict):
    X = []; # Feature Vector X (will be 2D)

    f = open(filename, 'r');
    lineCount = 1.0;
    for line in f:
        itokens = line.split();
        tokens = find_bigrams(itokens);

        sampleX = [];

        # code to pre-process tokens begins
        words = [];
        for tnum in range(startColIdx, len(tokens)):    
            if tokens[tnum]!=("","") and tokens[tnum]!=(" "," "):
                words.append(tokens[tnum]);
        # code to pre-process tokens ends

        # code to extract sampleX starts
        for key in typesDict:
            keyCount = words.count(key);
            sampleX.append(float(keyCount));
        # code to extract sampleX ends
        
        X.append(sampleX);
        lineCount = lineCount + 1.0;

    f.close();
    return X;
