# Code for extracting trigram "types" from training data

import string
from collections import OrderedDict

def extractTrigrams (filename, startColIdx):
    Trigramtypes = OrderedDict({}); # declaring an empty dictionary of types
    # To collect statistics for the training data, we also count the number of times each word occurs

    f = open(filename, 'r');
    
    for line in f:
        tokens = line.split();
        # @TODO - Take each token from the tweet, strip the LHS and RHS of punctuations. Add to Dictionary.
        types=[];
        for tnum in range(startColIdx, len(tokens)):
            word = tokens[tnum].translate(string.maketrans("",""), string.punctuation);
            if word!="" and word!=" ":
                types.append(word);
        for i in range(0,len(types)-2):
            tgram=types[i]+" "+types[i+1]+" "+types[i+2];
            if tgram in Trigramtypes.keys():
                Trigramtypes[tgram]+=1;
            else:
                Trigramtypes[tgram]=1;
                
    f.close();
    return Trigramtypes;
