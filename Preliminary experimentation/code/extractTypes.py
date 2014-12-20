# Code for extracting unigram "types" from training data
# @Author - Prerna Chikersal

import string
from collections import OrderedDict

def extractTypes (filename, startColIdx):
    types = OrderedDict({}); # declaring an empty dictionary of types
    # To collect statistics for the training data, we also count the number of times each word occurs

    f = open(filename, 'r');
    
    for line in f:
        tokens = line.split();
        # @TODO - Take each token from the tweet, strip the LHS and RHS of punctuations. Add to Dictionary.
        for tnum in range(startColIdx, len(tokens)):
            word = tokens[tnum].translate(string.maketrans("",""), string.punctuation);
            if word!="" and word!=" ":
                if word in types:
                    types[word]+=1;
                else:
                    types[word]=1;

    
    f.close();
    return types;
