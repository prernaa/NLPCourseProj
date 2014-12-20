#Code for extracting unigram "types" from training data
# @Author - Vidur Puliani

import string
import getPOStags
from collections import OrderedDict

def extractTypes (filename, startColIdx):
    types = OrderedDict({}); # declaring an empty dictionary of types
    # To collect statistics for the training data, we also count the number of times each word occurs

    f = open(filename, 'r');
    
    for line in f:
        
        tags = getPOStags.getPOStags(line.split()[3:]);
		
        for key in tags:
            
            if key[0] not in string.punctuation: # for tokens that are not punctuation
                word = key[0] + "_" + key[1]; # token + POS tag
                
                if word in types:
                    types[word]+=1;
                else:
                    types[word]=1;
			
		
        
       
    f.close();
    return types;
