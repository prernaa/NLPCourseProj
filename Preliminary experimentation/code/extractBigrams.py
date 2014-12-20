import string
from collections import OrderedDict

def find_bigrams(input_list):
  bigram_list = []
  for i in range(len(input_list)-1):
    bigram_word1 = input_list[i].translate(string.maketrans("",""), string.punctuation);
    bigram_word2 = input_list[i+1].translate(string.maketrans("",""), string.punctuation);
    if(bigram_word1 != "" and bigram_word1 != " " and bigram_word2 != "" and bigram_word2 != " "):
      bigram_list.append((bigram_word1, bigram_word2))
  return bigram_list

def extractBigramTypes (filename, startColIdx):
    types = OrderedDict({}); # declaring an empty dictionary of types
    # To collect statistics for the training data, we also count the number of times each word occurs

    f = open(filename, 'r');
    
    for line in f:
        itokens = line.split();
        tokens = find_bigrams(itokens);
        #print tokens;
        
        # @TODO - Take each token from the tweet, strip the LHS and RHS of punctuations. Add to Dictionary.
        for tnum in range(startColIdx, len(tokens)):
            if tokens[tnum]!=("","") and tokens[tnum]!=(" "," "):
                if tokens[tnum] in types:
                    types[tokens[tnum]]+=1;
                else:
                    types[tokens[tnum]]=1;

    
    f.close();
    #print types;
    return types;







