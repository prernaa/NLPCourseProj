import nltk;
import getPOStags;
import numpy;

noun = ['NN', 'NNP', 'NNPS', 'NNS'];
verb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'];
adj = ['JJ', 'JJR', 'JJS'];
adv = ['RB', 'RBR', 'RBS'];

def addNounNum(filename, X):
        f = open(filename, 'r');
        linecount = 0
        
        for line in f:
                numNoun = 0;
		
                tokens = getPOStags.getPOStags(line.split()[3:]); #dict
                
                for tags in tokens:
                        if tags[1] in noun:
                                numNoun = numNoun + 1;
                                
                numNoun = [numNoun];               
                #tokens = [x for x in tokens if x not in string.punctuation];#
                X[linecount] = X[linecount] + numNoun;
                print linecount;
                linecount = linecount + 1;
        return X;

def addVerbNum(filename, X):
        f = open(filename, 'r');
        linecount = 0
        
        for line in f:
                numVerb = 0;
				
                tokens = getPOStags.getPOStags(line.split()[3:]); #dict
                
                for tags in tokens:
                        if tags[1] in verb:
                                numVerb = numVerb + 1;
                                
                #tokens = [x for x in tokens if x not in string.punctuation];#
                
                numVerb = [numVerb];

                X[linecount] = X[linecount] + numVerb;
                linecount = linecount + 1;

        return X;

        
                
        return;

def addAdjAdvNum(filename, X):
        f = open(filename, 'r');
        linecount = 0
        
        for line in f:
                numAdjAdv = 0;
				
                tokens = getPOStags.getPOStags(line.split()[3:]); #dict
                
                for tags in tokens:
                        if tags[1] in adj or adv:
                                numAdjAdv = numAdjAdv + 1;
                                
                #tokens = [x for x in tokens if x not in string.punctuation];#
                numAdjAdv = [numAdjAdv];
                X[linecount] = X[linecount] + numAdjAdv;
                linecount = linecount + 1;
        return X;
                
        
def addNounAdjRatio(filename, X):
        f = open(filename, 'r');
        linecount = 0
        
        for line in f:
                numNoun = 0;
                numAdj = 0;
				
                tokens = getPOStags.getPOStags(line.split()[3:]); #dict
                
                for tags in tokens:
                        if tags[1] in noun:
                                numNoun = numNoun + 1;

                for tags in tokens:
                        if tags[1] in adj:
                                numAdj = numAdj + 1;

                ratioNounAdj = numNoun/(numAdj+1);

                ratioNounAdj = [ratioNounAdj];
                #tokens = [x for x in tokens if x not in string.punctuation];#
                X[linecount] = X[linecount] + ratioNounAdj;
                linecount = linecount + 1;
        return X;

def addVerbAdvRatio(filename, X):
        f = open(filename, 'r');
        linecount = 0
        
        for line in f:
                numVerb = 0;
                numAdv = 0;
			
                tokens = getPOStags.getPOStags(line.split()[3:]); #dict
                
                for tags in tokens:
                        if tags[1] in verb:
                                numVerb = numVerb + 1;

                for tags in tokens:
                        if tags[1] in adv:
                                numAdv = numAdv + 1;

                ratioVerbAdv = numVerb/(numAdv+1);                                
                #tokens = [x for x in tokens if x not in string.punctuation];#
                ratioVerbAdv = [ratioVerbAdv];
                X[linecount] = X[linecount] + ratioVerbAdv;
                linecount = linecount + 1;

        return X;

def addExclNum():

        return;
