# Code for extracting feature vector X from training data
# @Author - Prerna Chikersal

import string
import getPOStags

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
	
        tags = getPOStags.getPOStags(line.split()[3:]);
        tokens=[];
        i=0;
        for key in tags:
            
            if key[0] in string.punctuation:
                del tags[i];
            else:
                tokens.append(key[0] + '_' + key[1]);

            i = i+1;
		#tokens = [x for x in tokens if x not in string.punctuation];

        
        # code to pre-process tokens ends


        # code to extract sampleX starts
        for key in typesDict:
            keyCount = tokens.count(key);
            sampleX.append(float(keyCount));
        # code to extract sampleX ends
        
        X.append(sampleX);
        lineCount = lineCount + 1;

    f.close();
    return X;
