import sys
sys.path.append("/Users/Prerna/Desktop/Prerna/NTU/Courses-Year4-Sem1/NLP/SemEval15/code");


import extractTypes #importing file for extracting types from input
from collections import OrderedDict # OrderedDict to sort feature names in dictionary struct

filename = "test_input.tsv";
startColIdx = 3;

typesDict = extractTypes.extractTypes(filename, startColIdx);

OrderedDict(sorted(typesDict.items(), key=lambda t: t[0]));
            
for key in typesDict:
    print key+" : ",typesDict[key];
