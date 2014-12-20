import string

f = open('nlp_proj_test_gold.txt', 'r');

sumAvg=0.0
sumNeu=0.0
sumP=0.0
sumN=0.0

for line in f:
    tokens = line.split();
    if tokens[2]=='neutral':
        sumNeu+=1
    elif tokens[2]=='positive':
        sumP+=1
    else:
        sumN+=1
    # @TODO - Take each token from the tweet, strip the LHS and RHS of punctuations. Add to Dictionary.
##    for tnum in range(3, len(tokens)):
##        word = tokens[tnum].translate(string.maketrans("",""), string.punctuation);
##        if word!="" and word!=" ":
##            sumAvg+=1

print ('ratio of opinionated to neutral tweets')
print (sumP+sumN)/sumNeu
print ('ratio of positive to negative tweets')
print (sumP/sumN)
##print ('avg wrds per tweet')
##print (sumAvg/1000.0)

f.close();

