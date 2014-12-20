from one_tweet_pipeline import unsupPolarityOfTweet
filename = "./data/sncp_test3.txt";
f = open(filename, 'r');
writeFile = "./data/sncp_test3_predict.txt"
#fw=open(writeFile, 'w')

for line in f:
        lst = line.split();
        lst=lst[3:len(lst)]
        #flst=[]
        #for i in range (3,len(lst)):
              #flst.append(lst[i]) 
        tweet=' '.join(lst)
        polarity=unsupPolarityOfTweet(tweet)
        print polarity
        #Writing predicted labels to file
        with open(writeFile, "a") as fw:
            fw.write(str(polarity))
            fw.write('\n')
##        fw.write('\n')
        fw.close() 
f.close()
       
        #print tweet
##lst=['apple' 'orabge' 'dog']
##lst=lst[2:0]
##t=' '.join(lst)
##print t
