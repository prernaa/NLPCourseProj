import sys
##sys.path.append("/Users/Prerna/Desktop/Prerna/NTU/Courses-Year4-Sem1/NLP/SemEval15/external/cp_python_wrapper");

##import concept_parser_pywrapper;

from nltk.tag.stanford import POSTagger
stpos = POSTagger('/usr/share/stanford-postagger-3.4-eng/models/english-left3words-distsim.tagger', '/usr/share/stanford-postagger-3.4-eng/stanford-postagger.jar')
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.stem.snowball import EnglishStemmer
snowstem = EnglishStemmer()
#import os
#os.environ['STANFORD_PARSER'] = '/usr/share/stanford-parser-3.4'
#os.environ['STANFORD_MODELS'] = '/usr/share/stanford-parser-3.4'
#from nltk.parse.stanford import StanfordParser
#parser = StanfordParser('/usr/share/stanford-parser-3.4/stanford-parser-3.4-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
import json
import urllib2
import unicodedata
import string
import unicodedata
import sys
tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
                      if unicodedata.category(unichr(i)).startswith('P'))
def remove_punctuation(text):
    return text.translate(tbl)
negInv = ["never", "not", "but"]
def tryInvert(val, t):
    if (t in negInv) and val==1:
        return 0
    elif (t in negInv) and val==0:
        return 1
    else:
        return val

cpExclude = ["never", "not", "but", "most", "such", "too", "is", "had", "that", "thats", "me"]

#SenticNet integration
from SenticNetQuery import SenticNetQuery 



def getTermsFromTags(tagArr):
    terms = []
    adjTags = ["JJ", "JJR", "JJS"]
    nounTags = ["NN", "NNS", "NNP", "NNPS"]
    adverbTags = ["RB", "RBR", "RBS"]
    verbTags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    pairTags = [("JJ", "NN"), ("NN", "VB"), ("VB", "NN"), ("NN", "NN"), ("RB", "NN"), ("RB", "JJ")]
    invert = 0
    # parsedSentence = parser.tagged_parse(tagArr);
    # print parsedSentence
    # Use stanford parser to get negated terms
    # consider only JJ, JJR, JJS, NN, NNS, NNP, NNPS, RB, RBR, RBS, VB, VBD, VBG, VBN, VBP, VBZ

    ##print tagArr

    #Add single terms
    for i in range(0, len(tagArr)):
        el = tagArr[i]
        t = remove_punctuation(el[0])
        t=t.lower()
        invert = tryInvert(invert, t)
        ##print t
        if t!="" and ((el[1] in adjTags) or (el[1] in nounTags) or (el[1] in adverbTags) or (el[1] in verbTags)):
            terms.append(tuple([t,invert]))
            tstem=snowstem.stem(t)
            if t!=tstem:
                terms.append(tuple([tstem,invert]))
                
    invert = 0
    #Add pair or words/ phrases
    #Take <JJ* NN*>, <NN* VB*>, <VB* NN*>, <NN* NN*>, <RB* NN*>
    for i in range(0, len(tagArr)-1):
        el1 = tagArr[i]
        el2 = tagArr[i+1]
        t1 = unicodedata.normalize('NFKD', el1[1]).encode('ascii','ignore')
        t2 = unicodedata.normalize('NFKD', el2[1]).encode('ascii','ignore')
        w1 = el1[0]
        w2 = el2[0]
        w1 = w1.lower()
        w2 = w2.lower()
        w1 = remove_punctuation(w1)
        w2 = remove_punctuation(w2)
        w1stem = snowstem.stem(w1)
        w2stem = snowstem.stem(w2)
        t1=t1[:2]
        t2=t2[:2]
        tagpairL = []
        tagpairL.append(t1)
        tagpairL.append(t2)
        tagpairT = tuple(tagpairL)
        invert = tryInvert(invert, w1)
        ##print tagpairT
        if tagpairT in pairTags:
            terms.append(tuple([w1+" "+w2, invert]))
            if w1!=w1stem and w2!=w2stem:
                terms.append(tuple([w1stem+" "+w2stem, invert]));
            if w1!=w1stem and w2==w2stem:
                terms.append(tuple([w1stem+" "+w2, invert]));
            if w1==w1stem and w2!=w2stem:
                terms.append(tuple([w1+" "+w2stem, invert]));
        
    return terms

def cleanNodeText (txt):
    splitSlash = txt.split("/")
    txt = splitSlash[0]
    forSpace = txt.split("_")
    txt = " ".join(forSpace)
    return txt

def expandUsingConceptNet(sterms):
    # expand using concept net
    urlprefix = "http://conceptnet5.media.mit.edu/data/5.2/search?minWeight=1&text="
    #relations = ["&rel=/r/RelatedTo", "&rel=/r/IsA", "&rel=/r/PartOf", "&rel=/r/HasA", "&rel=/r/HasProperty", "&rel=/r/Synonym"]
    #relations = ["&rel=/r/IsA","&rel=/r/Synonym"]
    relations = ["&rel=/r/IsA"]
    expTerms = []
    for i in range (0, len(sterms)):
        t = sterms[i]
        expandedT = []
        if t[0] in cpExclude:
            continue
        for r in relations :
            txt = t[0].encode('ascii', 'ignore')
            txt = txt.split()
            txt = "_".join(txt)
            url = urlprefix+txt+r
            ##print t
            ##print url
            response = urllib2.urlopen(url)
            data = json.load(response)
            numFound = data["numFound"]
            if numFound > 0:
                #print data["numFound"]
                r1 = data["edges"][0]
                end1 = r1["end"]
                start1 = r1["start"]
                end1 = end1[6:]
                start1 = start1[6:]
                start1 = cleanNodeText(start1)
                end1 = cleanNodeText(end1)
                if end1 == txt:
                    expandedT.append(start1)
                    ##print start1
                elif start1 == txt:
                    expandedT.append(end1)
                    ##print end1
                if numFound >1:
                    r2 = data["edges"][1]
                    end2 = r2["end"]
                    start2 = r2["start"]
                    end2 = end2[6:]
                    start2 = start2[6:]
                    start2 = cleanNodeText(start2)
                    end2 = cleanNodeText(end2)
                    if end2 == txt:
                        expandedT.append(start2)
                        ##print start2
                    elif start2 == txt:
                        expandedT.append(end2)
                        ##print end2
        
        #add expanded terms to sterms
        expandedT=list(set(expandedT))
        myTerm = tuple([sterms[i][0], sterms[i][1], expandedT])
        expTerms.append(myTerm)
    return expTerms


# For each tweet,
def getTermsForSenticNet(twt):
    sentences = tokenizer.tokenize(twt)
    twtTerms = []
    for s in sentences:
        sentenceTokens = s.split()
        sentenceTags = stpos.tag(sentenceTokens)
        sterms = getTermsFromTags(sentenceTags)
        sterms = expandUsingConceptNet(sterms)
        twtTerms.extend(sterms)
    return twtTerms


def getPolarityOfTerms(twtTerms):
    #get polarities and invert them wherever needed
    polarityPerTerm=[]
    for t in twtTerms:
        ##print t
        if t[1]==0:
            inv=1;
        else:
            inv=-1
        main = t[0]
        mainPolarity = SenticNetQuery(main)
        
        if mainPolarity is not None:
            ##print main
            ##print mainPolarity
            mainPolarity = mainPolarity*inv
            mainMultiplier = 2;
        else:
            mainMultiplier = 0;
        ##print "main polar"
        ##print mainPolarity
        count = 0
        sumbcPolarity=0;
        for bc in t[2]:
            bcPolarity = SenticNetQuery(bc)
            ##print bc
            ##print bcPolarity
            if bcPolarity is not None:
                ##print bc
                ##print bcPolarity
                bcPolarity = bcPolarity*inv
                count=count+1;
                sumbcPolarity=sumbcPolarity+bcPolarity
        ##print "vals"
        ##print mainMultiplier
        ##print mainPolarity
        ##print count
        ##print sumbcPolarity
        toApp = None
        if(mainMultiplier==0 and count==0):
            toApp=None
        elif(mainMultiplier==0 and count!=0):
            toApp=sumbcPolarity/count;
        elif(mainMultiplier!=0 and count == 0):
            toApp=mainPolarity
        else:
            toApp=(((mainPolarity*mainMultiplier)+(sumbcPolarity/count))/(mainMultiplier+1))
        polarityPerTerm.append(toApp)
        ##print "appended"
        ##print toApp
    return polarityPerTerm
    
            
def unsupPolarityOfTweet(twt):
    twtTerms = getTermsForSenticNet(twt)
    ##print twtTerms
    ##print len(twtTerms)
    polarity = getPolarityOfTerms(twtTerms)
    ##print polarity
    c=0
    sump=0
    for p in polarity:
        if p is not None:
            c=c+1
            sump=sump+p
    if c!=0:
        return (sump/c)
    else:
        return None

def getPolarityOfTweets_OLD(twtTerms):
    for t in twtTerms:
        inv = t[1]*-1
        main = t[0]
        mainPolarity = SenticNetQuery(main)
        count = 0;
        if mainPolarity is not None:
            mainPolarity = mainPolarity*inv
    
def unsupPolarityOfTweet_OLD(twt):
    twtTerms = getTermsForSenticNet(twt)
    print twtTerms
    polarity = getPolarityOfTerms(twtTerms)
    print polarity
        

##myTweet1 = "Who had two thumbs and is the happiest girl in the world because she might get Greys Anatomy season 4 tomorrow? Oh yeah, thats me! ;)"
##myTweet = "She is the most beautiful woman ever!"
##myTweet2 = "She is not good, but he is not bad"

##twtTerms = getTermsForSenticNet(myTweet)
##print twtTerms
##print getPolarity(twtTerms)

##print unsupPolarityOfTweet(myTweet)





    


## Fetching polarity from SenticNet
## Query SenticNet for each term and get the polarity score. Remove expanded terms with no polarity values

