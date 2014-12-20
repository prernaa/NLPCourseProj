
import nltk;

def getPOStags(twt):
        tags = {};
        twt = ' '.join(twt);        
        twt_tokens = nltk.word_tokenize(twt);        
        tags = nltk.pos_tag(twt_tokens);        
        return tags;
	
	
	
	

