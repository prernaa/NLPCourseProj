# Code for evaluation of prediction using gold and predicted files.
# @AUTHOR - Prerna Chikersal

import string

class Evaluator:
    def __init__(self):
        #print "constructor called";
        self.TP_pos = 0.0;
        self.TP_neg = 0.0;
        self.TP_neu = 0.0;
        self.FP_pos = 0.0;
        self.FP_neg = 0.0;
        self.FP_neu = 0.0;
        self.FN_pos = 0.0;
        self.FN_neg = 0.0;
        self.FN_neu = 0.0;
        self.lineCount = 0.0;
    def getPrecision_pos(self):
        return (self.TP_pos/(self.TP_pos+self.FP_pos));
        
    def getPrecision_neg(self):
        return (self.TP_neg/(self.TP_neg+self.FP_neg));
        
    def getPrecision_neu(self):
        return (self.TP_neu/(self.TP_neu+self.FP_neu));

    def getRecall_pos(self):
        return (self.TP_pos/(self.TP_pos+self.FN_pos));
        
    def getRecall_neg(self):
        return (self.TP_neg/(self.TP_neg+self.FN_neg));
        
    def getRecall_neu(self):
        return (self.TP_neu/(self.TP_neu+self.FN_neu));

    def getAvgPrecision(self):
        return ((self.getPrecision_pos()+self.getPrecision_neg()+self.getPrecision_neu())/3.0);

    def getAvgRecall(self):
        return ((self.getRecall_pos()+self.getRecall_neg()+self.getRecall_neu())/3.0);

    def getFmeasure_pos(self):
        return ((2*self.getPrecision_pos()*self.getRecall_pos())/(self.getPrecision_pos()+self.getRecall_pos()));

    def getFmeasure_neg(self):
        return ((2*self.getPrecision_neg()*self.getRecall_neg())/(self.getPrecision_neg()+self.getRecall_neg()));

    def getFmeasure_neu(self):
        return ((2*self.getPrecision_neu()*self.getRecall_neu())/(self.getPrecision_neu()+self.getRecall_neu()));

    def getAvgFmeasure(self):
        return ((self.getFmeasure_pos()+self.getFmeasure_neg()+self.getFmeasure_neu())/3.0);

    def getAvgPosNegFmeasure(self):
        return ((self.getFmeasure_pos()+self.getFmeasure_neg())/2.0);

    def getPercentCorrect(self):
        return ((self.TP_pos+self.TP_neg+self.TP_neu)/self.lineCount)*100;
    
    def evalPrediction(self, predictedPath, predictedIdx, goldPath, goldIdx):
        with open(predictedPath) as fp, open(goldPath) as fg:
            while True:
                try:
                    lineP = next(fp);
                    lineG = next(fg);
                    self.lineCount = self.lineCount+1;
                    tokensP = lineP.split();
                    tokensG = lineG.split();
                    labelP = tokensP[predictedIdx];
                    labelG = tokensG[goldIdx];
                    if labelP == "positive" and labelG == labelP:
                        self.TP_pos=self.TP_pos+1;
                    if labelP == "negative" and labelG == labelP:
                        self.TP_neg=self.TP_neg+1;
                    if labelP == "neutral" and labelG == labelP:
                        self.TP_neu=self.TP_neu+1;
                        
                    if labelP == "positive" and labelG != labelP:
                        self.FP_pos=self.FP_pos+1;
                    if labelP == "negative" and labelG != labelP:
                        self.FP_neg=self.FP_neg+1;
                    if labelP == "neutral" and labelG != labelP:
                        self.FP_neu=self.FP_neu+1;

                    if labelG == "positive" and labelG != labelP:
                        self.FN_pos=self.FN_pos+1;
                    if labelG == "negative" and labelG != labelP:
                        self.FN_neg=self.FN_neg+1;
                    if labelG == "neutral" and labelG != labelP:
                        self.FN_neu=self.FN_neu+1;
                except StopIteration:
                    break
        print "Number of lines evaluated = ", self.lineCount;
        
