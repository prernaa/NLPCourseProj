#TOP LEVEL EVALUATE FILE
import sys
sys.path.append("/Users/Prerna/Desktop/Prerna/NTU/Courses-Year4-Sem1/NLP/SemEval15/code");

import evaluate
E = evaluate.Evaluator();
pPath = "UnigramPosSVM_C500.0.txt";
#gPath = "../rawdata/test/gold/twitter-dev-gold-B_rmnotav.tsv";
gPath = "../rawdata/test/AB_SemEval2013_task2_test_fixed/gold/twitter-test-gold-B_500.tsv";
PIdx = 0;
gIdx = 2;
E.evalPrediction(pPath, PIdx, gPath, gIdx);

print "Average Pos & Neg F-measure = ", E.getAvgPosNegFmeasure();
print "Average F-measure = ", E.getAvgFmeasure();
print "Average Precision = ", E.getAvgPrecision();
print "Average Recall = ", E.getAvgRecall();

print "Positive F-measure = ", E.getFmeasure_pos();
print "Negative F-measure = ", E.getFmeasure_neg();
print "Neutral F-measure = ", E.getFmeasure_neu();

print "Positive Precision = ", E.getPrecision_pos();
print "Negative Precision = ", E.getPrecision_neg();
print "Neutral Precision = ", E.getPrecision_neu();

print "Positive Recall = ", E.getRecall_pos();
print "Negative Recall = ", E.getRecall_neg();
print "Neutral Recall = ", E.getRecall_neu();


print "Percentage Correct = ", E.getPercentCorrect();
