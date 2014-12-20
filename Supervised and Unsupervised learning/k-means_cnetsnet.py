## None is assumed to be neutral in this approach
from sklearn.cluster import KMeans

polarityFilename = "./data/sncp_test_all.txt"
outputFilename = "./data/sncp_test_all_pol_kmeans.txt"

f = open(polarityFilename, 'r');

polArrSamples = []
for line in f:
    lst = line.split()
    ##print lst[0]
    if lst[0] == "None":
        ##print "assigning 0"
        pol = 0.0
    else:
        pol = float(lst[0])
    polArr = [pol]
    polArrSamples.append(polArr)

kmeans = KMeans(init='k-means++', n_clusters=3)
C = kmeans.fit(polArrSamples)

P = kmeans.predict(polArrSamples)

# 0 = positive, 1 = neutral, 2 = negative

for cls in P:
    with open(outputFilename, "a") as fw:
        if cls==0:
            classify = "positive"
        elif cls==1:
            classify = "neutral"
        else:
            classify = "negative"
        fw.write(str(classify))
        fw.write('\n')
    fw.close()

f.close()
    
