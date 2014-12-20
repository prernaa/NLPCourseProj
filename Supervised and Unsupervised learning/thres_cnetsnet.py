polarityFilename = "./data/sncp_test_all.txt"
outputFilename = "./data/sncp_test_all_pol_thres.txt"

f = open(polarityFilename, 'r');

for line in f:
    lst = line.split()
    if lst[0] == "None":
        classify = "neutral"
    elif float(lst[0])==0.0 :
        classify = "neutral"
    elif float(lst[0])>0.0 :
        classify = "positive"
    else:
        classify = "negative"
    with open(outputFilename, "a") as fw:
        fw.write(str(classify))
        fw.write('\n')
    fw.close()

f.close()
