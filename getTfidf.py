import nltk,os,time,pandas as pd, subprocess

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

def computeIDF(files):
    import math
    N = len(files)
    idfDict = dict.fromkeys(files[0].keys(), 0)
    for f in files:
        for word, val in f.items():
            if val > 0:
                idfDict[word] += 1
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val)) #+ 1
    return idfDict

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf
print("Berechne TF-IDF-Werte...\n")
flags = {}
for flag in os.listdir("ngramOutput"):
    for binary in os.listdir("ngramOutput/"+flag):
        with open(os.path.join("ngramOutput",flag,binary), "r") as f:
            lines = f.readlines()
            for line in lines:
                if "})" in line:
                    line = line.replace("})","")
                line = line.strip()
                ngram, cnt = line.split(":")
                if flag in flags:
                    if ngram in flags[flag]:
                        flags[flag][ngram] += int(cnt)
                    else:
                        flags[flag][ngram] = int(cnt)
                else:
                    flags[flag] = {}
                    flags[flag][ngram] = int(cnt)
all_grams = []
for i in flags:
	for j in flags[i]:
		all_grams.append(j)
unique_grams = set(all_grams)
print(str(len(unique_grams))+" wurden geschrieben")
numOfWords = {}
for flag in flags:
	numOfWords[flag] = dict.fromkeys(unique_grams,0)
	for ngram in flags[flag]:
		 numOfWords[flag][ngram] = flags[flag][ngram]
tf_dict = {}
length = len(flags)
i = 0
for flag in flags:
	tf_dict[flag] = computeTF(numOfWords[flag],flags[flag])
	i += 1
	print("TF-Value berechnet für "+str(i)+"/"+str(length)+"\n")
tf_dicts = []
for flag in tf_dict:
	tf_dicts.append(tf_dict[flag])
idfs = computeIDF(tf_dicts)
print("\n\n IDFS wurden berechnet \n\n")
result_dict = {}
result = []

i = 0
subprocess.Popen(["mkdir", "tfidfOutput"])
for flag in flags:
	result_dict[flag] = computeTFIDF(tf_dict[flag], idfs)
	result.append(result_dict[flag])
	subprocess.Popen(["touch", "tfidfOutput/"+flag+"_tfidf.txt"])
	with open("tfidfOutput/"+flag+"_tfidf.txt", "w") as output:
		output.write("TF_IDF Werte für flag: "+flag+"\n\n")
		for key,value in sorted(result_dict[flag].items(), key=lambda item: item[1], reverse=True):
			output.write("%s: %s \n" % (key,value))
	i += 1
	print("TF_IDF-Value berechnet für "+str(i)+"/"+str(length)+"\n")

