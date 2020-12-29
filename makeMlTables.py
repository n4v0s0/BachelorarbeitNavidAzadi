from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import os, pandas as pd,sys,time
from sklearn.model_selection import train_test_split
import subprocess
subprocess.call(["mkdir", "ML_Tabellen"])
for opt_flag in os.listdir("tfidfOutput"):
    with open(os.path.join("tfidfOutput",opt_flag), "r") as inp:
        top_5 = inp.readlines()
    top_5 = top_5[2:7]
    for count, g in enumerate(top_5):
        top_5[count] = g.split(":")[0]
    binaries = {}
    for root, dirs, files in os.walk("ngramOutput"):
        for dirc in dirs:
            path = os.path.join(root,dirc)
            for f in os.listdir(path):
                with open(os.path.join(path,f), "r") as f1:
                    grams = f1.readlines()
                top_dict = {}
                for top in top_5:
                    top = top.replace(',', '')	
                    for gram in grams:
                        if top in gram:
                            gram = int(gram.split(':')[1])
                            top_dict[top] = gram
                        if top not in top_dict:
                            top_dict[top] = 0
                cur_bin = f+"_"+dirc
                binaries[cur_bin] = top_dict
    flag = opt_flag.split("_")[0]
    df = pd.DataFrame()		
    for i in binaries:
        values = []
        keys = []
        for top in binaries[i]:
            values.append(str(binaries[i][top]))
            keys.append(top)
        values.append(i)
        keys.append("binary")
        if flag in i:
            values.append(1)
        else:
            values.append(0)
        keys.append("optimized")
        df2 = pd.DataFrame([values], columns=keys)
        df = pd.concat([df2,df])
    df.to_csv("ML_Tabellen/"+flag+"_Tabelle.txt", sep="\t", index=False)
