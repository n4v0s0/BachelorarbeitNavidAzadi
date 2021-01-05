from sklearn import metrics
import os, pandas as pd,sys,time
from sklearn.model_selection import train_test_split,KFold,cross_val_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
print("Vergleiche ML-Klassifizierer...\n")
dirc = "ML_Tabellen"
counter = 0
scoreDict = {}
for opt in os.listdir(dirc):
    print("Berechne ",counter,"/",len(os.listdir(dirc))
    counter += 1
    path = os.path.join(dirc,opt)
    df = pd.read_csv(path,sep="\t")	
    features_new =  df.loc[df['optimized']  == 0]
    tenPercent = len(features_new)//10
    features_new = features_new.iloc[:tenPercent]
    target_new = df.loc[df['optimized']  == 1]
    df_new = features_new.append(target_new)
    features = df_new[df.columns[:5]].values
    target = df_new["optimized"].values

    tree = DecisionTreeClassifier()
    nb = GaussianNB()
    rf = RandomForestClassifier()
    tree_score = cross_val_score(tree, features, target)
    nb_score = cross_val_score(nb, features, target)
    rf_score = cross_val_score(rf, features, target)
    opt = opt.split("_")[0]
    scoreDict[opt] = {"tree": sum(tree_score)/len(tree_score), "rf": sum(rf_score)/len(rf_score), "nb": sum(nb_score)/len(nb_score)}

treeVals = []
nbVals = []
rfVals = [] 
for i in scoreDict:
    treeVals.append(scoreDict[i]["tree"])
    rfVals.append(scoreDict[i]["rf"])
    nbVals.append(scoreDict[i]["nb"])
plt.plot(treeVals, "g", label="Decision Tree")
plt.plot(rfVals, "b", label="Random Forest")
plt.plot(nbVals, "r", label="Naive Bayes")
plt.legend()
plt.show()
