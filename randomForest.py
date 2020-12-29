from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import numpy as np
import os, pandas as pd,sys,time
from sklearn.model_selection import train_test_split,KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
import subprocess
def plot_confusion_matrix(cm,
                          opt,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True
                          ):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Tatsächliches Label')
    plt.xlabel('Berechnetes Label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig("ML-Diagramme/konfusionsMatrizen/"+opt+"_confusionMatrix.png", dpi = 200)
    plt.clf()
def printDict(inp, write = False, fout = None):
    out = sorted(inp.items(), key=lambda x: x[1], reverse=True)
    if write == True:
        with open(fout,"w") as f:
            for i in out:
                f.write(str(i[0])+" "+str(i[1])+"\n")
    else:
        for i in out:
            print(i,inp[i])
def printMat(inp, write = False, fout = None):
    if write == True:
        with open(fout, "w") as f:
            for i in inp:
                f.write(str(i) + "\n")
                f.write(str(inp[i]))
    else:
        for i in inp:
            print(i,"\n")
            print(inp[i])

def barLists(inp):
    cnt = {}
    for i in inp:
        r = round(float(inp[i]), 3)
        if r in cnt:
            cnt[r] += 1
        else:
            cnt[r] = 1
    return list(cnt.keys()),list(cnt.values());
def matAdd(m1, m2):
    ret = m1
    ret[0][0] += m2[0][0]
    ret[0][1] += m2[0][1]
    ret[1][0] += m2[1][0]
    ret[1][1] += m2[1][1]
    return ret
dirc = "ML_Tabellen"
recallDict = dict.fromkeys(os.listdir(dirc), 0)
precisionDict = dict.fromkeys(os.listdir(dirc), 0)
f1Dict = dict.fromkeys(os.listdir(dirc), 0)
accuracyDict = dict.fromkeys(os.listdir(dirc), 0)
confDict = dict.fromkeys(os.listdir(dirc), [[0,0],[0,0]])  
test_size_val = 0.33
for j in range(10):
    for opt in os.listdir(dirc):
        path = os.path.join(dirc, opt)
        df = pd.read_csv(path, sep="\t")
        features = df[df.columns[:5]].values
        target = df["optimized"].values
        features_train, features_test, target_train, target_test = train_test_split(features,target, test_size=test_size_val)
        clf = RandomForestClassifier()
        clf.fit(features_train, target_train)
        target_predict = clf.predict(features_test)
        precisionDict[opt] += precision_score(target_test,target_predict,zero_division=0)
        recallDict[opt] += recall_score(target_test,target_predict,zero_division=0)
        f1Dict[opt] += f1_score(target_test,target_predict,zero_division=0)
        accuracyDict[opt] += accuracy_score(target_test,target_predict)
        matrix = metrics.confusion_matrix(target_test, target_predict)
        confDict[opt] = matAdd(confDict[opt], matrix) 
for i in f1Dict:
    precisionDict[i] = precisionDict[i] / 10
    accuracyDict[i] = accuracyDict[i] / 10
    recallDict[i] = recallDict[i] / 10
    f1Dict[i] = f1Dict[i] / 10
    confDict[i][0][0] = round(confDict[i][0][0] / 10, 2)
    confDict[i][0][1] = round(confDict[i][0][1] / 10, 2)
    confDict[i][1][0] = round(confDict[i][1][0] / 10, 2)
    confDict[i][1][1] = round(confDict[i][1][1] / 10, 2)
subprocess.Popen(["mkdir", "ML-Diagramme"])
##Precision Bar
keys, values = barLists(precisionDict)
y_pos = np.arange(len(keys))
plt.bar(y_pos, values)
plt.xticks(y_pos, keys,fontsize=20)
plt.bar(y_pos, values, color=['red', 'green','black','black','black','black'])
plt.xlabel("Precision Werte",fontsize=20)
plt.ylabel('Anzahl von Optimierungen mit jeweiligem Wert',fontsize=20)
plt.yticks(fontsize=20)
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.savefig("ML-Diagramme/precisionValues.png", dpi = 200)
plt.clf()
#Accuracy Bar
keys, values = barLists(accuracyDict)
y_pos = np.arange(len(keys))
plt.bar(y_pos, values)
plt.xticks(y_pos, keys,fontsize=20)
plt.bar(y_pos, values, color=['red', 'green','black','black','black','black'])
plt.xlabel("Accuracy Werte",fontsize=20)
plt.ylabel('Anzahl von Optimierungen mit jeweiligem Wert',fontsize=20)
plt.yticks(fontsize=20)
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.savefig("ML-Diagramme/accuracyValues.png", dpi = 200)
plt.clf()
#Recall Bar
keys, values = barLists(recallDict)
y_pos = np.arange(len(keys))
plt.bar(y_pos, values)
plt.xticks(y_pos, keys,fontsize=20)
plt.bar(y_pos, values, color=['red', 'green','black','black','black','black'])
plt.xlabel("Recall Werte",fontsize=20)
plt.ylabel('Anzahl von Optimierungen mit jeweiligem Wert',fontsize=20)
plt.yticks(fontsize=20)
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.savefig("ML-Diagramme/recallValues.png", dpi = 200)
plt.clf()
#F1 Bar
keys, values = barLists(f1Dict)
y_pos = np.arange(len(keys))
plt.bar(y_pos, values)
plt.xticks(y_pos, keys,fontsize=20)
plt.bar(y_pos, values, color=['red', 'green','black','black','black','black'])
plt.xlabel("f1 Werte",fontsize=20)
plt.ylabel('Anzahl von Optimierungen mit jeweiligem Wert',fontsize=20)
plt.yticks(fontsize=20)
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.savefig("ML-Diagramme/f1Values.png", dpi = 200)
plt.clf()
#Conf Matrix
subprocess.call(["mkdir", "ML-Diagramme/konfusionsMatrizen"])
for i in confDict:
    plot_confusion_matrix(cm=np.array(confDict[i]), normalize = False, target_names=['0', '1'],title="Konfusionsmatrix",opt = i)

