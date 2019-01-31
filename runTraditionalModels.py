import sklearn.metrics as skm
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def runModel(model, x, y, x_test, y_test):
    m = OneVsRestClassifier(model)
    m.fit(x,y)
    y_predict = m.predict(x_test)
    
    return m, y_predict

def runMetrics(y_test, y_predict):
    TP = {}
    FP = {}
    TN = {}
    FN = {}

    for i in range(y_predict.shape[1]): 
        TP[i] = np.sum(np.logical_and(y_predict[:,i]==1,y_test[:,i]==1))
        FP[i] = np.sum(np.logical_and(y_predict[:,i]==1,y_test[:,i]!=y_predict[:,i]))
        TN[i] = np.sum(np.logical_and(y_predict[:,i]==0,y_test[:,i]==0))
        FN[i] = np.sum(np.logical_and(y_predict[:,i]==0,y_test[:,i]!=y_predict[:,i]))
    
    out = {}
    for i in range(y_predict.shape[1]): 
        out[i] = [TP[i], FP[i], TN[i], FN[i]]
    return out

# AUC,PPV,NPV,SEN,SPE,F1
# TP,FP,TN,FN
def calcScores(rs):
#     print('?',rs)
    auc = rs[4]
    ppv = rs[0]/(rs[0]+rs[1])
    npv = rs[2]/(rs[2]+rs[3])
    sen = rs[0]/(rs[0]+rs[3])
    spe = rs[2]/(rs[0]+rs[1])
    f1  = (2*rs[0])/((2*rs[0])+rs[1]+rs[3])
    return (auc,ppv,npv,sen,spe,f1)   

def multiclass_auc(y_test, y_score):
    n_classes = 4
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = skm.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = skm.auc(fpr[i], tpr[i])
    roc_auc['avg'] = sum(roc_auc.values())/n_classes
    return roc_auc

def runTraditionalModels(x_train, ohe_y_train, x_test, ohe_y_test, datafn):
    models = {
        'SVC': SVC(),
        'SGDClassifier': SGDClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'GaussianNB': GaussianNB(),
        'KNeighborsClassifier': KNeighborsClassifier(),
    #     'GaussianProcessClassifier': GaussianProcessClassifier(), # this one was giving me an out of memory error
        'MLPClassifier': MLPClassifier(),
    }

    rawScores = {}
    accScores = {}
    for n, m in models.items():
        print(n)
        m, y_predict = runModel(m, x_train, ohe_y_train, x_test, ohe_y_test)
        accScores[n] = skm.accuracy_score(ohe_y_test, y_predict)
        rawScores[n] = runMetrics(ohe_y_test, y_predict)
        if n in {'SVC', 'SGDClassifier', 'GradientBoostingClassifier'}:
            aucs = multiclass_auc(ohe_y_test, m.decision_function(x_test))
        else:
            aucs = multiclass_auc(ohe_y_test, m.predict_proba(x_test))

        for k in rawScores[n].keys():
            rawScores[n][k].append(aucs[k])    

    with open("RAW_OUTPUT_"+datafn, 'w') as fout:
        header = ',' + ',,,,'.join("CLASS %d (%d)" % (c, np.sum(ohe_y_test[:,c])) for c in [0,1,2,3]) + '\n'
        header+= 'MODEL,' + ','.join("TP,FP,TN,FN" for c in [0,1,2,3]) + '\n'
        fout.write(header)
        for m in rawScores:
            fout.write(m+',')
            for c in rawScores[m]:
                for i in rawScores[m][c][:-1]:
                    fout.write(str(i)+',')
            fout.write('\n') 

    with open("OUTPUT_"+datafn, 'w') as fout:
        header = ',AVG (%d),,,,,,' % ohe_y_test.shape[0] + ',,,,,,'.join("CLASS %d (%d)" % (c, np.sum(ohe_y_test[:,c])) for c in [0,1,2,3]) + '\n'
        header+= 'MODEL,ACC,AUC,PPV,NPV,SEN,SPE,F1,' + ','.join("AUC,PPV,NPV,SEN,SPE,F1" for c in [0,1,2,3]) + '\n'
        fout.write(header)
        for m in rawScores:
            calcedScores = [calcScores(rawScores[m][k]) for k in rawScores[m]]
            avgScores = [0 for i in calcedScores[0]]
            for i, c in enumerate(calcedScores):
                for j, _ in enumerate(c):
                    avgScores[j] += calcedScores[i][j]
            for i, v in enumerate(avgScores):
                avgScores[i] = v/len(calcedScores)       

            fout.write(m+',')
            fout.write(str(accScores[m])+',')
            for v in avgScores:
                fout.write(str(v)+',')
            for c in calcedScores:
                for i in c:
                    fout.write(str(i)+',')
            fout.write('\n')