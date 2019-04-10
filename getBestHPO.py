import sys, os
import pandas as pd
import numpy as np

from constants import PROCESSED_PATH, RAW_PATH, DATA_PATH
from massageData import runPipeline, readData
from runTraditionalModels import runMetrics, runModel, multiclass_auc, calcScores
from chocolate_one import load_or_gen_data

import sklearn.metrics as skm
from sklearn.preprocessing import normalize, LabelBinarizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

models = {
    'SVC': SVC,
    'XGBClassifier': XGBClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'GaussianNB': GaussianNB,
    'KNeighborsClassifier': KNeighborsClassifier,
}

def runTraditionalModels(x_train, ohe_y_train, x_test, ohe_y_test, datafn, model, n, label):
    datafn += '.csv'
    print(datafn)
    print(n)
    model, y_predict = runModel(model, x_train, ohe_y_train, x_test, ohe_y_test)
    accScores = skm.accuracy_score(ohe_y_test, y_predict)
    rawScores = runMetrics(ohe_y_test, y_predict)
    if n in {'SVC', 'SGDClassifier', 'GradientBoostingClassifier'}:
        aucs = multiclass_auc(ohe_y_test, model.decision_function(x_test))
    else:
        aucs = multiclass_auc(ohe_y_test, model.predict_proba(x_test))

    for k in rawScores.keys():
        rawScores[k].append(aucs[k])    

    with open("RAW_OUTPUT_"+datafn, 'w') as fout:
        header = n + '_' + str(label) + '\n'
        header+= ',,,,'.join("CLASS %d (%d)" % (c, np.sum(ohe_y_test[:,c])) for c in [0,1,2,3]) + '\n'
        header+= ','.join("TP,FP,TN,FN" for c in [0,1,2,3]) + '\n'
        fout.write(header)
        for c in rawScores:
            for i in rawScores[c][:-1]:
                fout.write(str(i)+',')

    with open("OUTPUT_"+datafn, 'w') as fout:
        header = ',AVG (%d),,,,,,,' % ohe_y_test.shape[0] + ',,,,,,'.join("CLASS %d (%d)" % (c, np.sum(ohe_y_test[:,c])) for c in [0,1,2,3]) + '\n'
        header+= n + '_' + str(label) + ','
        # header+= 'ACC,AUC,PPV,NPV,SEN,SPE,F1,' + ','.join("AUC,PPV,NPV,SEN,SPE,F1" for c in [0,1,2,3]) + '\n'
        fout.write(header)
        calcedScores = [calcScores(rawScores[k]) for k in rawScores]
        avgScores = [0 for i in calcedScores[0]]
        for i, c in enumerate(calcedScores):
            for j, _ in enumerate(c):
                avgScores[j] += calcedScores[i][j]
        for i, v in enumerate(avgScores):
            avgScores[i] = v/len(calcedScores)       

        fout.write(str(accScores)+',')
        for v in avgScores:
            fout.write(str(v)+',')
        for c in calcedScores:
            for i in c:
                fout.write(str(i)+',')

ids_fn = os.path.join(RAW_PATH, 'd_ids_split.pickle')
datapath = os.path.join(DATA_PATH, 'hpotest')
rp = os.path.join('hpo', 'csv')
labelHour = [6, 24, 48, 72]

for fn in os.listdir(rp):
    f, e = fn.rsplit('.', 1)
    if e  == 'csv':
        datafn, l = f.rsplit('_', 1)
        l = l[:-3]
        datafn = datafn.split('_', 1)[1]
        datafn += '.csv'
        label = 'STAGE' + l
        df = pd.read_csv(os.path.join(rp, fn))

        df = df[df['n_estimators'] > 100]

        bestparams = df.loc[df['_loss'].idxmin()]
        print('\n', f, '\n', bestparams)
        bestparams = bestparams.to_dict()
        model = bestparams['model']
        p = {k: v for k, v in bestparams.items() if k not in ['id', '_loss', 'model']}
        print(p)

        m = models[model](**p)
        trn_x, trn_y, tst_x, tst_y = load_or_gen_data(os.path.join(datapath, datafn), ids_fn)
        
        print(trn_x.shape, trn_y.shape, tst_x.shape, tst_y.shape)
        i = labelHour.index(int(l))

        ohe = lambda a: (np.arange(4) == a[...,None]).astype(int)

        trn_y = ohe(trn_y[:, i])
        tst_y = ohe(tst_y[:, i])

        runTraditionalModels(trn_x, trn_y, tst_x, tst_y, f, m, model, label)

        print("DONE\n\n")
        # break




    

    