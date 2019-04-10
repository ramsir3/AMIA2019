import sys, os, csv
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
from sklearn.linear_model import LogisticRegression
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
    'LogisticRegression': LogisticRegression,
}

def runTraditionalModels(x_train, y_train, x_test, y_test, datafn, model, n, label, hour):
    datafn += '.csv'
    print(datafn)
    print(n)
    model, y_predict = runModel(model, x_train, y_train, x_test, y_test)

    m.fit(x_train, y_train)
    y_predict = m.predict(x_test)

    print('y_predict', y_predict.shape)
    print('y_test', y_test.shape)
    # print('pp', model.predict_proba(x_test).shape)

    acc = skm.accuracy_score(y_test, y_predict)
    auc = skm.roc_auc_score(y_test, model.decision_function(x_test)) \
        if n in {'SVC', 'SGDClassifier', 'GradientBoostingClassifier'} else \
        skm.roc_auc_score(y_test, model.predict_proba(x_test)[:,0])

    cm = skm.confusion_matrix(y_test, y_predict, labels=[True, False])

    TP, FP, FN, TN = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    auc, ppv, npv, sen, spe, f1 = calcScores((TP, FP, TN, FN, auc))
    with open(n + '_binary_' + datafn, 'w') as outf:
        fw = csv.writer(outf)
        fw.writerows(
            [
                ['datahour', 'labelhour', 'TP', 'FP', 'TN', 'FN','ACC', 'AUC', 'PPV', 'NPV', 'SEN', 'SPE', 'F1'],
                [hour, label, TP, FP, TN, FN, acc, auc, ppv, npv, sen, spe, f1],
            ]
        )

ids_fn = os.path.join(RAW_PATH, 'd_ids_split.pickle')
datapath = os.path.join(DATA_PATH, 'test', 'test.zip')
rp = os.path.join('hpo', 'csv')
labelHour = [6, 24, 48, 72]


trn_x, trn_y, tst_x, tst_y = None, None, None, None
datafns = ['HOUR_00001.csv', 'HOUR_00012.csv', 'HOUR_00024.csv']
# for h in datafns:
#     print(h)
#     if trn_x is None:
#         trn_x, trn_y, tst_x, tst_y = load_or_gen_data(datapath, h, ids_fn)
#         trn_y, tst_y = trn_y > 0, tst_y > 0
#     else:
#         trn_xt, trn_yt, tst_xt, tst_yt = load_or_gen_data(datapath, h, ids_fn)
#         trn_yt, tst_yt = trn_yt > 0, tst_yt > 0
#         print(0, trn_x.shape, trn_y.shape)
#         print(1, trn_xt.shape, trn_yt.shape)

#         trn_x = np.concatenate([trn_x, trn_xt], axis=0)
#         trn_y = np.concatenate([trn_y, trn_yt], axis=0)
#         tst_x = np.concatenate([tst_x, tst_xt], axis=0)
#         tst_y = np.concatenate([tst_y, tst_yt], axis=0)

# print(2, trn_x.shape, trn_y.shape)

# all_nan_cols = np.logical_not(np.any(np.isnan(trn_x), axis=0))
# trn_x = trn_x[:,all_nan_cols]
# tst_x = tst_x[:,all_nan_cols]


for fn in os.listdir(rp):
    f_e = fn.rsplit('.', 1)
    if len(f_e) > 1:
        f, e = f_e
        if e  == 'csv':
            splitfn = f.split('_')

            l = splitfn[-1].split('.', 1)[0]
            datafn = '_'.join(splitfn[2:4]) + '.csv'
            label = 'STAGE' + l
            df = pd.read_csv(os.path.join(rp, fn))

            # df = df[df['n_estimators'] > 100]

            bestparams = df.loc[df['_loss'].idxmin()]
            print('\n', f, '\n', bestparams)
            bestparams = bestparams.to_dict()
            model = bestparams['model']

            p = {k: v for k, v in bestparams.items() if k not in ['id', '_loss', 'model']}

            for k in ['max_depth', 'n_estimators', 'min_child_weight']:
                if not np.isnan(p[k]):
                    p[k] = int(p[k])


            p = {k: v for k, v in p.items() if type(v) is str or not np.isnan(v)}
            m = models[model](**p)

            print(0, datapath, datafn)
            trn_x, trn_y, tst_x, tst_y = load_or_gen_data(datapath, datafn, ids_fn)



            print(trn_x.shape, trn_y.shape, tst_x.shape, tst_y.shape)
            i = labelHour.index(int(l))
            print(i, l, labelHour)

            # ohe = lambda a: (np.arange(4) == a[...,None]).astype(int)

            # trn_y = ohe(trn_y[:, i])
            # tst_y = ohe(tst_y[:, i])
            trn_yi = trn_y[:, i] > 0
            tst_yi = tst_y[:, i] > 0


            runTraditionalModels(trn_x, trn_yi, tst_x, tst_yi, f, m, model, label, datafn)

            print("DONE\n\n")
            # break







