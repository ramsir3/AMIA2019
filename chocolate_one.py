import os, pickle, datetime, sys, warnings
import pathos.multiprocessing as mp
import sklearn.metrics as skm
import numpy as np
import chocolate as choco

from constants import PROCESSED_PATH, RAW_PATH, DATA_PATH
from massageData import runPipeline, readZipData
from sklearn.preprocessing import normalize, LabelBinarizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

def prepareData(datafn, name, ids_fn, one_hot=False, rfa=False):
    df = readZipData(datafn, name, dropNan=True)
    df = df[df.columns[[0,1,4,2,3]+list(range(5,len(df.columns)))]]

    featuresCols = df.columns[3:-12]

    split_df = runPipeline(df, ids_fn, featuresCols, [6, 24, 48, 72], normImpute=True, doRFA=rfa)


    test = split_df['test']
    # valid = split_df['valid']
    devel = split_df['devel']


    dropcols = ['SUBJECT_ID','HADM_ID','ETHNICITY', 'PTSTAGE6', 'PSTAGE6', 'PTSTAGE12',
                        'PSTAGE12', 'PTSTAGE24', 'PSTAGE24', 'PTSTAGE36', 'PSTAGE36', 'PTSTAGE48',
                        'PSTAGE48', 'PTSTAGE72', 'PSTAGE72', 'TSTAGE6', 'TSTAGE12', 'STAGE12',
                        'TSTAGE24', 'TSTAGE36', 'STAGE36', 'TSTAGE48', 'TSTAGE72']
    dropcols = [c for c in dropcols if c in devel.columns]
    # keep 6, 24, 48, 72
    train = devel.drop(dropcols, axis=1)
    testv = test.drop(dropcols, axis=1)

    # return train

    # print(train.columns[-4:])
    x_train = train.values[:, :-4]
    y_train = train.values[:, -4:].astype(int)
    x_test = testv.values[:, :-4]
    y_test = testv.values[:, -4:].astype(int)

    # print(np.sum(np.isnan(x_train)))
    # print(np.all(np.isnan(x_train), axis=0))
    # print(np.any(np.isnan(x_train), axis=0))
    # x_test = normalize(x_test, axis=0)
    # x_train = normalize(x_train, axis=0)

    # if one_hot:
    #     ohe = LabelBinarizer()
    #     ohe.fit(y_train.reshape(-1, 1))
    #     y_train = ohe.transform(y_train.reshape(-1,1))
    #     y_test = ohe.transform(y_test.reshape(-1,1))

    return x_train, y_train, x_test, y_test

def load_or_gen_data(datafn, name, ids_fn, ohe=False, rfa=False):
    folder, _ = os.path.split(datafn)
    check = os.path.join(folder, name.rsplit('.')[0] + '.pickle')
    if os.path.isfile(check):
        with open(check, 'rb') as f:
            return pickle.load(f)
    else:
        d = prepareData(datafn, name, ids_fn, ohe, rfa=rfa)
        with open(check, 'wb') as f:
            pickle.dump(d, f)
        return d

def getProcFunc(nseed, nruns):
    def func(data):
        trn_x, trn_y, tst_x, tst_y, dbid = data
        conn = choco.SQLiteConnection(url="sqlite:///hpo/hpo_%s.db" % str(dbid))
        sampler = choco.Random(conn, space)
        searcher = choco.Bayes(conn, space)
        print('START %s' % dbid)

        for _ in range(nseed):
            token, params = sampler.next()
            # print('START % 4d %s' % (i, params['model']))
            loss = f1_score_model(trn_x, trn_y, tst_x, tst_y, **params)
            sampler.update(token, loss)
            # print('DONE  % 4d %s' % (i, params['model']))
        for _ in range(nruns):
            token, params = searcher.next()
            # print('START % 4d %s' % (i, params['model']))
            loss = f1_score_model(trn_x, trn_y, tst_x, tst_y, **params)
            searcher.update(token, loss)
            # print('DONE  % 4d %s' % (i, params['model']))
    return func

def f1_score_model(trn_x, trn_y, tst_x, tst_y, model, **params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = models[model](**params)
        m.fit(trn_x, trn_y)
        y_pred = m.predict(tst_x)
        return -1*skm.f1_score(tst_y, y_pred, average='macro')

space = [
    # {'model': 'RandomForestClassifier',
    #     "max_depth"       : choco.quantized_uniform(2, 32, 2),
    #     "min_samples_split": choco.quantized_uniform(2, 600, 2),
    #     "n_estimators"    : choco.quantized_uniform(125, 800, 25),},
    {'model': 'SVC',
        "gamma": 'auto',
        "C": choco.log(-3, 3, 10),
        "kernel": choco.choice(['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']),
        "tol": choco.log(-5, -2, 10),},
    {'model': 'XGBClassifier',
        "learning_rate"   : choco.uniform(0.001, 0.1),
        "max_depth"       : choco.quantized_uniform(2, 16, 2),
        "min_child_weight": choco.quantized_uniform(2, 10, 2),
        "subsample"       : choco.quantized_uniform(0.7, 1.05, 0.05),
        "n_estimators"    : choco.quantized_uniform(25, 525, 25),},
    {'model': 'LogisticRegression',
        "penalty"         : choco.choice(['l1', 'l2']),
        "C"               : choco.log(-2, 1, 10),},
]

models = {
    'RandomForestClassifier': RandomForestClassifier,
    'SVC': SVC,
    'XGBClassifier': XGBClassifier,
    'LogisticRegression': LogisticRegression,
}


if __name__ == "__main__":

    N_SEED = 1
    N_RUNS = 1
    N_PROC = 12

    datapath = os.path.join(DATA_PATH, 'test', 'test.zip')

    ids_fn = os.path.join(RAW_PATH, 'd_ids_split.pickle')
    datafns = ['HOUR_00001.csv', 'HOUR_00012.csv', 'HOUR_00024.csv']

    # for h in datafns:
    #    t = prepareData(datapath, h, ids_fn, False, True)
    #    print(h, list(t.columns))


    labelHour = [6, 24, 48, 72]
    data = []
    # trn_x, trn_y, tst_x, tst_y = None, None, None, None
    # for h in datafns:
    #     print(h)
    #     if trn_x is None:
    #         trn_x, trn_y, tst_x, tst_y = load_or_gen_data(datapath, h, ids_fn, rfa=True)
    #         trn_y, tst_y = trn_y > 0, tst_y > 0
    #     else:
    #         trn_xt, trn_yt, tst_xt, tst_yt = load_or_gen_data(datapath, h, ids_fn, rfa=True)
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

    # print(np.any(np.isnan(trn_x), axis=0))

    # print(3, trn_x.shape, trn_y.shape)

    for h in datafns:
        for i, lh in enumerate(labelHour):
            trn_x, trn_y, tst_x, tst_y = load_or_gen_data(datapath, h, ids_fn, rfa=True)
            # print(np.sum(np.isnan(trn_x)))
            data.append((trn_x, trn_y[:,i], tst_x, tst_y[:,i], 'Data_%sLabel_%d' % (h.split('.')[0], lh)))
            # print(trn_x.shape, trn_y[:,i].shape)
            # break

    f = getProcFunc(N_SEED, N_RUNS)
    with mp.Pool(processes=N_PROC) as pool:
        pool.map(f, data)
