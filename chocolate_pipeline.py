import os, pickle, datetime
import pathos.multiprocessing as mp
import sklearn.metrics as skm
import numpy as np
import chocolate as choco

from constants import PROCESSED_PATH, RAW_PATH, DATA_PATH
from massageData import runPipeline, readData
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

def prepareData(datafn, ids_fn, one_hot=False):
    df = readData(datafn)
    df = df[df.columns[[0,1,4,2,3]+list(range(5,len(df.columns)))]]

    featuresCols = df.columns[3:-2]

    split_df = runPipeline(df, ids_fn, featuresCols)

    test = split_df['test']
    # valid = split_df['valid']
    devel = split_df['devel']

    train = devel.drop(['SUBJECT_ID','HADM_ID','ETHNICITY','TSTAGE', 'P TSTAGE', 'P STAGE'], axis=1)
    testv = test.drop(['SUBJECT_ID','HADM_ID','ETHNICITY','TSTAGE', 'P TSTAGE', 'P STAGE'], axis=1)

    x_train = train.values[:, :-2]
    y_train = train.values[:, -1]
    x_train = normalize(x_train, axis=0)

    x_test = testv.values[:, :-2]
    x_test = normalize(x_test, axis=0)
    y_test = testv.values[:, -1]

    if one_hot:
        ohe = LabelBinarizer()
        ohe.fit(y_train.reshape(-1, 1))
        y_train = ohe.transform(y_train.reshape(-1,1))
        y_test = ohe.transform(y_test.reshape(-1,1))
    
    return x_train, y_train, x_test, y_test

def load_or_gen_data(datafn, ids_fn):
    folder, fn = os.path.split(datafn)
    check = os.path.join(folder, fn.rsplit('.')[0] + '.pickle')
    if os.path.isfile(check):
        with open(check, 'rb') as f:
            return pickle.load(f)
    else:
        with open(check, 'wb') as f:
            d = prepareData(datafn, ids_fn)
            pickle.dump(d, f)
            return d

def getProcFunc(conn, sampler):
    def func(i):
        token, params = sampler.next()
        print('START % 4d %s' % (i, params['model']))
        loss = f1_score_model(trn_x, trn_y, tst_x, tst_y, **params)
        sampler.update(token, loss)
        print('DONE  % 4d %s' % (i, params['model']))
    return func

def f1_score_model(trn_x, trn_y, tst_x, tst_y, model, **params):
    m = models[model](**params)
    m.fit(trn_x, trn_y)
    y_pred = m.predict(tst_x)
    return -1*skm.f1_score(tst_y, y_pred, average='macro')

space = [
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
    {'model': 'RandomForestClassifier',
        "max_depth"       : choco.quantized_uniform(2, 10, 2),
        "min_samples_leaf": choco.quantized_uniform(2, 10, 2),
        "n_estimators"    : choco.quantized_uniform(25, 525, 25),},
    {'model': 'GaussianNB',
        "var_smoothing"   : choco.log(-12, -6, 10)},
    {'model': 'KNeighborsClassifier',
        "n_neighbors"     : choco.quantized_uniform(1, 10, 1),
        "weights"         : choco.choice(['uniform', 'distance']),
        "leaf_size"       : choco.quantized_uniform(15, 315, 20),
        "p"               : choco.choice([1,2,3]),},
    # {'model': MLPClassifier,},
    # {'model': GaussianProcessClassifier}, # this one was giving me an out of memory error
]

models = {
    'SVC': SVC,
    'XGBClassifier': XGBClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'GaussianNB': GaussianNB,
    'KNeighborsClassifier': KNeighborsClassifier,
}


if __name__ == "__main__":
    datafn = 'HOUR_00024.csv'
    dbid = datafn.split('_')[1].split('.')[0]
    # dbid = datetime.datetime.now().strftime('%m%d%y%H%M%S')
    # dbid = 1

    N_RUNS = 1024
    N_PROC = 8

    datafn = os.path.join(DATA_PATH, 'hour', datafn)
    ids_fn = os.path.join(RAW_PATH, 'd_ids_split.pickle')

    trn_x, trn_y, tst_x, tst_y = load_or_gen_data(datafn, ids_fn)

    conn = choco.SQLiteConnection(url="sqlite:///hpo/hpo_%s.db" % str(dbid))
    # searcher = choco.Random(conn, space)
    searcher = choco.Bayes(conn, space)

    f = getProcFunc(conn, searcher)
    with mp.Pool(processes=N_PROC) as pool:
        pool.map(f, range(N_RUNS))

    df = conn.results_as_dataframe()
    df.to_csv("hpo/hpo_%s.csv" % str(dbid))






