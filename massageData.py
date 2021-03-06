#!/usr/bin/env python
# coding: utf-8
import os.path, pickle, math
import numpy as np
import scipy as sp
import pandas as pd

from zipfile import ZipFile
from constants import PROCESSED_PATH, RAW_PATH
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split


def readZipData(dataPath, name, dropNan=True):
    with ZipFile(dataPath) as zf:
        with zf.open(name) as f:
            df = pd.read_csv(f, na_values=['?', '!'])
            df.replace('!.+', np.nan, regex=True, inplace=True)
            check_for_nan_columns = set(df.columns) - {'SUBJECT_ID', 'HADM_ID', 'AGE', 'GENDER', 'ETHNICITY','P TSTAGE','P STAGE','TSTAGE','STAGE'}
            check_for_nan_columns = check_for_nan_columns - {c for c in df.columns if 'STAGE' in c}
            df = df.astype({k: np.float64 for k in check_for_nan_columns}, inplace=True)
            # drop rows where all features=nan
            if dropNan:
                row_nan_bool = np.logical_not(np.all(np.isnan(df[check_for_nan_columns]), axis=1))
                df = df[row_nan_bool]
            df.sort_values(['SUBJECT_ID', 'HADM_ID'], inplace=True)
            return df

def readData(dataPath, dropNan=True):
    df = pd.read_csv(dataPath, na_values=['?', '!'])
    df.replace('!.+', np.nan, regex=True, inplace=True)
    check_for_nan_columns = set(df.columns) - {'SUBJECT_ID', 'HADM_ID', 'AGE', 'GENDER', 'ETHNICITY','P TSTAGE','P STAGE','TSTAGE','STAGE'}
    check_for_nan_columns = check_for_nan_columns - {c for c in df.columns if 'STAGE' in c}
    df = df.astype({k: np.float64 for k in check_for_nan_columns}, inplace=True)
    # drop rows where all features=nan
    if dropNan:
        row_nan_bool = np.logical_not(np.all(np.isnan(df[check_for_nan_columns]), axis=1))
        df = df[row_nan_bool]
    df.sort_values(['SUBJECT_ID', 'HADM_ID'], inplace=True)
    return df

def getDevelTestValidSplit(idsPath, df):
    split_ids = pickle.load(open(idsPath, 'rb'))
    split_df = {}
    for dataset in split_ids:
        split_df[dataset] = df[(df['SUBJECT_ID'].isin(split_ids[dataset][:,0])) & (df['HADM_ID'].isin(split_ids[dataset][:,1]))]
    # for k in split_df:
        # print(k, split_df[k].shape)
    #drop columns where all rows=nan
    check_nan = split_df['devel'].isna().sum()
    split_df['devel'] = split_df['devel'].drop(labels=check_nan[(check_nan == split_df['devel'].shape[0])].keys(), axis=1)
    return split_df

def runRiskFactorAnalysis(data, kruskalCols, threshold=0.05):
# calculate Kruskal-Wallis H-test for each feature
    labels = [l for l in data.columns if ('STAGE' in l) and ('P' not in l) and (l[0] != 'T')]
    dfs_by_class = [data.loc[data[labels[0]] == c] for c in [0,1,2,3]]
    kruskals = {}
    for col in kruskalCols:
        col_in_classes = [np.asarray(c[col].dropna()) for c in dfs_by_class]
        try:
            kruskals[col] = sp.stats.kruskal(*col_in_classes)[1]
        except ValueError:
            kruskals[col] = 0
    return [k for k, v in kruskals.items() if v > threshold]

def meanImpute(data, means=None):
    if means is None:
        means = data.mean()
    data.fillna(means, inplace=True)
    return means

# calculate VIFs
def runVIFAnalysis(data, vifCols):
    features = data[vifCols]
    done = False
    while not done:
        vifs = {}
        for i, n in enumerate(features):
            if i in range(features.shape[1]):
                vifs[n] = variance_inflation_factor(np.asarray(features), i)

        drop_items = sorted(vifs.items(), reverse=True, key=lambda kv: kv[1])
        if len(drop_items) > 0 and drop_items[0][1] >= 5:
            # print(drop_items[0])
            features.drop(labels=[drop_items[0][0]], axis=1, inplace=True)
        else:
            # print(drop_items)
            done = True
    return list(features.columns)

def mode_or_mean(x):
    if x.name in ["SUBJECT_ID", "HADM_ID"]:
        return 99999
    elif x.name in ["ETHNICITY", "TSTAGE", "STAGE"]:
        return np.random.choice(x.mode().dropna())
    else:
        return x.mean()

def synthesizeData(split_df, means, labelHour, n_samples=5):
    # print("synthesizing data")
    stage = 'STAGE%d' % labelHour
    devel_dist = {}
    for s in [0,1,2,3]:
        devel_dist[s] = split_df['devel'][(split_df['devel'][stage] == s)].shape[0] / split_df['devel'].shape[0]

    counts = {subset: dict() for subset in ['test', 'valid']}
    for subset in counts:
        for s in [0,1,2,3]:
            counts[subset][s] = split_df[subset][(split_df[subset][stage] == s)].shape[0]

    dists = {subset: dict() for subset in ['test', 'valid']}
    for subset in dists:
        for s in [0,1,2,3]:
            dists[subset][s] = counts[subset][s] / split_df[subset].shape[0]
    ratios = {}
    for subset in dists:
        # print(subset, dists[subset])
        ratios[subset] = {c: dists[subset][c]/devel_dist[c] for c in devel_dist}
        # print('ratio', ratios[subset])

    split_df['test'].fillna(means, inplace=True)
    split_df['valid'].fillna(means, inplace=True)

    split_df['test'] = split_df['test'][split_df['devel'].columns]
    split_df['valid'] = split_df['valid'][split_df['devel'].columns]

    synthesize = {subset: dict() for subset in ['test', 'valid']}
    for subset in ratios:
        for c in ratios[subset]:
            if ratios[subset][c] < 1:
                num_syn = math.ceil((1-ratios[subset][c])*counts[subset][c])
                # print(subset, c, counts[subset][c], num_syn)
                synthesize[subset][c] = num_syn
    for subset in synthesize:
        for c in synthesize[subset]:
            for s in range(synthesize[subset][c]):
                samples = resample(
                        split_df[subset][(split_df[subset][stage] == c)],
                        n_samples=n_samples,
                    )
                synth = samples.apply(mode_or_mean)
                split_df[subset] = split_df[subset].append(synth, ignore_index=True)
            # print(subset, split_df[subset].shape)
    # for subset in synthesize:
    #     for c in synthesize[subset]:
    #         # print(subset, c, split_df[subset][(split_df[subset][stage] == c)].shape[0])

    return split_df

def runPipeline(data, splitIDsPath, featuresCols, labelHours, doRFA=False, doVIFA=False, normImpute=False):
    split_df = getDevelTestValidSplit(splitIDsPath, data)
    featuresCols = [c for c in featuresCols if c in split_df['devel']]

    if doRFA:
        kcols = runRiskFactorAnalysis(split_df['devel'], featuresCols)
        split_df['devel'] = split_df['devel'].drop(list(set(featuresCols)-set(kcols)), axis=1)
        featuresCols = kcols

    means = None
    # fc = split_df['devel'].columns.groupby(split_df['devel'].dtypes)[np.dtype(float)]
    # print(np.sum(np.isnan(split_df['devel'][fc].values)))
    if normImpute:
        means = pickle.load(open(os.path.join(PROCESSED_PATH, 'avgvector_normals.pickle'), 'rb'))
        split_df['devel'] = split_df['devel'].fillna(means)
        split_df['test'] = split_df['test'].fillna(means)
        split_df['valid'] = split_df['valid'].fillna(means)

    means = meanImpute(split_df['devel'])
    meanImpute(split_df['test'], means)
    meanImpute(split_df['valid'], means)
    # print(np.sum(np.isnan(split_df['devel'][fc].values)))

    if doVIFA:
        vcols = runVIFAnalysis(split_df['devel'], featuresCols)
        split_df['devel'] = split_df['devel'].drop(list(set(featuresCols)-set(vcols)), axis=1)

    for labelHour in labelHours:
        split_df = synthesizeData(split_df, means, labelHour)
    return split_df



