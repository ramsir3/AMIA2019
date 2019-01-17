#!/usr/bin/python3

import csv, pickle, os.path
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from constants import RAW_PATH

MAX_STAGE = 3
N_STAGES  = 4

def parseDataAvg(fn):
    data = {}
    rd = np.genfromtxt(fn, dtype=int, delimiter=',')
    for r in range(rd.shape[0]):
        k = (rd[r,0], rd[r,1])
        t = rd[r,2]
        v = rd[r,3]
        if k not in data:
            data[k] = {}
        data[k][t] = v

    ofn = fn.rsplit('.')[0]+'_avg.csv'
    with open(ofn, 'w') as fout:
        cf = csv.writer(fout)
        for k, v in data.items():
            avg = sum(v.values()) / len(v.values())
            cf.writerow([k[0], k[1], int(avg / (MAX_STAGE / N_STAGES))])
    return ofn

def stratify(fn):
    data = np.genfromtxt(fn, dtype=int, delimiter=',')
    label = data[:,-1]

    rest, test = train_test_split(data, stratify=label, test_size=0.20)
    devel, valid = train_test_split(rest, stratify=rest[:,-1], test_size=0.25)

    return {"test": test[:,:-1], "devel": devel[:,:-1], "valid": valid[:,:-1]}



# COHORT_ID_FILE = 'simple_stage.csv'
# COHORT_AVG_STAGE_FILE = 'simple_stage_avg.csv'

# # COHORT_AVG_STAGE_FILE = parseData(os.path.join(RAW_PATH, COHORT_ID_FILE)) # only need to run once
# data = stratify(os.path.join(RAW_PATH, COHORT_AVG_STAGE_FILE))
# with open(COHORT_AVG_STAGE_FILE.rsplit('.')[0], 'wb') as f:
#     pickle.dump(data, f)

def splitAndPickel(fn):
    data = np.genfromtxt(fn, dtype=int, delimiter=',')
    rest, test = train_test_split(data, test_size=0.20)
    devel, valid = train_test_split(rest, test_size=0.25)
    split = {"test": test, "devel": devel, "valid": valid}
    with open(fn.rsplit('.')[0]+'_split.pickle', 'wb') as f:
        pickle.dump(split, f)

splitAndPickel(os.path.join(RAW_PATH, 'd_ids.csv'))