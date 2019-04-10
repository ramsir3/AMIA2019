import pickle
import numpy as np
import pandas as pd
from constants import CE_ITEM_LABELS, TOP_90_LE_ITEM_IDS, ITEM_IDS_UOM, CONVERSIONS
from lablabels import LAB_LABELS

def getHeaders(fn):
    return np.genfromtxt(fn, max_rows=1, delimiter=',', dtype=str)

ITEM_IDS = CE_ITEM_LABELS + TOP_90_LE_ITEM_IDS

avgchart = {k: list() for k in ITEM_IDS_UOM}
with open('data/processed/avgchart.pickle', 'rb') as ac:
    ac = pickle.load(ac)
    # print(ac)
    for itemid, values in ac.items():
        for cat in ITEM_IDS_UOM:
            if itemid in ITEM_IDS_UOM[cat]:
                # print(itemid)
                avgchart[cat].append((CONVERSIONS[cat](values[0], ITEM_IDS_UOM[cat][itemid]), values[1]))

for cat in avgchart:
    num = sum(v[0]*v[1] for v in avgchart[cat])
    den = sum(v[1] for v in avgchart[cat])
    avgchart[cat] = num/den

labelHours = [6, 12, 24, 36, 48, 72]
headers = ["HOUR", "SUBJECT_ID", "HADM_ID", "ETHNICITY", "AGE", "GENDER", ]
i1start = len(headers)

headers += ITEM_IDS
i1end = len(headers)

headers += [f(h) for h in labelHours for f in (lambda x: '%s%d'%(s,x) for s in ["PTSTAGE", "PSTAGE"])]
i2start = len(headers)

headers += ITEM_IDS
i2end = len(headers)

headers += [f(h) for h in labelHours for f in (lambda x: '%s%d'%(s,x) for s in ["TSTAGE", "STAGE"])]

normals = np.asarray([
            np.nan, np.nan, 120, 80, 37, 16, 80, 97.5, # CE items
            1, 14, 43.75, 250, 7.3, 14.75, 34, 30, 90, 5.05, 13, 4.25, 140.5, 102,25.5, 10,
            np.nan, 1.95, 3.75, 9.75, 0.95, 12, 37.5, 7.41, 6, 1.016, np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, 25.5, 90, 40, np.nan, 17.5, 17.5, 80, 0.75, 64, np.nan, 0.6, np.nan, 4.5,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2.5, 4, np.nan, np.nan, 4.25, np.nan, 100,
            np.nan, np.nan, np.nan, np.nan, 110, np.nan, 14.75, 140.5, 97.5, np.nan, 90, 0.03, 102, 37,
            np.nan, 65, np.nan, 250, 500, np.nan, np.nan, np.nan, np.nan, 160, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan, 20
])

avgvect = np.empty_like(headers, dtype=float)
avgvect[:] = np.nan
avgvect[i1start:i1end] = normals
avgvect[i2start:i2end] = normals

with open('data/processed/avglabs.pickle', 'rb') as ai:
    ai = pickle.load(ai)
    print(ai.keys())
    for i, h in enumerate(headers):
        rh = h
        # print(rh)
        # if i > 10:
        #     break
        if avgvect[i] is np.nan:
            if rh in avgchart:
                # print(h)
                avgvect[i] = avgchart[rh]
            elif rh in ai:
                    avgvect[i] = ai[rh]
            else:
                print(h)
print(len(headers))
print(np.sum(np.logical_not(np.isnan(avgvect))))
print(len(avgvect))

ITEM_LABELS = CE_ITEM_LABELS + [LAB_LABELS[i] for i in TOP_90_LE_ITEM_IDS]

headers = [ 
    "HOUR",
    "SUBJECT_ID",
    "HADM_ID",
    "ETHNICITY",
    "AGE",
    "GENDER",
] + ['P ' + i for i in ITEM_LABELS] + [f(h) for h in labelHours for f in (lambda x: '%s%d'%(s,x) for s in ["PTSTAGE", "PSTAGE"])] + ITEM_LABELS + [f(h) for h in labelHours for f in (lambda x: '%s%d'%(s,x) for s in ["TSTAGE", "STAGE"])]

# print(pd.Series(index=headers, data=avgvect))

with open('data/processed/avgvector_normals.pickle', 'wb') as av:
    avgvect = pd.Series(index=headers, data=avgvect)
    pickle.dump(avgvect, av)
