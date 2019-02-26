import pickle
import numpy as np
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
headers = ["HOUR", "SUBJECT_ID", "HADM_ID", "ETHNICITY", "AGE", "GENDER", ] \
    + ITEM_IDS \
    + [f(h) for h in labelHours for f in (lambda x: '%s%d'%(s,x) for s in ["PTSTAGE", "PSTAGE"])] \
    + ITEM_IDS \
    + [f(h) for h in labelHours for f in (lambda x: '%s%d'%(s,x) for s in ["TSTAGE", "STAGE"])]


avgvect = np.empty_like(headers, dtype=float)
avgvect[:] = np.nan
with open('data/processed/avglabs.pickle', 'rb') as ai:
    ai = pickle.load(ai)
    print(ai.keys())
    for i, h in enumerate(headers):
        rh = h
        # print(rh)
        # if i > 10:
        #     break
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

with open('data/processed/avgvector.pickle', 'wb') as av:
    pickle.dump(avgvect, av)
