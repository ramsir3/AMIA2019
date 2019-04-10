import os
import pandas as pd
import numpy as np
from constants import DATA_PATH

labelpath = os.path.join(DATA_PATH, 'hourlabel')
hourpath = os.path.join(DATA_PATH, 'hour')
outpath = os.path.join(DATA_PATH, 'test')

mask = lambda h: np.logical_not(np.isin(label[h].values, data[h].values))

for file in os.listdir(hourpath):
    if file.rsplit('.')[1] == 'csv':
        print(file)
        data = pd.read_csv(os.path.join(hourpath, file), dtype=str)
        label = pd.read_csv(os.path.join(labelpath, file), dtype=str)
        dataids = data['HADM_ID']
        labelids = label['HADM_ID']
        # print(len(label['HADM_ID'].unique()))

        v = labelids.isin(dataids).values
        # b = np.logical_and(v[:,0], v[:,1])
        # print(b[:10])
        label = label.iloc[v,:]

        # label.drop(labels=label.index[
        #         (mask('SUBJECT_ID')) &
        #         (mask('HADM_ID'))
        #     ],
        #     inplace=True
        # )
        # print(label.shape, data.shape)
        if label.shape[0] == data.shape[0]:
            data.drop(['P TSTAGE', 'P STAGE', 'TSTAGE', 'STAGE'], axis=1, inplace=True)
            lcols = list(label.columns[2:])
            pl = lcols[:12]
            l = lcols[12:]
            dcols = list(data.columns)
            data[lcols] = label[lcols]
            data = data[dcols[:103]+pl+dcols[103:]+l]
            f = os.path.join(outpath, file)
            # print(f)
            data.to_csv(f, index=False)

        # break