import pandas as pd
import numpy as np

import os, pickle, datetime
from collections import defaultdict

from sklearn.utils import shuffle
import sklearn.metrics as skm

from constants import PROCESSED_PATH, DATA_PATH


__all__ = ['getSplitIds', 'DataBatch', 'dropNanCols', 'prepareData', 'saveAUCs', 'saveMetrics']

avgvec = pickle.load(open(os.path.join(PROCESSED_PATH, 'avgvector.pickle'), 'rb'))

def getSplitIds(idsPath):
    split_ids = pickle.load(open(idsPath, 'rb'))
    for subset in split_ids:
        split_ids[subset] = set(zip(split_ids[subset][:,0], split_ids[subset][:,1]))
    return split_ids
    
def getData(batch):
    d = lambda fn: np.genfromtxt(fn, skip_header=1, delimiter=',', missing_values=['?', '!'])[:,:,np.newaxis]
    
    data = None
    for fn in batch:
        if data is None:
            data = d(fn)
        else:
            pt = d(fn)
            data = np.concatenate([data, pt], axis=2)
            
    return np.moveaxis(data,-1,0)

class BatchIterator():
    def __init__(self, path, files, batchSize):
        self.path = path
        self.files = files
        self.batchSize = batchSize
        self.seed = None
    
    def setSeed(self, seed):
        self.seed = seed
        return self
    
    def __iter__(self):
        self.batchNum = 0
        shuffle(self.files, random_state=self.seed)
        return self
    
    def __next__(self):
        start = self.batchNum*self.batchSize
        if start >= len(self.files):
            raise StopIteration
        end = (self.batchNum+1)*self.batchSize
        self.batchNum += 1
        if end > len(self.files):
            end = len(self.files)
        return getData(os.path.join(self.path, f) for f in self.files[start:end])
          

class DataBatch():
    def __init__(self, path, split_ids=None, batchSize=1000):
        if batchSize <= 0:
            raise ValueError('batchSize must be > 0')
        self.path = path
        self.batchSize = batchSize
        self.files = {}
        filels = os.listdir(path)
        if split_ids != None:
            for fn in filels:
                if fn.startswith('.'):
                    continue
                fis = fn.split('_')
                vid = (int(fis[0][1:]), int(fis[1][1:-4]))
                for subset in split_ids:
                    if vid in split_ids[subset]:
                        if subset in self.files:
                            self.files[subset].append(fn)
                        else:
                            self.files[subset] = [fn]
        else:
            self.files['all'] = [f for f in filels if not f.startswith('.')]
                        
    def getBatchIterator(self, key='all'):
        return BatchIterator(self.path, self.files[key], self.batchSize)

    def getHeaders(self):
        fn = os.path.join(self.path, next(iter(self.files.values()))[0])
        return np.genfromtxt(fn, max_rows=1, delimiter=',', missing_values=['?', '!'], dtype=str)

def dropNanCols(data=None, headers=None, row_nan_bool=None):
    if data is None and row_nan_bool is None:
        raise ValueError('data and row_nan_bool cannot both be none')
    if row_nan_bool is None:
        row_nan_bool = np.logical_not(np.all(np.all(np.isnan(data), axis=1), axis=0))
    if data is not None:
        data = data[:,:,row_nan_bool]
    if headers is not None:
        headers = headers[row_nan_bool]
    return data, headers, row_nan_bool

def prepareData(data, headers, nclasses=4, dropPhrases=['PSTAGE', 'PTSTAGE'], labelCol='STAGE', labelStart=-2, avgvec=avgvec, debug=False):
    hdr = {h: i for i, h in enumerate(headers)}
    delcols = ['HOUR', 'P TSTAGE', 'P STAGE']
    delcols += [h for h in headers for p in dropPhrases if p in h]
    badcols = [
        'ESTIMATED GFR (MDRD EQUATION)', 'BILIRUBIN', 'BLOOD', 'VENTILATOR',
        'MACROCYTES', 'SPECIMEN TYPE', 'URINE APPEARANCE', 'MICROCYTES',
        'INTUBATED', 'NITRITE', 'ETHNICITY'
        ]
    delcols += badcols
    delcols += ['P ' + bc for bc in badcols]
    data2 = np.delete(
        data,
        [hdr[h] for h in delcols if h in hdr], 
        axis=2
    )
    headers2 = np.delete(
        headers,
        [hdr[h] for h in delcols if h in hdr]
    )
    avgvec = np.delete(
        avgvec,
        [hdr[h] for h in delcols if h in hdr]
    )
    hdr = {h: i for i, h in enumerate(headers2)}

    ids = data2[:,0,[0,1]].astype(int)
    X = data2[:,:,2:labelStart]
    Y = data2[:,:,hdr[labelCol]]
    avgvec = avgvec[2:labelStart]
    headers2 = headers2[2:labelStart]

    ohe = lambda a: (np.arange(nclasses) == a[...,None]).astype(int)

    Y = ohe(Y)

    means = np.nanmean(X, axis=1)
    xmeans = np.nanmean(means, axis=0)
    xind = np.where(np.isnan(means))
    means[xind] = np.take(xmeans, xind[1])
    ind = np.where(np.isnan(X))
    X[ind] = means[ind[0],ind[2]]
    ind = np.where(np.isnan(X))
    X[ind] = avgvec[ind[2]]
    
    if debug:
        print('input:\n' + '%3d x %3d x %3d' % X.shape + '\n%3s x %3s x %3s' % ('v','h','f'))
        print('output:\n' + '%3d x %3d x %3d' % Y.shape + '\n%3s x %3s x %3s\n' % ('v','h','s'))
        
    return ids, X, Y, headers2

def runMetricsHour(y_test, y_predict, nclasses=4):
    TP = defaultdict(list)
    FP = defaultdict(list)
    TN = defaultdict(list)
    FN = defaultdict(list)

    for h in range(y_predict.shape[1]):
        for i in range(nclasses): 
            TP[i].append(np.sum(np.logical_and(y_predict[:,h]==i,y_test[:,h]==i)))
            FP[i].append(np.sum(np.logical_and(y_predict[:,h]==i,y_test[:,h]!=i)))
            TN[i].append(np.sum(np.logical_and(y_predict[:,h]!=i,y_test[:,h]!=i)))
            FN[i].append(np.sum(np.logical_and(y_predict[:,h]!=i,y_test[:,h]==i)))

    out = {}
    for i in range(nclasses): 
        out[i] = np.asarray([TP[i], FP[i], TN[i], FN[i]])
    return out

def runMetrics(y_test, y_predict, nclasses=4):
    TP = {}
    FP = {}
    TN = {}
    FN = {}

    for i in range(nclasses): 
        TP[i]=(np.sum(np.logical_and(y_predict[:,:]==i,y_test[:,:]==i)))
        FP[i]=(np.sum(np.logical_and(y_predict[:,:]==i,y_test[:,:]!=i)))
        TN[i]=(np.sum(np.logical_and(y_predict[:,:]!=i,y_test[:,:]!=i)))
        FN[i]=(np.sum(np.logical_and(y_predict[:,:]!=i,y_test[:,:]==i)))

    out = {}
    for i in range(nclasses): 
        out[i] = np.asarray([TP[i], FP[i], TN[i], FN[i]])
    return out


# AUC,PPV,NPV,SEN,SPE,F1
# TP,FP,TN,FN
def calcScores(rs):
#     print('?',rs)
    acc = (rs[0]+rs[2])/sum(rs)
    ppv = rs[0]/(rs[0]+rs[1])
    npv = rs[2]/(rs[2]+rs[3])
    sen = rs[0]/(rs[0]+rs[3])
    spe = rs[2]/(rs[0]+rs[1])
    f1  = (2*rs[0])/((2*rs[0])+rs[1]+rs[3])
    return [acc,ppv,npv,sen,spe,f1]

def multiclass_auc(y_test, y_score, n_classes=4):
    roc_auc_hour = defaultdict(list)
    for h in range(y_score.shape[1]): 
        for i in range(n_classes):
            fprh, tprh, _ = skm.roc_curve(y_test[:,h]==i, y_score[:,h,i])
            roc_auc_hour[i].append(skm.auc(fprh, tprh))
            
    roc_auc = {}
    for i in range(n_classes):
        fprh, tprh, _ = skm.roc_curve(y_test[:,:].flatten()==i, y_score[:,:,i].flatten())
        roc_auc[i] = skm.auc(fprh, tprh)
        
#     roc_auc['avg'] = sum(roc_auc.values())/n_classes
    return roc_auc, roc_auc_hour

def saveMetrics(tru, pre, outfn):
    rsh = runMetricsHour(tru, pre)
    metrich = [pd.DataFrame(np.asarray(list(map(calcScores, rsh[i].T))), columns=['acc','ppv','npv','sen','spe','f1']).transpose() for i in range(4)]
    metrich = pd.concat(metrich, keys=['0','1','2','3'])

    rs = runMetrics(tru, pre)
    metrics = None
    for c in range(4):
        if metrics is None:
            metrics = calcScores(rs[c])
        else:
            metrics += calcScores(rs[c])

    metrich['overall'] = metrics
    metrich.to_csv(outfn, na_rep='nan')

def saveAUCs(tru, pro, outfn, nclasses=4):
    sup = []
    for c in range(nclasses):
        sup.append(np.sum(tru==c))
    sup.append(np.nan)
    aucs, auch = multiclass_auc(tru, pro)
    auch = pd.DataFrame(auch)
    auch['hourmean'] = auch.apply(np.mean, axis=1)
    aucs = pd.Series(aucs, name='overall')
    auch = auch.append(aucs)
    aucht = auch.transpose()
    aucht.insert(0, 'SUPPORT', sup)
    aucht.to_csv(outfn, na_rep='nan')


if __name__ == "__main__":
    db = DataBatch('/media/ram/chpc/AMIA2019/data/bypt', batchSize=100)
    print('Filenames loaded')
    b = db.getBatchIterator()
    print('Got interator')
    d = next(iter(b))
    labelCol = 'STAGE6'
    labelStart = -12
    tst_ids, tst_X, tst_Y, _ = prepareData(d, db.getHeaders(), nclasses=4, labelCol=labelCol, labelStart=labelStart)
    print('Got Batch')
    print(tst_X.shape)
    print(np.sum(np.isnan(tst_X)))
    print(np.sum(np.isnan(tst_Y)))