from zipfile import ZipFile
import numpy as np
from sklearn.utils import shuffle
import os, pickle
from constants import RAW_PATH, DATA_PATH 
# zf = ZipFile('data/bypt500.zip')
# # zf.printdir()

# for n in zf.namelist():
#     with zf.open(n) as f:
#         print(n)
#         d = np.genfromtxt(f, skip_header=1, delimiter=',', missing_values=['?', '!'])
#         print(d)
#     break


def getSplitIds(idsPath):
    split_ids = pickle.load(open(idsPath, 'rb'))
    for subset in split_ids:
        split_ids[subset] = set(zip(split_ids[subset][:,0], split_ids[subset][:,1]))
    return split_ids
    
def getData(batch, zippath):
    d = lambda fn: np.genfromtxt(fn, skip_header=1, delimiter=',', missing_values=['?', '!'])[:,:,np.newaxis]
    
    data = None
    with ZipFile(zippath) as zf:
        for fn in batch:
            with zf.open(fn) as f:
                if data is None:
                    data = d(f)
                else:
                    pt = d(f)
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
        return getData(self.files[start:end], self.path)
          

class DataBatch():
    def __init__(self, zippath, split_ids=None, batchSize=1000):
        if batchSize <= 0:
            raise ValueError('batchSize must be > 0')
        self.path = zippath
        self.batchSize = batchSize
        self.files = {}

        with ZipFile(zippath) as zf:
            if split_ids != None:
                for fn in zf.namelist():
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
                self.files['all'] = [f for f in zf.namelist() if not f.startswith('.')]
                        
    def getBatchIterator(self, key='all'):
        return BatchIterator(self.path, self.files[key], self.batchSize)

    def getHeaders(self):
        fn = os.path.join(self.path, next(iter(self.files.values()))[0])
        return np.genfromtxt(fn, max_rows=1, delimiter=',', missing_values=['?', '!'], dtype=str)


if __name__ == "__main__":

    batchSize = 3
    idsPath = os.path.join(RAW_PATH, 'd_ids_split.pickle')
    datapath = os.path.join(DATA_PATH, 'bypt500.zip')

    split_ids = getSplitIds(idsPath)
    db = DataBatch(datapath, split_ids, batchSize=batchSize)
    print([(k, len(v)) for k, v in db.files.items()])

    trainBatches = db.getBatchIterator('devel')

    print(next(iter(trainBatches)))