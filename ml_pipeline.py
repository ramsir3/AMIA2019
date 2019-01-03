
# coding: utf-8

# In[55]:


import os.path
import numpy as np
import scipy as sp
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[2]:


from sklearn.utils import shuffle
from sklearn.preprocessing import normalize, OneHotEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics as skm


# In[3]:


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


# In[61]:


datafn = 'HOUR_00001.csv'


# In[62]:


data = pd.read_csv(os.path.join('~/Projects/Research/data/processed/', datafn), na_values='?')


# In[63]:


data.head()


# In[64]:


# drop rows where all features=nan
row_nan_bool = np.logical_not(np.all(np.isnan(data.iloc[:,5:-1]), axis=1))
data2 = data[row_nan_bool]


# In[65]:


#drop columns where all rows=nan
check_nan = data2.isna().sum()
data2.drop(labels=check_nan[(check_nan == data2.shape[0])].keys(), axis=1, inplace=True)


# In[66]:


data3 = data2.iloc[:,3:-1]


# In[67]:


# calculate Kruskal-Wallis H-test for each feature
dfs_by_class = [data3.loc[data2['STAGE'] == c] for c in [0,1,2,3]]
kruskals = {}
for col in data3:
    kruskals[col] = sp.stats.kruskal(*[np.asarray(c[col].dropna()) for c in dfs_by_class])[1]
    
for k, v in kruskals.items():
    if v > 0.05:
        print(k, v)


# In[80]:


meanImputeData = data2.fillna(data.mean())
meanImputeData.head()


# In[84]:


meanImputeFeatures = meanImputeData.iloc[:,3:-1]


# In[93]:


# calculate VIFs
done = -1
while done != 0:
    vifs = {}
    for i, n in enumerate(meanImputeData):
        if i in range(3,meanImputeData.shape[1]):
            vifs[n] = variance_inflation_factor(np.asarray(meanImputeData), i)

    drop_vifs = [k for k,v in vifs.items() if v >= 5 or np.isnan(v)]
    print(drop_vifs)
    meanImputeData.drop(labels=drop_vifs, axis=1, inplace=True)
    done = len(drop_vifs)


# In[94]:


meanImputeData


# In[97]:


meanImputeDataArray = meanImputeData.values[:, 3:]
x = meanImputeDataArray[:, :-1]
y = meanImputeDataArray[:, -1]
x = normalize(x, axis=0)
ohe = LabelBinarizer()
ohe.fit(y.reshape(-1, 1))
# print(x.shape)
# print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
print(x_train.shape)
print(x_test.shape)
ohe_y_train = ohe.transform(y_train.reshape(-1,1))
ohe_y_test = ohe.transform(y_test.reshape(-1,1))
print(ohe_y_train.shape)
print(ohe_y_test.shape)


# In[11]:


def runModel(model, x, y, x_test, y_test):
    m = OneVsRestClassifier(model)
    m.fit(x,y)
    y_predict = m.predict(x_test)
    
    return m, y_predict


# In[12]:


def runMetrics(y_test, y_predict):
    TP = {}
    FP = {}
    TN = {}
    FN = {}

    for i in range(y_predict.shape[1]): 
        TP[i] = np.sum(np.logical_and(y_predict[:,i]==1,y_test[:,i]==1))
        FP[i] = np.sum(np.logical_and(y_predict[:,i]==1,y_test[:,i]!=y_predict[:,i]))
        TN[i] = np.sum(np.logical_and(y_predict[:,i]==0,y_test[:,i]==0))
        FN[i] = np.sum(np.logical_and(y_predict[:,i]==0,y_test[:,i]!=y_predict[:,i]))
    
    out = {}
    for i in range(y_predict.shape[1]): 
        out[i] = [TP[i], FP[i], TN[i], FN[i]]
    return out


# In[13]:


def multiclass_auc(y_test, y_score):
    print('multiclass_auc')
    n_classes = 4
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = skm.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = skm.auc(fpr[i], tpr[i])
    roc_auc['avg'] = sum(roc_auc.values())/n_classes
    return roc_auc


# In[14]:


models1 = {
    'SVC': SVC(),
    'SGDClassifier': SGDClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'GaussianNB': GaussianNB(),
    'KNeighborsClassifier': KNeighborsClassifier(),
}
models2 = {
    'GaussianProcessClassifier': GaussianProcessClassifier(),
    'MLPClassifier': MLPClassifier(),
}


# In[15]:


rawScores = {}
accScores = {}
for n, m in models1.items():
    print(n)
    m, y_predict = runModel(m, x_train, ohe_y_train, x_test, ohe_y_test)
    accScores[n] = skm.accuracy_score(ohe_y_test, y_predict)
    rawScores[n] = runMetrics(ohe_y_test, y_predict)
    if n in {'SVC', 'SGDClassifier', 'GradientBoostingClassifier'}:
        aucs = multiclass_auc(ohe_y_test, m.decision_function(x_test))
    else:
        aucs = multiclass_auc(ohe_y_test, m.predict_proba(x_test))
        
    for k in rawScores[n].keys():
        rawScores[n][k].append(aucs[k])
    
for n, m in models2.items():
    print(n)
    m, y_predict = runModel(m, x_train, ohe_y_train, x_test, ohe_y_test)
    accScores[n] = skm.accuracy_score(ohe_y_test, y_predict)
    rawScores[n] = runMetrics(ohe_y_test, y_predict)
    if n in {'SVC', 'SGDClassifier', 'GradientBoostingClassifier'}:
        aucs = multiclass_auc(ohe_y_test, m.decision_function(x_test))
    else:
        aucs = multiclass_auc(ohe_y_test, m.predict_proba(x_test))
        
    for k in rawScores[n].keys():
        rawScores[n][k].append(aucs[k])

# In[16]:


rawScores


# In[ ]:


with open("RAW_OUTPUT_"+datafn, 'w') as fout:
    header = ',' + ',,,,'.join("CLASS %d (%d)" % (c, np.sum(ohe_y_test[:,c])) for c in [0,1,2,3]) + '\n'
    header+= 'MODEL,' + ','.join("TP,FP,TN,FN" for c in [0,1,2,3]) + '\n'
    fout.write(header)
    for m in rawScores:
        fout.write(m+',')
        for c in rawScores[m]:
            for i in rawScores[m][c][:-1]:
                fout.write(str(i)+',')
        fout.write('\n')


# In[ ]:


# AUC,PPV,NPV,SEN,SPE,F1
# TP,FP,TN,FN
def calcScores(rs):
#     print('?',rs)
    auc = rs[4]
    ppv = rs[0]/(rs[0]+rs[1])
    npv = rs[2]/(rs[2]+rs[3])
    sen = rs[0]/(rs[0]+rs[3])
    spe = rs[2]/(rs[0]+rs[1])
    f1  = (2*rs[0])/((2*rs[0])+rs[1]+rs[3])
    return (auc,ppv,npv,sen,spe,f1)


# In[ ]:


m = 'SVC'
calcedScores = {k:calcScores(rawScores[m][k]) for k in rawScores[m]}


# In[ ]:


calcedScores


# In[ ]:


with open("OUTPUT_"+datafn, 'w') as fout:
    header = ',AVG (%d),,,,,,' % ohe_y_test.shape[0] + ',,,,,,'.join("CLASS %d (%d)" % (c, np.sum(ohe_y_test[:,c])) for c in [0,1,2,3]) + '\n'
    header+= 'MODEL,ACC,AUC,PPV,NPV,SEN,SPE,F1,' + ','.join("AUC,PPV,NPV,SEN,SPE,F1" for c in [0,1,2,3]) + '\n'
    fout.write(header)
    for m in rawScores:
        calcedScores = [calcScores(rawScores[m][k]) for k in rawScores[m]]
        avgScores = [0 for i in calcedScores[0]]
        for i, c in enumerate(calcedScores):
            for j, _ in enumerate(c):
                avgScores[j] += calcedScores[i][j]
        for i, v in enumerate(avgScores):
            avgScores[i] = v/len(calcedScores)       
        
        fout.write(m+',')
        fout.write(str(accScores[m])+',')
        for v in avgScores:
            fout.write(str(v)+',')
        for c in calcedScores:
            for i in c:
                fout.write(str(i)+',')
        fout.write('\n')

