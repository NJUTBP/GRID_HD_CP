
import cpnet
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt   
from pandas import  DataFrame
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

import sys
sys.path.append('../') # 将上级目录加入 sys.path
from DetectCP_max import GenerateGraph, CalCoreness, Metrics, DrawNetwork
from DrawStandard import *
width = 100 / 25.4 

from sklearn.metrics import confusion_matrix

import pickle

def triu(a):
    # Extract the upper triangular elements of the matrix
    triulist = []
    for a1 in range(len(a)):
        for a2 in np.arange(a1+1, len(a)):
            triulist.append(a[a1, a2])
    return triulist

prefix_in = '../../Dataset/HD/ProcessedData/'
pathout = '../../Dataset/HD/ProcessedData/DetectCP_max/'

IDs = ['Mouse12-120806', 'Mouse12-120807', 'Mouse12-120808', 'Mouse12-120809', 'Mouse12-120810',
      'Mouse20-130517', 'Mouse28-140313', 'Mouse25-140130']
Clus = ['1_8', '1_8', '1_8', '1_8', '1_8', '1_8', '8_11', '5_8']
States = ['Wake', 'REM', 'SWS']
# IDs = ['Mouse12-120806']
# Clus = ['1_8']
# States = ['REM']

aucs=np.load("../../Dataset/HDAUC.npz")['aus']
pop_nums=[]

for i in range(len(IDs)):
    for j in range(len(States)):
# for i in range(1):
#     for j in range(1):
        FilePath = IDs[i]
        State = States[j]

        Clus_min = Clus[i].split('_')[0]
        Clus_max = Clus[i].split('_')[1]
        
        AllNeu = np.load(prefix_in + FilePath+'/'+FilePath+'_Neu_Clu'+Clus_min+'to'+Clus_max+'.npz')['AllNeu']
        SortedFineHDIndex = np.load(prefix_in + FilePath+'/'+FilePath+'_Neu_Clu'+Clus_min+'to'+Clus_max+'.npz')['SortedFineHDIndex']
        NoneHDIndex = np.load(prefix_in + FilePath+'/'+FilePath+'_Neu_Clu'+Clus_min+'to'+Clus_max+'.npz')['NoneHDIndex']
        FinalIndex = np.append(SortedFineHDIndex, NoneHDIndex)
        
        pop_nums.append(len(FinalIndex))

plt.figure()
plt.scatter(pop_nums,aucs)

np.savez("./PoPNumAUC_HD", aus=aucs, pop_nums=pop_nums)
        