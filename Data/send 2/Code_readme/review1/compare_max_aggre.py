#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:35:29 2024

@author: qianruixin
"""

import pickle
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

#HD
path_HD1 = '../../Dataset/HD/ProcessedData/DetectCP/'
path_HD2 = '../../Dataset/HD/ProcessedData/DetectCP_max/'

IDs = ['Mouse12-120806', 'Mouse12-120807', 'Mouse12-120808', 'Mouse12-120809', 'Mouse12-120810',
      'Mouse20-130517', 'Mouse28-140313', 'Mouse25-140130']
Clus = ['1_8', '1_8', '1_8', '1_8', '1_8', '1_8', '8_11', '5_8']
States = ['Wake', 'REM', 'SWS']

# HD在文章中的例子
m = 'Mouse12-120806'
# Clus = ['1_8']
s = 'REM'
with open(path_HD1+'/output.pickle', 'rb') as file:
    data = pickle.load(file)
HD_scores_aggre=list(data['xlist_w_dict'].values())
with open(path_HD2+m+'_'+s+'.pickle', 'rb') as file:
    data = pickle.load(file)
HD_scores_max=list(data['xlist_w_dict'].values())   

r, p_value = scipy.stats.pearsonr(HD_scores_aggre, HD_scores_max)

plt.figure()
plt.scatter(HD_scores_aggre,HD_scores_max)
