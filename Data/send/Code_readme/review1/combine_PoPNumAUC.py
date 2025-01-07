#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 20:29:08 2024

@author: qianruixin
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
#HD
loaded_data = np.load("./PoPNumAUC_HD.npz")
aucs_HD = loaded_data['aus']
pop_nums_HD = loaded_data['pop_nums']

#GC1
loaded_data = np.load("./PoPNumAUC_GC1.npz")
aucs_GC1 = loaded_data['aus']
pop_nums_GC1 = loaded_data['pop_nums']

#GC2
loaded_data = np.load("./PoPNumAUC_GC2.npz")
aucs_GC2 = loaded_data['aus']
pop_nums_GC2 = loaded_data['pop_nums']


plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.scatter(pop_nums_HD, aucs_HD)
plt.xlabel('The number of mixed population', fontsize=8)
plt.ylabel('AUC', fontsize=8)
plt.title(r'$\it{HD}$'+'  p>0.05', fontsize=8)

plt.subplot(132)
plt.scatter(pop_nums_GC1, aucs_GC1)
plt.xlabel('The number of mixed population', fontsize=8)
plt.ylabel('AUC', fontsize=8)
plt.title(r'$\it{GC-1}$'+'  p>0.05', fontsize=8)

plt.subplot(133)
plt.scatter(pop_nums_GC2, aucs_GC2)
plt.xlabel('The number of mixed population', fontsize=8)
plt.ylabel('AUC', fontsize=8)
plt.title(r'$\it{GC-2}$'+'  p>0.05', fontsize=8)

plt.tight_layout()
plt.savefig('./combine_PoPNumAUC', dpi=400)


# 计算线性相关系数和 p 值
corr_hd, p_hd = pearsonr(pop_nums_HD, aucs_HD)
corr_gc1, p_gc1 = pearsonr(pop_nums_GC1, aucs_GC1)
corr_gc2, p_gc2 = pearsonr(pop_nums_GC2, aucs_GC2)

# 输出结果
print(f"HD: Correlation coefficient = {corr_hd}, p-value = {p_hd}")
print(f"GC-1: Correlation coefficient = {corr_gc1}, p-value = {p_gc1}")
print(f"GC-2: Correlation coefficient = {corr_gc2}, p-value = {p_gc2}")
