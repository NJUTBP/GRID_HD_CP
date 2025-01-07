#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 21:26:26 2023

@author: qianruixin
"""

import os
import numpy as np
from matplotlib import pyplot as plt

mi=np.load('./Mouse12-120806_REM_hd_mi.npy',allow_pickle=True).item()
mi_shuffle=np.load('./Mouse12-120806_REM_hd_mi_shuffle.npy',allow_pickle=True).item()

cs=list(mi.keys())
for c in range(len(cs)):
    plt.figure()
    plt.hist(mi_shuffle[cs[c]],alpha=0.4)
    plt.axvline(mi[cs[c]])

