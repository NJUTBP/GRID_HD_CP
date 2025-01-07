#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 22:17:21 2024

@author: qianruixin
"""

import pickle
import numpy as np

file_name='R_2_OF_day1_align.pickle'
# Open the file for reading in binary mode
with open(file_name, 'rb') as file:  # 'rb' -> read; 'b' -> binary
    # Load the content of the file into a variable
    data = pickle.load(file)
errs_pos2=data['errs_pos2']
errs_posr=data['errs_posr']




    # output['pos1']=pos1
    # output['pos2']=pos2
    # output['posrs']=posrs
    # output['errs_pos2']=errs_pos2
    # output['errs_posr']=errs_posr
    # output['dist2']=np.array(output['dist2'])[idx,:]
    # output['distr']=np.array(output['distr'])
    # import pickle
    # with open(pathout+file_name2+'_align.pickle', 'wb') as file: #w -> write; b -> binary
    #     pickle.dump(output, file)