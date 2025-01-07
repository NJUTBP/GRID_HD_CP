#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 23:23:17 2024

@author: qianruixin
"""

import csv
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('white')
  
import sys
sys.path.append('..') # 将上级目录加入 sys.path
from DrawStandard import *

def triu(a):
    # Extract the upper triangular elements of the matrix
    triulist = []
    for a1 in range(len(a)):
        for a2 in np.arange(a1+1, len(a)):
            triulist.append(a[a1, a2])
    return triulist


def main():
    fig=plt.figure(figsize=(8.5,11))

    prefix = '../Data/SimulatedData/'
    csv_reader = csv.reader(open(prefix + '/population/activity_csv/activity_dir.csv'))
    hd_cords = np.load(prefix + "/CoordDistHD.npz")['Coord']            
    Dist = np.load(prefix + "/CoordDistHD.npz")['Dist']     
    Rs = np.load(prefix + "/RsHD.npz")['Rs'] 
    Dist = Dist[:, :100][:100, :]
    Rs = Rs[:, :100][:100, :]
    d2 = []
    r2 = []
    for n1 in range(len(Dist)):
        for n2 in np.arange(n1+1, len(Dist)):
            d2.append(Dist[n1, n2])
            r2.append(Rs[n1, n2])


    IDs = ['Mouse12-120807']
    Clus = ['1_8']
    States = ['Wake']
    FilePath = IDs[0]
    State = States[0]

    prefix_in = '../Data/HD/'
    path1 = '../Data/HD//Rs_align_noise_supp/'

    Clus_min = Clus[0].split('_')[0]
    Clus_max = Clus[0].split('_')[1]

    AllNeu = np.load(prefix_in + FilePath+'/'+FilePath+'_Neu_Clu'+Clus_min+'to'+Clus_max+'.npz')['AllNeu']
    SortedFineHDIndex = np.load(prefix_in + FilePath+'/'+FilePath+'_Neu_Clu'+Clus_min+'to'+Clus_max+'.npz')['SortedFineHDIndex']
    NoneHDIndex = np.load(prefix_in + FilePath+'/'+FilePath+'_Neu_Clu'+Clus_min+'to'+Clus_max+'.npz')['NoneHDIndex']
    FinalIndex = np.append(SortedFineHDIndex, NoneHDIndex)
    num_hd = len(SortedFineHDIndex)
    num_non = len(NoneHDIndex)

    Rs = np.load(prefix_in + FilePath+'/'+State+'/LagCM_AllFullPeriod_Clu'+Clus_min+'to'+Clus_max+'.npz',allow_pickle = True)['Rs']
    LagRs = np.load(prefix_in + FilePath+'/'+State+'/LagCM_AllFullPeriod_Clu'+Clus_min+'to'+Clus_max+'.npz',allow_pickle = True)['LagRs'] #5s
    Rs = Rs[FinalIndex,:][:,FinalIndex]
    LagRs = LagRs[FinalIndex,:][:,FinalIndex]
    HDRs = Rs[:, :num_hd][:num_hd, :]

    data = np.load(prefix_in+'/'+FilePath+'/Wake/RateOrien_Clu'+Clus_min+'to'+Clus_max+'.npz',allow_pickle = True)
    ori = data['PreferOrien'].item() 
    oris = []
    for n in range(num_hd): 
        oris.append(ori[AllNeu[SortedFineHDIndex[n]]])
    Dist=np.zeros((num_hd,num_hd))
    for n1 in range(num_hd):
        for n2 in np.arange(n1+1,num_hd):
            ori1=ori[AllNeu[SortedFineHDIndex[n1]]]
            ori2=ori[AllNeu[SortedFineHDIndex[n2]]]
            Dist[n1,n2]=np.sqrt(np.sum(np.square(np.arctan2(np.sin(ori1 - ori2),
                              np.cos(ori1 - ori2)))))
            Dist[n2,n1]=Dist[n1,n2]

    r1 = []
    d1 = []
    for n1 in range(num_hd):
        for n2 in np.arange(n1+1, num_hd):        
            r1.append(HDRs[n1, n2])
            d1.append(Dist[n1, n2])    
    rs = []
    for f in np.arange(1,10.001,0.1):
        Rssim_noise = np.load(path1+'RsHDNoise_{:.1f}.npy'.format(f), allow_pickle=True)
        r3 = []
        for n1 in range((num_hd)): #100
            for n2 in np.arange(n1+1, (num_hd)):
                r3.append(Rssim_noise[n1, n2])
        hist_r3, bins = np.histogram(r3, bins=100)
        hist_r1, _ = np.histogram(r1, bins=bins)
        # hist_r1, bins = np.histogram(r1, bins=100)
        # hist_r3, _ = np.histogram(r3, bins=bins)
        rs.append(np.corrcoef(hist_r1, hist_r3)[0, 1])
    idx = np.argsort(rs)[-1]
    f = np.arange(1,10.001,0.1)[idx]
    Rssim_noise = np.load(path1+'RsHDNoise_{:.1f}.npy'.format(f), allow_pickle=True)
    r3 = []
    for n1 in range(num_hd):
        for n2 in np.arange(n1+1, num_hd):
            r3.append(Rssim_noise[n1, n2])
            
      
    f = np.arange(0.1,10.001,0.1)
    f1_idx = 0
    f3_idx = 4
    f5_idx = 9
    f_cs=['darkorange', 'lightblue', 'steelblue']

    ax1=fig.add_axes([0.25,0.7,0.2,0.15]) 
    # sns.kdeplot(np.array(r1), cut=0, c='royalblue', label='Experiment')  
    # sns.kdeplot(np.array(r2), cut=0, c='pink', label='Simulation')  
    # sns.kdeplot(np.array(r3), cut=0, c='orange', label='Simulation + Noise (F = {:.1f})'.format(f))  

    sns.kdeplot(np.array(r1), c='purple', label=r'one session in $\it{HD}$',cut=0)  
    sns.kdeplot(np.array(r2), c='pink', label='Simulation',cut=0)  

    for no,f_idx in enumerate([f1_idx,f3_idx,f5_idx]):
        Rssim_noise = np.load(path1+'RsHDNoise_{:.1f}.npy'.format(f[f_idx]), allow_pickle=True)
        r3 = []
        for n1 in range(num_hd):
            for n2 in np.arange(n1+1, num_hd):
                r3.append(Rssim_noise[n1, n2])   
        sns.kdeplot(np.array(r3), c=f_cs[no], label='Simulation + Noise (F = {:.1f})'.format(f[f_idx]),cut=0)  
    plt.legend(fontsize='small') 
    plt.xlabel('Correlation coefficient',labelpad=1)  
    plt.ylabel('PDF',labelpad=0.1)

    ax2=fig.add_axes([0.50,0.7,0.2,0.15])
    # sns.kdeplot(np.array(d1), cut=0, c='royalblue', label='Experiment')  
    # sns.kdeplot(np.array(d2), cut=0, c='pink', label='Simulation')  

    sns.kdeplot(np.array(d1), c='grey', label=r'one session in $\it{HD}$',cut=0)  
    sns.kdeplot(np.array(d2), c='black', label='Simulation',cut=0)  
    plt.legend()
    plt.xlabel('$\\Delta \\theta$ (rad)',labelpad=1 )  
    plt.ylabel('PDF',labelpad=0.1 )
    ax2.set_yticks([0.15,0.20,0.25,0.25,0.30])

    prefix = '../Data/SimulatedData/'
    path1 = '../Data/GC3/Rs_align_noise_supp/'
    file_path='../Data/GC3/'
    csv_reader = csv.reader(open(prefix + '/population/activity_csv/activity_grid.csv'))

    cords = np.load(prefix + "/CoordDist.npz")['Coord']            
    Dist = np.load(prefix + "/CoordDist.npz")['Dist']     
    Rs_sim = np.load(prefix + "/Rs.npz")['Rs'] 
    numGC = 300
    Dist = Dist[:, :numGC][:numGC, :]
    Rs = Rs_sim[:, :numGC][:numGC, :]
    d2 = []
    r2 = []
    for n1 in range(len(Dist)):
        for n2 in np.arange(n1+1, len(Dist)):
            d2.append(Dist[n1, n2])
            r2.append(Rs[n1, n2])
    rat_name='Q'
    mod_name='1'
    sess_name='OF'
    day_name=''
    file_name =   rat_name + '_' + mod_name + '_' + sess_name  
    if len(day_name)>0:
        file_name += '_' + day_name 
    Rs_exp = np.load(file_path + '/Results/' + file_name + '_Rs.npz')['Rs']
    num_neurons = len(Rs_exp)
    file_name =   rat_name + '_' + mod_name + '_' + sess_name + '_OF' 
    if len(day_name)>0:
        file_name += '_' + day_name
    Dist_exp = np.load(file_path + '/results/' + file_name + '_dist.npz')['dist']
    # Rs_exp = np.corrcoef(binned_spikes_matrix)  
    num_neurons_sele=num_neurons
    # num_neurons_sele=40
    Rs_exp = Rs_exp[:, :num_neurons_sele][:num_neurons_sele, :]
    Dist_exp = Dist_exp[:, :num_neurons_sele][:num_neurons_sele, :]
      
    r1 = triu(Rs_exp)
    d1 = triu(Dist_exp)

    f = np.arange(1,10.001,0.1)
    f1_idx = 0
    f3_idx = int(3/0.1-10)
    f5_idx = int(5/0.1-10)
    f_cs=['darkorange', 'lightblue', 'steelblue']
    ax3=fig.add_axes([0.25,0.50,0.20,0.15])
    # sns.kdeplot(np.array(r1), cut=0, c='royalblue', label='Experiment')  
    # sns.kdeplot(np.array(r2), cut=0, c='pink', label='Simulation')  
    # sns.kdeplot(np.array(r3), cut=0, c='orange', label='Simulation + Noise (F = {:.1f})'.format(f))  

    sns.kdeplot(np.array(r1), c='purple', label=r'one session in $\it{GC-3}$',cut=0)  
    sns.kdeplot(np.array(r2), c='pink', label='Simulation',cut=0)  

    for no,f_idx in enumerate([f1_idx,f3_idx,f5_idx]):
        Rssim_noise = np.load(path1+'RsGCNoise_{:.1f}.npy'.format(f[f_idx]), allow_pickle=True)
        r3 = []
        for n1 in range(numGC):
            for n2 in np.arange(n1+1, len(Dist)):
                r3.append(Rssim_noise[n1, n2])   
        sns.kdeplot(np.array(r3), c=f_cs[no], label='Simulation + Noise (F = {:.1f})'.format(f[f_idx]),cut=0)  
    plt.legend(fontsize='small') 
    plt.xlabel('Correlation coefficient',labelpad=1)  
    plt.ylabel('PDF',labelpad=0.1)

    ax4=fig.add_axes([0.50,0.50,0.2,0.15])
    # sns.kdeplot(np.array(d1), cut=0, c='royalblue', label='Experiment')  
    # sns.kdeplot(np.array(d2), cut=0, c='pink', label='Simulation')  

    sns.kdeplot(np.array(d1), c='grey', label=r'one session in $\it{GC-3}$',cut=0)  
    sns.kdeplot(np.array(d2), c='black', label='Simulation',cut=0)  
    plt.legend()
    plt.xlabel('$|\\Delta \\bf{g}|$ (rad)',labelpad=1 )  
    plt.ylabel('PDF',labelpad=0.1 )   

    ax1_pos=ax1.get_position()
    ax2_pos=ax2.get_position()
    ax3_pos=ax3.get_position()
    ax4_pos=ax4.get_position()

    plt.figtext(ax1_pos.x0-0.03,ax1_pos.y1+0.01,'A',fontsize=15)
    plt.figtext(ax2_pos.x0-0.03,ax2_pos.y1+0.01,'B',fontsize=15)
    plt.figtext(ax3_pos.x0-0.03,ax3_pos.y1+0.01,'C',fontsize=15)
    plt.figtext(ax4_pos.x0-0.03,ax4_pos.y1+0.01,'D',fontsize=15)

    plt.savefig('../Figures/SuppFigure14.png',format='PNG',dpi=300)
    plt.show()
    plt.close()

if __name__ == '__main__':
    main() 
