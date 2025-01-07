#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 21:51:22 2023

@author: qianruixin
"""


####加载库####
from tqdm import tqdm
import numpy as np
np.random.seed(20231230)
import pandas as pd
import matplotlib
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import colors as mpcolors
from matplotlib.cm import Blues
from matplotlib import cm  # Import the cm module
import matplotlib.ticker as mtick
from matplotlib.patches import Circle
from statannotations.Annotator import Annotator
from scipy import stats
from scipy.stats import pearsonr

import random
import networkx as nx
import sys
sys.path.append('../') # 将上级目录加入 sys.path
from DrawStandard import *
import pickle
from DetectCP import GenerateGraph, CalCoreness, Metrics, DrawNetwork
import seaborn as sns

from sklearn.linear_model import RANSACRegressor

sys.path.append('.')
from draw_diamond import *


font = {
        'family' : 'arial',#'monospace',#'Times New Roman',#
        #'weight' : 'bold',
        'size'   : 6,
}
mathtext = {
        'fontset' : 'dejavusans',#'stix',
}
lines = {
        'linewidth' : 1.5,
}
xtick = {
        'direction' : 'in',
        'major.size' : 2,
        'major.width' : 1,
        'minor.size' : 1,
        'minor.width' : 0.5,
        'labelsize' : 6,
}
ytick = {
        'direction' : 'in',
        'major.size' : 2,
        'major.width' : 1,
        'minor.size' : 1,
        'minor.width' : 0.5,
        'labelsize' : 6,
}
axes = {
        'linewidth' : 1,
        'titlesize' : 6,
        #'titleweight': 'bold',
        'labelsize' : 6,
        #'labelweight' : 'bold',
}

matplotlib.rc('font',**font)
matplotlib.rc('mathtext',**mathtext)
matplotlib.rc('lines',**lines)
matplotlib.rc('xtick',**xtick)
matplotlib.rc('ytick',**ytick)
matplotlib.rc('axes',**axes)

def F(z):
    
    return (abs(z)<1)*((1+np.cos(np.pi*z))/2)
    

l = 40
phi = 0
A = l*np.array([[np.cos(phi),np.cos(phi+np.pi/3)],[np.sin(phi),np.sin(phi+np.pi/3)]])
A_inv = np.linalg.inv(A)

def s_grid(x,b):
    #生成特定相位的网格细胞
    return F(np.linalg.norm(A@(np.mod(A_inv@x-b[:,None],1)-0.5),axis=0)/(0.45*l))

def orientation_change(grid_x,grid_y):
    shape = grid_x.shape[0]
    grid_all = np.concatenate([grid_x[...,None],grid_y[...,None]],axis=2)
    grid_all = grid_all.reshape(grid_all.shape[0]*grid_all.shape[1],2)
    grid_change = grid_all@A
    grid_change = grid_change.reshape(shape,shape,2)
    
    return grid_change


def main():
    fig=plt.figure(figsize=(8.5,11))

    pathout='./'
    res = 40
    num = res+1
    X = np.linspace(-res,res,num)
    Y = np.linspace(-res,res,num)
    grid_x,grid_y = np.meshgrid(X,Y)

    orientation_x = np.linspace(-0.5,0.5,10)
    orientation_y = np.linspace(-0.5,0.5,10)
    grid_orientation_x,grid_orientation_y = np.meshgrid(orientation_x,orientation_y)


    # l = 40
    # phi = 0
    # A = l*np.array([[np.cos(phi),np.cos(phi+np.pi/3)],[np.sin(phi),np.sin(phi+np.pi/3)]])
    # A_inv = np.linalg.inv(A)

    # 如何生成一个网格细胞的放电图
    # plt.figure()
    # plt.subplot(1,1,1)
    b = np.array([-0.5,-0.5])
    grid_all = np.concatenate([grid_x[...,None],grid_y[...,None]],axis=2)
    grid_all = grid_all.reshape(grid_all.shape[0]*grid_all.shape[1],2)
    grid_value = s_grid(grid_all.T,b)
    # plt.scatter(grid_x,grid_y,c = grid_value)
    # plt.show()

    # 接下来的distance和角度都是以这个baseneuron为圆心 ，左下角？
    base = s_grid(grid_all.T,np.array([-0.5,-0.5]))
    base = (base-base.mean())/base.std()

    # 在一个菱形圆包内生成100个神经元的放电图
    nums = 100

    base = s_grid(grid_all.T,np.array([-0.5,-0.5]))
    base = (base-base.mean())/base.std()

    orientation_x = np.linspace(-0.5,0.5,nums)
    orientation_y = np.linspace(-0.5,0.5,nums)
    grid_orientation_x,grid_orientation_y = np.meshgrid(orientation_x,orientation_y)
    values = np.zeros((nums,nums))
    for i in tqdm(range(nums)):
        for j in range(nums):
            tmp_value = s_grid(grid_all.T,np.array([grid_orientation_x[i][j],grid_orientation_y[i][j]]))
            tmp_value = (tmp_value-tmp_value.mean())/tmp_value.std()
            values[i][j] = np.dot(base,tmp_value)/base.shape[0]
            
    Dist = np.zeros_like(values)
    Dist1 = np.zeros_like(values)
    grid_orientation_x_real,grid_orientation_y_real = grid_orientation_x*2*np.pi,grid_orientation_y*2*np.pi
    ori2 = np.array([grid_orientation_x_real[0][0],grid_orientation_y_real[0][0]])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ori1 =  np.array([grid_orientation_x_real[i][j],grid_orientation_y_real[i][j]])
            Dist[i,j]=np.sqrt(np.sum(np.square(np.arctan2(np.sin(ori1 - ori2),
                                          np.cos(ori1 - ori2)))))
            Dist1[i,j]=np.sqrt(np.sum(np.square(ori1-ori2)))
    # plt.figure()
    #plt.scatter(grid_orientation_x*2*np.pi+np.pi,grid_orientation_y*2*np.pi+np.pi,c = values)
    ax1=fig.add_axes([0.05,0.90,0.08,0.08])
    grid_change = orientation_change(grid_orientation_x,grid_orientation_y)
    sc1=ax1.scatter(grid_change[:,:,0],grid_change[:,:,1],c = values,s=1,cmap='jet')
    axcb1=fig.add_axes([0.14,0.90,0.01,0.08])
    fig.colorbar(sc1,cax=axcb1)
    ax1.set_title('Correlation between rate maps')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')
    ax2=fig.add_axes([0.20,0.90,0.08,0.08])
    axcb2=fig.add_axes([0.29,0.90,0.01,0.08])
    sc2=ax2.scatter(grid_change[:,:,0],grid_change[:,:,1],c = Dist,s=1,cmap='jet')
    fig.colorbar(sc2,cax=axcb2)
    # plt.title('Distance')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')
    # ax2.tight_layout()
    ax2.set_title('$\\Delta_\\mathrm{M}$ (rad)')
    # ax2.savefig(pathout+'corr_dist_ball', dpi=300, bbox_inches='tight') 
    # ax2.show()

    # 计算100个神经元相对于左下角神经元的角度
    angle = np.zeros((nums,nums))
    angle3 = np.zeros_like(angle)
    for i in range(nums):
        for j in range(nums):
            if grid_orientation_y[i,j]<0 and grid_orientation_x[i,j]<0:
                min_y = -0.5
                min_x = -0.5
                angle3[i][j] = np.arctan((grid_orientation_y[i,j]-min_y)/(grid_orientation_x[i,j]-min_x))
                angle3[i][j] *= 2/3
            if grid_orientation_y[i,j]>0 and grid_orientation_x[i,j]<0:
                min_y = 0.5
                min_x = -0.5
                angle3[i][j] = np.arctan((grid_orientation_y[i,j]-min_y)/(grid_orientation_x[i,j]-min_x))+2*np.pi
                angle3[i][j] = angle3[i][j]*4/3 -2*np.pi/3
                
            if grid_orientation_y[i,j]>0 and grid_orientation_x[i,j]>0:
                min_y = 0.5
                min_x = 0.5
                angle3[i][j] = np.arctan((grid_orientation_y[i,j]-min_y)/(grid_orientation_x[i,j]-min_x))+np.pi
                if angle3[i][j]<np.pi:
                    angle3[i][j] = 3*np.pi/2
                angle3[i][j] = angle3[i][j]*2/3 +np.pi/3
                
            if grid_orientation_y[i,j]<0 and grid_orientation_x[i,j]>0:
                min_y = -0.5
                min_x = 0.5
                angle3[i][j] = np.arctan((grid_orientation_y[i,j]-min_y)/(grid_orientation_x[i,j]-min_x))+np.pi
                if angle3[i][j]>np.pi:
                    angle3[i][j] = np.pi/2
                angle3[i][j] = angle3[i][j]*4/3 -np.pi/3
                
    # plt.figure(figsize=(np.sqrt(3)*2*1.2,3))
    
    ax3=fig.add_axes([0.35,0.90,0.08,0.08])
    axcb3=fig.add_axes([0.44,0.90,0.01,0.08])
    # plt.title('angle distribution')
    min_y = 0
    min_x = 0
    for i in range(nums):
        for j in range(nums):
            angle[i][j] = np.arctan((grid_change[:,:,1][i,j]-min_y)/(grid_change[:,:,0][i,j]-min_x))
            if grid_change[:,:,0][i,j]<0:
                angle[i][j]+=np.pi
    #angle[angle<0]+=np.pi
    sc3=ax3.scatter(grid_change[:,:,0],grid_change[:,:,1],c = angle3/np.pi*180,s=1,cmap='jet')
    fig.colorbar(sc3,cax=axcb3)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.axis('off')
    ax3.set_title('$\\Delta_\\mathrm{\\theta}$ ($^\circ$)')
    # plt.tight_layout()
    # plt.savefig(pathout+'corr_dist_ball', dpi=300, bbox_inches='tight') 
    # plt.show()

    ax1_pos=ax1.get_position()
    plt.figtext(ax1_pos.x0-0.05,ax1_pos.y1,'A',fontsize=15)


    # 固定的角度，r和dist的变化
    delta = 0.03
    tmp_angle = angle3.reshape(-1,1)
    tmp_dist = Dist.reshape(-1,1)
    tmp_value = values.reshape(-1,1)
    # sp_angle = (np.array([0,10,20,30,40,50,60,70,80]))*np.pi/180
    sp_angle = np.arange(0, 360, 10)*np.pi/180
    # plt.figure(figsize=(18,12))
    for i in range(len(sp_angle)):
        ax4=fig.add_axes([0.05+0.14*(i%6),0.83-0.07*int(i/6),0.12,0.05])
        
        if i==0:
            ax4_pos=ax4.get_position()
            plt.figtext(ax4_pos.x0-0.05,ax4_pos.y1,'B',fontsize=15)


        if int(i/6)==5:
            ax4.set_xlabel('$\\Delta_\\mathrm{M}$ (rad)',labelpad=0.1)

        if int(i%6)==0:
            ax4.set_ylabel('Corr. coef.',labelpad=0.1)

        # plt.subplot(6,6,i+1)
        pos_min = sp_angle[i-1]-delta<=tmp_angle
        pos_max = tmp_angle<=sp_angle[i-1]+delta
        idx = np.where(pos_min&pos_max)
        tmp_value_part = tmp_value[idx]
        #tmp_value_part = tmp_value_part[tmp_angle<=sp_angle[i-1]+delta]
        
        tmp_dist_part = tmp_dist[idx]
        # plt.scatter(tmp_value_part,tmp_dist_part)
        ax4.scatter(tmp_dist_part,tmp_value_part,s=5)
        ax4.set_title('$\\Delta_\\mathrm{\\theta}$'+' = ${}^{{\circ}}$'.format(round(sp_angle[i]*180/np.pi)),pad=0.1)
        # plt.tight_layout()
        ax4.set_xlim(0,4)
        ax4.set_ylim(-0.5,1.2)
    # plt.tight_layout()
    # plt.savefig(pathout+'angle_budong', dpi=300, bbox_inches='tight') 
    # plt.show()

    # plt.figure(figsize=(18,12))
    sp_dis = np.linspace(0.05,4.05,36)
    for i in range(1,37):
        ax5=fig.add_axes([0.05+0.14*((i-1)%6),0.39-0.07*int((i-1)/6),0.12,0.05])
        
        if i==1:
            ax5_pos=ax5.get_position()
            plt.figtext(ax5_pos.x0-0.05,ax5_pos.y1,'C',fontsize=15)

        if int((i-1)/6)==5:
            ax5.set_xlabel('$\\Delta_\\mathrm{\\theta}$ ($\degree$)',labelpad=0.1)

        if int((i-1)%6)==0:
            ax5.set_ylabel('Corr. coef.',labelpad=0.1)


        pos_min = sp_dis[i-1]-delta<=tmp_dist
        pos_max = tmp_dist<=sp_dis[i-1]+delta
        idx = np.where(pos_min&pos_max)
        tmp_value_part = tmp_value[idx]
        #tmp_value_part = tmp_value_part[tmp_angle<=sp_angle[i-1]+delta]
        
        tmp_angle_part = tmp_angle[idx]
        # plt.scatter(tmp_angle_part*180/np.pi,tmp_value_part)
        ax5.scatter(tmp_angle_part/np.pi*180,tmp_value_part,s=5)
        ax5.set_xticks([0, 180, 360])
        ax5.set_title('$\\Delta_\\mathrm{M}$'+' = {:.2f}  rad'.format(sp_dis[i-1]),pad=0.1)
        # plt.tight_layout()
        ax5.set_ylim(-0.5,1.2)
    # plt.tight_layout()

    plt.savefig('../Figures/SuppFigure13.png',format='PNG',dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()