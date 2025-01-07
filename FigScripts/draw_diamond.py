#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 20:25:18 2023

@author: qianruixin
"""
import numpy as np
from matplotlib import transforms
from matplotlib import pyplot as plt
from matplotlib.collections import PathCollection

def plot_phase_distribution(masscenters_1, c1, masscenters_2, c2, ax, s=8,):
    num_neurons = len(masscenters_1[:,0])
    for i in np.arange(num_neurons):
        ReshapeMat=np.mat([[1,1/2],[0,np.sqrt(3)/2]])
        masscenters_1_to_draw=np.zeros(masscenters_1.shape)
        masscenters_2_to_draw=np.zeros(masscenters_2.shape)
        masscenters_1_to_draw[i,:]=np.matmul(ReshapeMat,masscenters_1[i,:])
        masscenters_2_to_draw[i,:]=np.matmul(ReshapeMat,masscenters_2[i,:])
        
        ax.scatter(masscenters_1_to_draw[i,0], masscenters_1_to_draw[i,1], s = s, c = c1)
        ax.scatter(masscenters_2_to_draw[i,0], masscenters_2_to_draw[i,1], s = s, c = c2)
        line = masscenters_1[i,:] - masscenters_2[i,:]
        dline = line[1]/line[0]
        if line[0]< - np.pi and line[1] < -np.pi:
            line = (-2*np.pi + masscenters_2[i,:]) - masscenters_1[i,:]
            dline = line[1]/line[0]
            if (masscenters_1[i,1] + (- masscenters_1[i,0])*dline)>0:
                Start=np.matmul(ReshapeMat,np.array([masscenters_1[i,0], masscenters_1[i,1]]))
                End=np.matmul(ReshapeMat,np.array([0, masscenters_1[i,1] + (- masscenters_1[i,0])*dline]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([masscenters_1[i,0], 0],
                #         [masscenters_1[i,1], masscenters_1[i,1] + (- masscenters_1[i,0])*dline],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                
                Start=np.matmul(ReshapeMat,np.array([2*np.pi,masscenters_1[i,1] + (- masscenters_1[i,0])*dline]))
                End=np.matmul(ReshapeMat,np.array([2*np.pi + -(masscenters_1[i,1] + (- masscenters_1[i,0])*dline)/dline,0]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([2*np.pi,2*np.pi + -(masscenters_1[i,1] + (- masscenters_1[i,0])*dline)/dline], 
                #         [masscenters_1[i,1] + (- masscenters_1[i,0])*dline, 0],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                Start=np.matmul(ReshapeMat,np.array([2*np.pi + -(masscenters_1[i,1] + (- masscenters_1[i,0])*dline)/dline,2*np.pi]))
                End=np.matmul(ReshapeMat,np.array([masscenters_2[i,0],masscenters_2[i,1]]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([2*np.pi + -(masscenters_1[i,1] + (- masscenters_1[i,0])*dline)/dline, 
                #          masscenters_2[i,0]], 
                #         [2*np.pi,masscenters_2[i,1]],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
            else:
                Start=np.matmul(ReshapeMat,np.array([masscenters_1[i,0],masscenters_1[i,1]]))
                End=np.matmul(ReshapeMat,np.array([masscenters_1[i,0] + (- masscenters_1[i,1])/dline,0]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (- masscenters_1[i,1])/dline],
                #         [masscenters_1[i,1], 0],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                
                Start=np.matmul(ReshapeMat,np.array([masscenters_1[i,0] + (- masscenters_1[i,1])/dline,2*np.pi]))
                End=np.matmul(ReshapeMat,np.array([0,2*np.pi + -(masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([masscenters_1[i,0] + (- masscenters_1[i,1])/dline, 0],
                #         [2*np.pi, 2*np.pi + -(masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline], 
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                
                Start=np.matmul(ReshapeMat,np.array([2*np.pi,2*np.pi + -(masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline]))
                End=np.matmul(ReshapeMat,np.array([masscenters_2[i,0],masscenters_2[i,1]]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([2*np.pi, 
                #          masscenters_2[i,0]], 
                #         [2*np.pi + -(masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline,
                #         masscenters_2[i,1]],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
        elif line[0]> np.pi and line[1] >np.pi:
            line = (2*np.pi + masscenters_2[i,:]) - masscenters_1[i,:]
            dline = line[1]/line[0]
            if (masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline)<2*np.pi:
                Start=np.matmul(ReshapeMat,np.array([masscenters_1[i,0],masscenters_1[i,1]]))
                End=np.matmul(ReshapeMat,np.array([masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline,2*np.pi]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                       c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline],
                #        [masscenters_1[i,1],2*np.pi],
                #        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                Start=np.matmul(ReshapeMat,np.array([masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline,0]))
                End=np.matmul(ReshapeMat,np.array([2*np.pi,(2*np.pi - (masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline))*dline]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                       c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline, 2*np.pi],
                #        [0,(2*np.pi - (masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline))*dline], 
                #        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                Start=np.matmul(ReshapeMat,np.array([0,(2*np.pi - (masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline))*dline]))
                End=np.matmul(ReshapeMat,np.array([masscenters_2[i,0],masscenters_2[i,1]]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                       c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)  
                # ax.plot([0,masscenters_2[i,0]],
                #        [(2*np.pi - (masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline))*dline, 
                #         masscenters_2[i,1]], 
                #        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)          
            else:
                Start=np.matmul(ReshapeMat,np.array([masscenters_1[i,0],masscenters_1[i,1]]))
                End=np.matmul(ReshapeMat,np.array([2*np.pi,masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([masscenters_1[i,0],2*np.pi],
                #         [masscenters_1[i,1], masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                
                Start=np.matmul(ReshapeMat,np.array([0,masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline]))
                End=np.matmul(ReshapeMat,np.array([(2*np.pi - (masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline,2*np.pi]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([0,(2*np.pi - (masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline], 
                #         [masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline, 2*np.pi],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                
                Start=np.matmul(ReshapeMat,np.array([(2*np.pi - (masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline,0]))
                End=np.matmul(ReshapeMat,np.array([masscenters_2[i,0],masscenters_2[i,1]]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)#
                # ax.plot([(2*np.pi - (masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline, 
                #          masscenters_2[i,0]], 
                #         [0,masscenters_2[i,1]],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)#
        elif line[0]>np.pi and line[1] <-np.pi:  
            line = [2*np.pi + masscenters_2[i,0], -2*np.pi + masscenters_2[i,1]] - masscenters_1[i,:]
            dline = line[1]/line[0]            
            if (masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline)>0:
                Start=np.matmul(ReshapeMat,np.array([masscenters_1[i,0],masscenters_1[i,1]]))
                End=np.matmul(ReshapeMat,np.array([2*np.pi,masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([masscenters_1[i,0],2*np.pi],
                #         [masscenters_1[i,1], masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                
                Start=np.matmul(ReshapeMat,np.array([0,masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline]))
                End=np.matmul(ReshapeMat,np.array([(-(masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline,0]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([0,(-(masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline], 
                #         [masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline, 0],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                
                Start=np.matmul(ReshapeMat,np.array([(-(masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline,2*np.pi]))
                End=np.matmul(ReshapeMat,np.array([masscenters_2[i,0],masscenters_2[i,1]]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([(-(masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline, 
                #          masscenters_2[i,0]], 
                #         [2*np.pi,masscenters_2[i,1]],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)

            else:
                line = [2*np.pi + masscenters_2[i,0], -2*np.pi + masscenters_2[i,1]] - masscenters_1[i,:]
                dline = line[1]/line[0]
                Start=np.matmul(ReshapeMat,np.array([masscenters_1[i,0],masscenters_1[i,1]]))
                End=np.matmul(ReshapeMat,np.array([masscenters_1[i,0] + (- masscenters_1[i,1])/dline,0]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (- masscenters_1[i,1])/dline],
                #         [masscenters_1[i,1], 0],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                
                Start=np.matmul(ReshapeMat,np.array([masscenters_1[i,0] + (- masscenters_1[i,1])/dline,2*np.pi]))
                End=np.matmul(ReshapeMat,np.array([2*np.pi,2*np.pi + (2*np.pi- masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([masscenters_1[i,0] + (- masscenters_1[i,1])/dline, 2*np.pi], 
                #         [2*np.pi, 2*np.pi + (2*np.pi- masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                
                Start=np.matmul(ReshapeMat,np.array([0,2*np.pi + (2*np.pi- masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline]))
                End=np.matmul(ReshapeMat,np.array([masscenters_2[i,0],masscenters_2[i,1]]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([0, masscenters_2[i,0]], 
                #         [2*np.pi + (2*np.pi- masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline,masscenters_2[i,1]],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
        elif line[0]<-np.pi and line[1] >np.pi:
            line = [-2*np.pi + masscenters_2[i,0], 2*np.pi + masscenters_2[i,1]] - masscenters_1[i,:]
            dline = line[1]/line[0]
            if ((masscenters_1[i,1] + -(masscenters_1[i,0])*dline)<2*np.pi):

                Start=np.matmul(ReshapeMat,np.array([masscenters_1[i,0],masscenters_1[i,1]]))
                End=np.matmul(ReshapeMat,np.array([0,masscenters_1[i,1] + -(masscenters_1[i,0])*dline]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([masscenters_1[i,0],0],
                #         [masscenters_1[i,1], masscenters_1[i,1] + -(masscenters_1[i,0])*dline],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                
                Start=np.matmul(ReshapeMat,np.array([2*np.pi,masscenters_1[i,1] + -(masscenters_1[i,0])*dline]))
                End=np.matmul(ReshapeMat,np.array([2*np.pi + (2*np.pi - (masscenters_1[i,1] + -(masscenters_1[i,0])*dline))/dline,2*np.pi]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([2*np.pi, 2*np.pi + (2*np.pi - (masscenters_1[i,1] + -(masscenters_1[i,0])*dline))/dline], 
                #         [masscenters_1[i,1] + -(masscenters_1[i,0])*dline, 2*np.pi],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                
                Start=np.matmul(ReshapeMat,np.array([2*np.pi + (2*np.pi - (masscenters_1[i,1] + -(masscenters_1[i,0])*dline))/dline,0]))
                End=np.matmul(ReshapeMat,np.array([masscenters_2[i,0],masscenters_2[i,1]]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([2*np.pi + (2*np.pi - (masscenters_1[i,1] + -(masscenters_1[i,0])*dline))/dline, 
                #          masscenters_2[i,0]], 
                #         [0,masscenters_2[i,1]],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
            else:
                Start=np.matmul(ReshapeMat,np.array([masscenters_1[i,0],masscenters_1[i,1]]))
                End=np.matmul(ReshapeMat,np.array([masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline,2*np.pi]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline],
                #         [masscenters_1[i,1], 2*np.pi],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                
                Start=np.matmul(ReshapeMat,np.array([masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline,0]))
                End=np.matmul(ReshapeMat,np.array([0,0 + -(masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline)*dline]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline, 0], 
                #         [0, 0 + -(masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline)*dline],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                
                Start=np.matmul(ReshapeMat,np.array([2*np.pi,0 + -(masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline)*dline]))
                End=np.matmul(ReshapeMat,np.array([masscenters_2[i,0],masscenters_2[i,1]]))
                ax.plot([Start[0,0], End[0,0]],
                        [Start[0,1], End[0,1]],
                        c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                # ax.plot([2*np.pi, masscenters_2[i,0]], 
                #         [0 + -(masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline)*dline,masscenters_2[i,1]],
                #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
                
        elif line[0]< -np.pi:
            line = [(2*np.pi + masscenters_1[i,0]), masscenters_1[i,1]] - masscenters_2[i,:]
            dline = line[1]/line[0]
            Start=np.matmul(ReshapeMat,np.array([masscenters_2[i,0],masscenters_2[i,1]]))
            End=np.matmul(ReshapeMat,np.array([2*np.pi,masscenters_2[i,1] + (2*np.pi - masscenters_2[i,0])*dline]))
            ax.plot([Start[0,0], End[0,0]],
                    [Start[0,1], End[0,1]],
                    alpha = 1.0, linestyle='-', linewidth=0.6, c = 'grey')            
            # ax.plot([masscenters_2[i,0],2*np.pi],
            #         [masscenters_2[i,1], masscenters_2[i,1] + (2*np.pi - masscenters_2[i,0])*dline], 
            #         alpha = 1.0, linestyle='-', linewidth=0.6, c = 'grey')            
            Start=np.matmul(ReshapeMat,np.array([0,masscenters_2[i,1] + (2*np.pi - masscenters_2[i,0])*dline]))
            End=np.matmul(ReshapeMat,np.array([masscenters_1[i,0],masscenters_1[i,1]]))
            ax.plot([Start[0,0], End[0,0]],
                    [Start[0,1], End[0,1]], 
                    alpha = 1.0, linestyle='-', linewidth=0.6, c = 'grey')
            # ax.plot([0,masscenters_1[i,0]],
            #         [masscenters_2[i,1] + (2*np.pi - masscenters_2[i,0])*dline, masscenters_1[i,1]], 
            #         alpha = 1.0, linestyle='-', linewidth=0.6, c = 'grey')
        elif line[0]> np.pi:
            line = [ masscenters_2[i,0]+ 2*np.pi, masscenters_2[i,1]] - masscenters_1[i,:]
            dline = line[1]/line[0]


            Start=np.matmul(ReshapeMat,np.array([masscenters_1[i,0],masscenters_1[i,1]]))
            End=np.matmul(ReshapeMat,np.array([2*np.pi,masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline]))
            ax.plot([Start[0,0], End[0,0]],
                    [Start[0,1], End[0,1]],
                    c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
            # ax.plot([masscenters_1[i,0],2*np.pi],
            #         [masscenters_1[i,1], masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline],
            #         c = 'grey', alpha = 1.0, linestyle='-', linewidth=0.6)
            Start=np.matmul(ReshapeMat,np.array([0,masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline]))
            End=np.matmul(ReshapeMat,np.array([masscenters_2[i,0],masscenters_2[i,1]]))
            ax.plot([Start[0,0], End[0,0]],
                    [Start[0,1], End[0,1]],
                    alpha = 1.0, linestyle='-', linewidth=0.6, c = 'grey')
            # ax.plot([0,masscenters_2[i,0]],
            #         [masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline, masscenters_2[i,1]], 
            #         alpha = 1.0, linestyle='-', linewidth=0.6, c = 'grey')
        elif line[1]< -np.pi:
            line = [ masscenters_1[i,0], (2*np.pi + masscenters_1[i,1])] - masscenters_2[i,:]
            dline = line[1]/line[0]

            Start=np.matmul(ReshapeMat,np.array([masscenters_2[i,0],masscenters_2[i,1]]))
            End=np.matmul(ReshapeMat,np.array([masscenters_2[i,0] + (2*np.pi - masscenters_2[i,1])/dline,2*np.pi]))
            ax.plot([Start[0,0], End[0,0]],
                    [Start[0,1], End[0,1]], alpha = 1.0, linestyle='-', linewidth=0.6, c = 'grey'),
            # ax.plot([masscenters_2[i,0], masscenters_2[i,0] + (2*np.pi - masscenters_2[i,1])/dline], 
            #         [masscenters_2[i,1],2*np.pi], alpha = 1.0, linestyle='-', linewidth=0.6, c = 'grey'),
            Start=np.matmul(ReshapeMat,np.array([masscenters_1[i,0] - masscenters_1[i,1]/dline,0]))
            End=np.matmul(ReshapeMat,np.array([masscenters_1[i,0],masscenters_1[i,1]]))
            ax.plot([Start[0,0], End[0,0]],
                    [Start[0,1], End[0,1]],
                    alpha = 1.0, linestyle='-', linewidth=0.6, c = 'grey')
            # ax.plot([masscenters_1[i,0] - masscenters_1[i,1]/dline,masscenters_1[i,0]],
            #         [0, masscenters_1[i,1]], 
            #         alpha = 1.0, linestyle='-', linewidth=0.6, c = 'grey')
        elif line[1]> np.pi:
            line = [ masscenters_2[i,0], masscenters_2[i,1]+ 2*np.pi] - masscenters_1[i,:]
            dline = line[1]/line[0]

            Start=np.matmul(ReshapeMat,np.array([masscenters_1[i,0],masscenters_1[i,1]]))
            End=np.matmul(ReshapeMat,np.array([masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline,2*np.pi]))
            ax.plot([Start[0,0], End[0,0]],
                    [Start[0,1], End[0,1]],alpha = 1.0, linestyle='-', linewidth=0.6, c = 'grey'),
            # ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline], 
            #         [masscenters_1[i,1], 2*np.pi], alpha = 1.0, linestyle='-', linewidth=0.6, c = 'grey'),

            Start=np.matmul(ReshapeMat,np.array([masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline,0]))
            End=np.matmul(ReshapeMat,np.array([masscenters_2[i,0],masscenters_2[i,1]]))
            ax.plot([Start[0,0], End[0,0]],
                    [Start[0,1], End[0,1]],
                    alpha = 1.0, linestyle='-', linewidth=0.6, c = 'grey')
            # ax.plot([masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline,masscenters_2[i,0]],
            #         [0, masscenters_2[i,1]], 
            #         alpha = 1.0, linestyle='-', linewidth=0.6, c = 'grey')
        else:
            Start=np.matmul(ReshapeMat,np.array([masscenters_1[i,0],masscenters_1[i,1]]))
            End=np.matmul(ReshapeMat,np.array([masscenters_2[i,0],masscenters_2[i,1]]))
            # print(Start[0,0], End[0,0])
            # print(Start[0,1], End[0,1])
            ax.plot([Start[0,0], End[0,0]],
                    [Start[0,1], End[0,1]],
                    alpha = 1.0, linestyle='-', linewidth=0.6, c = 'grey')
            # ax.plot([masscenters_1[i,0],masscenters_2[i,0]],
            #         [masscenters_1[i,1],masscenters_2[i,1]], 
            #         alpha = 1.0, linestyle='-', linewidth=0.6, c = 'grey')
    # for i in np.arange(num_neurons):
    #     ax.scatter(masscenters_1[i,0], masscenters_1[i,1], s = 15, c = 'royalblue')
    #     ax.scatter(masscenters_2[i,0], masscenters_2[i,1], s = 25, c = 'red')
        # ax.text(masscenters_1[i,0], masscenters_1[i,1], s=str(i),fontsize=7)
        # ax.text(x=x coordinate, y=y coordinate, s=string to be displayed)
    Start=np.matmul(ReshapeMat,np.array([0,0]))
    End=np.matmul(ReshapeMat,np.array([0,2*np.pi]))        
    ax.plot([Start[0,0], End[0,0]],[Start[0,1], End[0,1]], c = 'lightgrey', alpha=0.7)
    Start=np.matmul(ReshapeMat,np.array([0,0]))
    End=np.matmul(ReshapeMat,np.array([2*np.pi,0]))        
    ax.plot([Start[0,0], End[0,0]],[Start[0,1], End[0,1]], c = 'lightgrey', alpha=0.7)
    Start=np.matmul(ReshapeMat,np.array([2*np.pi,0]))
    End=np.matmul(ReshapeMat,np.array([2*np.pi,2*np.pi]))        
    ax.plot([Start[0,0], End[0,0]],[Start[0,1], End[0,1]], c = 'lightgrey', alpha=0.7)
    Start=np.matmul(ReshapeMat,np.array([0,2*np.pi]))
    End=np.matmul(ReshapeMat,np.array([2*np.pi,2*np.pi]))        
    ax.plot([Start[0,0], End[0,0]],[Start[0,1], End[0,1]], c = 'lightgrey', alpha=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    # r_box = transforms.Affine2D().skew_deg(15,15)
    # for x in ax.images + ax.lines + ax.collections + ax.get_xticklabels() + ax.get_yticklabels():
    #     trans = x.get_transform()
    #     x.set_transform(r_box+trans) 
    #     if isinstance(x, PathCollection):
    #         transoff = x.get_offset_transform()
    #         x._transOffset = r_box+transoff 
    ax.set_xlim([-0.2,0.2+3*np.pi])
    ax.set_ylim([-0.2,0.2+np.sqrt(3)*np.pi])
    # ax.set_aspect('equal', 'box')
    ax.set_aspect('equal')
    ax.axis('off')
    
# points1 = np.random.uniform(low=0, high=2*np.pi, size=(20, 2))
# points2 = np.random.uniform(low=0, high=2*np.pi, size=(20, 2))
# plt.figure()
# ax=plt.subplot(111)
# plot_phase_distribution(points1, points2, ax)