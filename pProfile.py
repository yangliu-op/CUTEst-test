# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 08:24:44 2019

@author: Liu Yang
"""

import numpy as np
import os
import matplotlib.pyplot as plt

def pProfile(methods, Matrix, length=300, arg='log10', cut=None, 
             ylabel='Performance Profile', ylim=None, Input_positive=False):
    """
    A performance profile plot is the CDF of the performance of every 
    algorithms. Put your pProfile record (objVal/gradNorm/err) matrix to 
    pProfile folder and run this file to regenerate plots with proper scale.
    
    Reference:        
        E. D. Dolan and J. J. More. Benchmarking optimization software with 
        performance pro
files. Mathematical programming, 91(2): 201-213, 2002.
        
        N. Gould and J. Scott. A note on performance pro
les for benchmarking 
        software. ACM Transactions on Mathematical Software (TOMS), 43(2):15, 
        2016.
    INPUT:
        methods: list of all methods
        Matrix: a record matrix s.t., every row are the best (f/g/error) result 
            of all listed methods. e.g., 500 total_run of 5 different 
            optimisation methods will be a 500 x 5 matrix.
        length: number of nodes to compute on range x
        arg: scalling methods of x-axis
        cute: number of nodes to plot/display
        ylabel: label of y-axis
        ylim: plot/display [ylim, 1]
    OUTPUT:
        performance profile plots of the cooresponding (f/g/error) record 
        matrix.
    """
    
    M_hat = np.min(Matrix, axis=1) + 1E-50
    if not Input_positive:
        ### If M_hat contains negative number, 
        ### shift all result with - 2*M_hat, Then the minimum = -M_hat 
        ### and all other results are larger accordingly 
        Matrix[np.where(M_hat<0)] = (Matrix[np.where(M_hat<0)].T - M_hat[np.where(M_hat<0)]*2).T
        M_hat[np.where(M_hat<0)] = -M_hat[np.where(M_hat<0)]
        
    R = Matrix.T/M_hat
    R = R.T   
    x_high = np.max(R)
    n, d = Matrix.shape
    
    if arg is None:
        x = np.linspace(1, x_high, length)
        myplt = plt.plot
        plt.xlabel('lambda')
    if arg == 'log2':
        x = np.logspace(0, np.log2(x_high), length)
        myplt = plt.semilogx
        plt.xlabel('log2(lambda)')
    if arg == 'log10':
        x = np.logspace(0, np.log10(x_high), length)
        myplt = plt.semilogx
        plt.xlabel('log(lambda)')
        
    if cut != None:
        x = x[:cut]
        length = cut
    
    colors = ['k', 'm', 'b', 'g', 'y', 'r', 'k', 'm', 'b', 'g']
    linestyles = ['-', '-', '-', '--', '--', '--', '-.', '-.', '-.', '-.']
    
    if ylim != None:
        axes = plt.axes()
        axes.set_ylim(ylim, 1)
    
    for i in range(len(methods)):
        myMethod = methods[i]
#        loop_i = int((i+1)/7)
        Ri = np.tile(R[:,i], (length, 1))
        xi = np.tile(x, (n,1))
        yi = np.sum((Ri.T <= xi), axis=0)/n
        # myplt(x, yi, color=colors[i-7*loop_i], linestyle=linestyles[loop_i], 
        #       label = myMethod)
        myplt(x, yi, color=colors[i], linestyle=linestyles[i], 
              label = myMethod)
        
    plt.ylabel(ylabel)
    plt.legend()
    
    
def main():         
    mypath = 'pProfile'
    if not os.path.isdir(mypath):
       os.makedirs(mypath)   
       
    with open('pProfile/methods.txt', 'r') as myfile:
        methods = myfile.read().split()
    F = np.loadtxt(open("pProfile/objVal.txt","rb"),delimiter=",",skiprows=0)
    figsz = (16,12)
    mydpi = 200
    length = 300
    
    fig = plt.figure(figsize=figsz)    
    pProfile(methods, F, length, arg='log10',cut=80, ylabel='Performance Profile on f', Input_positive=False) #80 150
    fig.savefig(os.path.join(mypath, 'F'), dpi=mydpi)
    
if __name__ == '__main__':
    main()