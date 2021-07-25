import matplotlib.pyplot as plt
import os
import numpy as np
import torch

global colors, linestyles, markers
colors = ['k', 'm', 'g', 'y', 'r', 'b', 'c']
linestyles = ['-', '--', '-.']
markers = ['o', '^', '+', '*', 'x', '1']

def draw(plt_fun, record, label, i, NC, yaxis, xaxis=None):
    loop_i = int((i+1)/5)
    if not (xaxis is not None):
        xaxis = torch.tensor(range(1,len(yaxis)+1))
    plt_fun(xaxis, yaxis, color=colors[i-7*loop_i], 
            linestyle=linestyles[loop_i], label = label)
    if NC:
        index = (record[:,5][1:] == True)
        if xaxis is not None:
            xNC = xaxis[:-1][index]
        else:
            xNC = torch.tensor(range(1,len(yaxis)))[index]
        yNC = yaxis[:-1][index]
        plt_fun(xNC, yNC, '.', color=colors[i-7*loop_i], 
                marker=markers[i-5*loop_i], markersize=4)
        
def showFigure(methods_all, record_all, mypath, plotAll=False):
    """
    Plots generator.
    Input: 
        methods_all: a list contains all methods
        record_all: a list contains all record matrix of listed methods, 
        s.t., [fx, norm(gx), oracle calls, time, stepsize, is_negative_curvature]
        prob: name of problem
        mypath: directory path for saving plots
    OUTPUT:
        Oracle calls vs. F
        Oracle calls vs. Gradient norm
        Iteration vs. Step Size
    """
    fsize = 24
    myplt = plt.loglog
    # myplt = plt.semilogx
    # myplt = plt.plot
    
    figsz = (10,6)
    mydpi = 200
    
    fig1 = plt.figure(figsize=figsz)
    for i in range(len(methods_all)):
        record = record_all[i]
        draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,0], record[:,2]+1)
    plt.xlabel('Oracle calls', fontsize=fsize)
    plt.ylabel('F', fontsize=fsize)
    plt.legend()
    fig1.savefig(os.path.join(mypath, 'F'), dpi=mydpi)
    
    
    fig2 = plt.figure(figsize=figsz)
    for i in range(len(methods_all)):
        record = record_all[i]
        draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,1], record[:,2]+1)
    plt.xlabel('Oracle calls', fontsize=fsize)
    plt.ylabel('Gradient norm', fontsize=fsize)
    plt.legend()
    fig2.savefig(os.path.join(mypath, 'Gradient_norm'), dpi=mydpi)
    
    fig3 = plt.figure(figsize=figsz)
    for i in range(len(methods_all)):
        record = record_all[i]
        draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,0], record[:,3]+1)
    plt.xlabel('Time', fontsize=fsize)
    plt.ylabel('F', fontsize=fsize)
    plt.legend()
    fig3.savefig(os.path.join(mypath, 'Time'), dpi=mydpi)
    
    fig4 = plt.figure(figsize=figsz)
    for i in range(len(methods_all)):
        record = record_all[i]
        draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,4])
    plt.xlabel('Iteration', fontsize=fsize)
    plt.ylabel('Alpha/Delta', fontsize=fsize)
    # plt.ylabel('Delta', fontsize=fsize)
    plt.legend()
    fig4.savefig(os.path.join(mypath, 'Delta'), dpi=mydpi)
            
    if plotAll == True:
        fig4 = plt.figure(figsize=figsz)
        for i in range(len(methods_all)):
            record = record_all[i]
            draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,0])
        plt.xlabel('Iteration', fontsize=fsize)
        plt.ylabel('F', fontsize=fsize)
        plt.legend()
        fig4.savefig(os.path.join(mypath, 'Iteration_F'), dpi=mydpi)
        
        fig5 = plt.figure(figsize=figsz)
        for i in range(len(methods_all)):
            record = record_all[i]
            draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,1])
        plt.xlabel('Iteration', fontsize=fsize)
        plt.ylabel('Gradient norm', fontsize=fsize)
        plt.legend()
        fig5.savefig(os.path.join(mypath, 'Iteration_G'), dpi=mydpi)
        
        fig6 = plt.figure(figsize=figsz)
        for i in range(len(methods_all)):
            record = record_all[i]
            draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,0], record[:,2]+1)
        plt.xlabel('Oracle calls', fontsize=fsize)
        plt.ylabel('Step Size', fontsize=fsize)
        plt.legend()
        fig6.savefig(os.path.join(mypath, 'OC_alpha'), dpi=mydpi)
        
def main():
    methods_all = []
    record_all = []
#    for method in os.listdir('showFig'): #only contains txt files
#        methods_all.append(method.rsplit('.', 1)[0])
#        record = np.loadtxt(open('showFig/'+method,"rb"),delimiter=",",skiprows=0)
#        record_all.append(record)
    for method in os.listdir('showFig'): #only contains txt files
        methods_all.append(method.rsplit('.', 1)[0])
        record = np.loadtxt(open('showFig/'+method,"rb"),delimiter=",",skiprows=0)
        record_all.append(record)
    mypath = 'showFig_plots'
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
# =============================================================================
    """
    regenerate any 1 total run plot via txt record matrix in showFig folder.
    Note that directory only contains txt files.
    For performace profile plots, see pProfile.py
    """
    showFigure(methods_all, record_all, None, mypath)
    
    
    
    
    
    
if __name__ == '__main__':
    main()