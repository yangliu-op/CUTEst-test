# import numpy as np
import torch

def lbfgs(g, s, y): 
    """
    INPUTS:
        g: column vector, gradient gk
        s: a matrix contains k columns of the past (x_{i+1} - x_i) vectors
        s: a matrix contains k columns of the past (g_{i+1} - g_i) vectors
    OUTPUS:
        d: lbfgs direction
    """
    p, k = s.shape
    eps = 1E-18
    ro = 1.0/(torch.sum(y*s, axis=0)+eps)
        
    # q = np.zeros((p,k+1))
    # r = np.zeros((p,k+1))
    # al = np.zeros((k,1))
    # be = np.zeros((k,1))
    q = torch.zeros(p, k+1, device=g.device)
    r = torch.zeros(p, k+1, device=g.device)
    al = torch.zeros(k, device=g.device)
    be = torch.zeros(k, device=g.device)
    q[:,k] = g
    
    for i in range(k-1,-1,-1):
        al[i] = ro[i]*(s[:,i].dot(q[:,i+1]))
        q[:,i] = q[:,i+1]-al[i]*y[:,i]
    if k >=1:
        r[:,0] = abs(y[:,k-1].dot(s[:,k-1]))/(y[:,k-1].dot(y[:,k-1])+eps)*q[:,0]
    else:
        r[:,0] = q[:,0]
#    r[:,0] = np.dot(Hdiag.toarray(),q[:,0])
    
    for i in range(k):
        be[i] = ro[i]*(y[:,i].dot(r[:, i]))
        r[:,i+1] = r[:,i] + s[:,i]*(al[i]-be[i])
        
    d = r[:,k]
    
    return d
