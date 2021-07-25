import torch
import numpy as np

def modifiedCR(A, b, maxiter, epsilon, tau_r, tau_a=0):
    T = 0
    d = len(b)
    device = b.device
    s = torch.zeros(d, device = device)
    r = b # -gradient
    u = Ax(A, r)
    zeta = torch.dot(r, u)
    p = r
    q = u
    delta = zeta
    rho = torch.dot(r, r)
    mu = rho
    pi = rho
    nu = torch.sqrt(rho)
    norm_g = nu
    rel_res = 1
    
    while nu >= tau_a + tau_r * norm_g and T < maxiter:
        T += 1
        if delta <= epsilon*pi or zeta <= epsilon*rho:
            if T == 1:
#                print('-g')
                return b, rel_res, T, 'NC'
            else:
                return s, rel_res, T, 'Sol'
        alpha = zeta/torch.norm(q)**2
        s = s + alpha*p
        r = r - alpha*q
        nu = r.norm()
        rho = nu**2
        u = Ax(A, r)
        zeta_new = torch.dot(r, u) #zeta = r'Hr
        beta = zeta_new/zeta
        zeta = zeta_new
        p = r + beta * p
        pi = p.norm()**2
        q = u + beta*q #q = Hp
        delta = zeta + beta**2*delta #delta = p'Hp
        rel_res = nu/norm_g
    return s, rel_res, T, 'Sol'

def CR(H, g, tau_a, tau_r, maxiter):
    T = 0
    d = len(g)
    device = g.device
    s = torch.zeros(d, device = device)
    r = -g
    u = Ax(H, r)
    zeta = torch.dot(r, u)
    p = r
    q = u
    rho = torch.dot(r, r)
    nu = torch.sqrt(rho)
    norm_g = nu
    
    while nu >= tau_a + tau_r * norm_g and T < maxiter:
        T += 1
        alpha = zeta/torch.norm(q)**2
        s = s + alpha*p
        r = r - alpha*q
        rho = rho - alpha*zeta #rho = torch.norm(r)^2
        nu = torch.sqrt(rho) #nu = torch.norm(r)
        u = Ax(H, r)
        zeta_new = torch.dot(r, u) #zeta = r'Hr
        beta = zeta_new/zeta
        zeta = zeta_new
        p = r + beta * p
        q = u + beta*q #q = Hp
    return s, T    
    

def Ax(A, v):
    if callable(A):
        Ax = A(v)
    else:
        Ax =torch.mv(A, v)
    return Ax

def main():
    from myCG import myCG
# =============================================================================
    from time import time
    # torch.manual_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = 50
    A = torch.randn(n, n, device=device)
    A = torch.mm(A.T, A)
    b = torch.randn(n, device=device)
    t1 = time()
    print(modifiedCR(A + A.T, b, 100, 0, 1E-2))
    print(' ')
    print(CR(A + A.T, -b, 0, 1E-2, 100))
    print(' ')

    
if __name__ == '__main__':
    main()
