"""
    Code for ongoing Newton-MR methods apply to Non-convex Unconstrained 
    optimisation use.
    Authors: Yang Liu, Fred Roosta

    Genetating experiments on CUTEST problems with different algorithms.

Reference:
    "Convergence of newton-mr under inexact hessian information".
Authors: Liu, Yang and Roosta, Fred
    
    "Newton-MR: Newton’s Method Without Smoothness or Convexity"
Authors: Roosta, Fred and Liu, Yang and Xu, Peng and Mahoney, Michael W

    "A Newton-CG algorithm with complexity guarantees for smooth unconstrained 
    optimization"
Authors: Royer, Cl{\'e}ment W and O’Neill, Michael and Wright, Stephen J

    "Trust-Region Newton-CG with Strong Second-Order Complexity Guarantees for 
    Nonconvex Optimization"
Authors: Curtis, Frank E and Robinson, Daniel P and Royer, Cl{\'e}ment W 
    and Wright, Stephen J
    
    "The Conjugate Residual Method in Linesearch and Trust-Region Methods"
Authors: Dahito, Marie-Ange and Orban, Dominique
"""

from initialize import initialize
        
class algPara():
    def __init__(self, value):
        self.value = value
    
#initialize methods
### Note: without restarting spyder console, pycutest may return wrong information.
prob = [
        # execute individual problem
        # 'ZANGWIL2',
        # 'HILBERTA',
        # 'PALMER1C',
        # 'PALMER4C',
        # 'PALMER5D',
        'PALMER7C',
# ########################################################
        # execute problemset and produce performance profile plots
        # 'Unconstrained', # manual problems, can be edited in initialized.py
        ]

methods = [
            # 'Newton_MR_invex', # Warning: the original Newton-MR are not designed to solve non-convex problems
            'L_BFGS',
            'Newton_CG', # Royer
            'Newton_CR', # Dahito
            'Newton_CG_TR', # Steihaug CG TR
            'Newton_CG_TR_Pert', # Curtis CG TR
            ]

#initial point
x0Type = [
            # 'randn',
            'rand',
            # 'ones'
            # 'zeros',
            ]

#initialize parameters
algPara.funcEvalMax = 1E5 #Set mainloop termination with Maximum Oracle calls
algPara.mainLoopMaxItrs = algPara.funcEvalMax #Set mainloop termination with Maximum Iterations
algPara.innerSolverMaxItrs = 1E3
algPara.lineSearchMaxItrs = 1E3
algPara.innerSolverTol = 1E-2 #residual vs. norm_y inexactness
algPara.innerSolverTol1 = 1E-1 #residual vs. norm_y inexactness

algPara.gradTol = 1e-10 #If norm(g)<gradTol, main loop breaks
algPara.epsilon = 1E-5 #perturbation for Hessian
algPara.show = True #print result for every iteration
algPara.zoom = 1 # starting x0 = x0/zoom

algPara.L = 10 # L-BFGS limited iterations usage
algPara.beta = 1E-4 #Armijo Line search para
algPara.beta2 = 0.9 #Wolfe's condition for L-BFGS

algPara.Delta = 100 #Trust-region radius
# algPara.Delta_min = 1E-6
algPara.Delta_max = 1E10
algPara.gama1 = 2 #Trust-region radius enlarge
algPara.gama2 = 0.23 #Trust-region radius shrink
algPara.eta = 0.2 #Trust-region succesful measure, < 0.25 in Trust-Region Newton-CG

#subsampling
HProp_all = [1] # full Hessian
plotAll = False

## Initialize
initialize(methods, prob, x0Type, HProp_all, algPara, plotAll)


