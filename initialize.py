import numpy as np
import torch
from optim_algo import (Newton_MR_invex, L_BFGS, Newton_CR, Damped_Newton_CG, 
                        Newton_CG_TR, Newton_CG_TR_Pert)
from showFigure import showFigure
import matplotlib.pyplot as plt
from pProfile import pProfile

### Note: After one set the linux system environment, one may still get error
### One solution is to use the following code (change to your location).

import os
#print(os.environ)
#os.environ['CUTEST'] = '/home/yang/Documents/cutest/cutest/'
#os.environ['SIFDECODE'] = '/home/yang/Documents/cutest/sifdecode/'
#os.environ['ARCHDEFS'] = '/home/yang/Documents/cutest/archdefs/'
#os.environ['MASTSIF'] = '/home/yang/Documents/cutest/mastsif/'
#os.environ['MYARCH'] = 'pc64.lnx.gf7'
import pycutest

def initialize(methods, prob, x0Type, HProp_all, algPara, plotAll=False): 
    """
    data: name of chosen dataset
    methods: name of chosen algorithms
    prob: name of chosen objective problems
    x0Type: type of starting point
    HProp_all: all Hessian proportion for sub-sampling methods
    batchsize: batchsize of first order algorithms
    algPara: a class that contains:
        mainLoopMaxItrs: maximum iterations for main loop
        funcEvalMax: maximum oracle calls (function evaluations) for algorithms
        innerSolverMaxItrs: maximum iterations for inner (e.g., CG) solvers
        lineSearchMaxItrs: maximum iterations for line search methods
        gradTol: stopping condition para s.t., norm(gk) <= Tol
        innerSolverTol: inexactness tolerance of inner solvers
        beta: parameter of Armijo line-search
        beta2: parameter of Wolfe curvature condition (line-search)
        plotAll: plot 3 plots more
    """
    print('Initialization...')        
    print('Problem:', prob, end='  ')
    print('innerSolverMaxItrs = %8s' % algPara.innerSolverMaxItrs, end='  ')
    print('lineSearchMaxItrs = %8s' % algPara.lineSearchMaxItrs, end='  ')
    print('gradTol = %8s' % algPara.gradTol, end='  ')
    print('Starting point = %8s ' % x0Type)  
    x0Type = x0Type[0]
        
    filename = '%s_orc_%s_solItr_%s_x0_%s_subH_%s_eps_%s_tol_%s_%s_ZOOM_%s' % (
            prob, algPara.funcEvalMax, algPara.innerSolverMaxItrs, x0Type, 
            len(HProp_all), algPara.epsilon, algPara.innerSolverTol, 
            algPara.innerSolverTol1, algPara.zoom)  
    mypath = filename
    print('filename', filename)
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
    
    if prob[0] == 'Unconstrained':
        execute_pProfile(prob, methods, x0Type, algPara, 
                         HProp_all, mypath)
    else:
        execute(prob, methods, x0Type, algPara, HProp_all, 
                     mypath, plotAll)    

def get_obj_cutest(x, p, arg):
    if arg is None:
        out = [torch.tensor(p.obj(x)).to(torch.float32), 
               torch.tensor(p.obj(x, gradient=True)[1]).to(torch.float32), 
               lambda v: torch.tensor(p.hprod(v, x=x)).to(torch.float32)]
    if arg == 'fg':
        out = [torch.tensor(p.obj(x)).to(torch.float32), 
               torch.tensor(p.obj(x, gradient=True)[1]).to(torch.float32)]
    if arg == 'f':
        out = torch.tensor(p.obj(x)).to(torch.float32)
    if arg == 'g':
        out = torch.tensor(p.obj(x, gradient=True)[1]).to(torch.float32)
    return out
  
def execute(prob, methods, x0Type, algPara, HProp_all, 
            mypath, plotAll):  
    """
    Excute all methods/problems with 1 run and give plots.
    """            
    
    p = pycutest.import_problem(prob[0])
    l = p.n     
    x0 = generate_x0(x0Type, l)  
    print('Samples, Dimensions', p.m, p.n)
    
    obj = lambda x, control=None: get_obj_cutest(x, p, control)
    
    methods_all, record_all = run_algorithms(
            obj, x0, methods, algPara, HProp_all, mypath)
    
    showFigure(methods_all, record_all, mypath, plotAll)
    
def execute_pProfile(prob, methods, x0Type, algPara, HProp_all, 
                     mypath): 
    """
    Excute all methods/problems with multiple runs. Compare the best results
    and give performance profile plots between methods.
    
    Record a pProfile matrix s.t., every row are the best (f/g/error) result 
        of all listed methods. e.g., run n cutest problems listed in data with
        m different optimisation methods will be a n x m matrix.
    """           
    
    mypath_pProfile = 'pProfile'
    if not os.path.isdir(mypath_pProfile):
       os.makedirs(mypath_pProfile) 
            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device', device)
    # probset_1 = pycutest.find_problems(objective='Q', constraints='U',
    #                               regular=True)  
    # print('Unconstrained, Quadratic', probset_1)
    # probset_2 = pycutest.find_problems(objective='S', constraints='U', 
    #                               regular=True)  
    # print('Unconstrained, Sum_of_squares', probset_2)
    # probset_3 = pycutest.find_problems(objective='O', constraints='U', 
    #                               regular=True)  
    # print('Unconstrained, Other', probset_3)
    # print(len(probset_1) + len(probset_2)  + len(probset_3))
    
    # With the above commented code, one can grab 3 problem sets, 285 
    # unconstrained CUTEst problems in total. However, with differnt intial 
    # starting point, different error may occer.
    # The following 252 CUTEST problems are from the above 3 datasets and 
    # will not cause error (personal view).
    
    prob = [
        'PALMER5C',
        'ZANGWIL2',
        'HILBERTA',
        'PALMER1C',
        'PALMER4C',
        'PALMER5D',
        'PALMER7C',
        # 'PALMER1D',
        # 'PALMER2C',
        # 'PALMER8C',
        # 'PALMER3C',
        # 'DENSCHND',
        # 'DENSCHNC',
        # 'RAT42LS',
        # 'POWELLSQLS',
        # 'PRICE4',
        # 'LANCZOS1LS',
        # 'DENSCHNF',
        # 'EXP2',
        # 'VESUVIALS',
        # 'BROWNDEN',
        # 'ROSENBR',
        # 'ENSOLS',
        # 'HEART8LS',
        # 'BARD',
        # 'MGH10SLS',
        # 'S308NE',
        # 'EXPFIT',
        # 'GULF',
        # 'EGGCRATE',
        # 'BOXBODLS',
        # 'S308',
        # 'RAT43LS',
        # 'THURBERLS',
        # 'PALMER6C',
        # 'POWERSUM',
        # 'HEART6LS',
        # 'HATFLDE',
        # 'FBRAIN3LS',
        # 'HIMMELBF',
        # 'BROWNBS',
        # 'LSC1LS',
        # 'SINEVAL',
        # 'LANCZOS3LS',
        # 'VIBRBEAM',
        # 'MGH09LS',
        # 'JENSMP',
        # 'MGH10LS',
        # 'GROWTHLS',
        # 'MISRA1BLS',
        # 'BEALE',
        # 'DANIWOODLS',
        # 'PRICE3',
        # 'POWELLBSLS',
        # 'LANCZOS2LS',
        # 'VESUVIOLS',
        # 'HATFLDFLS',
        # 'MGH17LS',
        # 'GBRAINLS',
        # 'COOLHANSLS',
        # 'SSI',
        # 'GAUSS2LS',
        # 'MEYER3',
        # 'KIRBY2LS',
        # 'YFITU',
        # 'MISRA1DLS',
        # 'ENGVAL2',
        # 'CLUSTERLS',
        # 'WAYSEA1',
        # 'MISRA1ALS',
        # 'RECIPELS',
        # 'HELIX',
        # 'ROSZMAN1LS',
        # 'LSC2LS',
        # 'MUONSINELS',
        # 'ELATVIDU',
        # 'NELSONLS',
        # 'GAUSS3LS',
        # 'CLIFF',
        # 'BRKMCC',
        # 'LOGHAIRY',
        # 'ROSENBRTU',
        # 'HIMMELBH',
        # 'MARATOSB',
        # 'DENSCHNA',
        # 'HAIRY',
        # 'ALLINITU',
        # 'SISSER',
        # 'HIMMELBB',
        # 'HUMPS',
        # 'MEXHAT',
        # 'HIMMELBCLS',
        # 'HIMMELBG',
        # 'SNAIL',
        # 'GAUSS1LS',
        # 'JUDGE',
        # 'HILBERTB',
        # 'TOINTQOR',
        # 'DMN15103LS',
        # 'DMN15333LS',
        # 'ERRINROS',
        # 'BA-L1SPLS',
        # 'LUKSAN12LS',
        # 'BA-L1LS',
        # 'HYDC20LS',
        # 'HATFLDGLS',
        # 'HYDCAR6LS',
        # 'ERRINRSM',
        # 'TRIGON2',
        # 'LUKSAN13LS',
        # 'WATSON',
        # 'DMN37143LS',
        # 'CHNRSNBM',
        # 'VANDANMSLS',
        # 'STRTCHDV',
        # 'CHNROSNB',
        # 'TOINTPSP',
        # 'TOINTGOR',
        # 'DIAMON2DLS',
        # 'DIAMON3DLS',
        # 'OSBORNEB',
        # 'LUKSAN14LS',
        # 'TRIDIA',
        # 'TESTQUAD',
        # 'DQDRTIC',
        # 'DIXON3DQ',
        # 'PENALTY2',
        # 'KSSLS',
        # 'SSBRYBND',
        # 'MODBEALE',
        # 'ARGTRIGLS',
        # 'PENALTY1',
        # 'LUKSAN22LS',
        # 'LUKSAN16LS',
        # 'EIGENALS',
        # 'MSQRTBLS',
        # 'INTEQNELS',
        # 'MNISTS5LS',
        # 'BROYDNBDLS',
        # 'GENROSE',
        # 'NONDIA',
        # 'ARGLINB',
        # 'SROSENBR',
        # 'EIGENCLS',
        # 'COATING',
        # 'YATP2LS',
        # 'LUKSAN21LS',
        # 'YATP2CLS',
        # 'BROWNAL',
        # 'MANCINO',
        # 'SPIN2LS',
        # 'LUKSAN15LS',
        # 'CHAINWOO',
        # 'ARGLINC',
        # 'CYCLOOCFLS',
        # 'MNISTS0LS',
        # 'OSCIPATH',
        # 'BDQRTIC',
        # 'BRYBND',
        # 'LUKSAN11LS',
        # 'EIGENBLS',
        # 'MSQRTALS',
        # 'SPMSRTLS',
        # 'QING',
        # 'EXTROSNB',
        # 'WOODS',
        # 'TQUARTIC',
        # 'FREUROTH',
        # 'ARGLINA',
        # 'BROYDN3DLS',
        # 'LUKSAN17LS',
        # 'LIARWHD',
        # 'NONMSQRT',
        # 'OSCIGRAD',
        # 'FMINSURF',
        # 'SENSORS',
        # 'DIXMAANB',
        # 'SINQUAD',
        # 'POWER',
        # 'DIXMAANH',
        # 'FLETCHBV',
        # 'QUARTC',
        # 'TOINTGSS',
        # 'EG2',
        # 'CURLY10',
        # 'DIXMAANJ',
        # 'DIXMAANE',
        # 'FMINSRF2',
        # 'PENALTY3',
        # 'SCOSINE',
        # 'ARWHEAD',
        # 'NCB20',
        # 'NCB20B',
        # 'FLETCBV2',
        # 'DIXMAANC',
        # 'SSCOSINE',
        # 'SCHMVETT',
        # 'VARDIM',
        # 'BOXPOWER',
        # 'POWELLSG',
        # 'DIXMAANM',
        # 'NONDQUAR',
        # 'FLETBV3M',
        # 'COSINE',
        # 'DIXMAANP',
        # 'GENHUMPS',
        # 'DIXMAANF',
        # 'DIXMAANN',
        # 'CURLY20',
        # 'DIXMAANI',
        # 'DIXMAANA',
        # 'FLETCBV3',
        # 'NONCVXUN',
        # 'NONCVXU2',
        # 'DIXMAANG',
        # 'ENGVAL1',
        # 'EDENSCH',
        # 'DIXMAANO',
        # 'FLETCHCR',
        # 'DIXMAANK',
        # 'SPARSINE',
        # 'DQRTIC',
        # 'DIXMAANL',
        # 'BROYDN7D',
        # 'INDEFM',
        # 'BOX',
        # 'VAREIGVL',
        # 'CRAGGLVY',
        # 'CURLY30',
        # 'DIXMAAND',
        # 'YATP1LS',
        # 'OSBORNEA',
        # 'SPARSQUR',
        # 'CHWIRUT2LS',
        # 'DJTL',
        # 'GAUSSIAN',
        # 'DEVGLA2NE',
        # 'KOWOSB',
        # 'BOX3',
        # 'DENSCHNB',
        # 'CHWIRUT1LS',
        # 'WAYSEA2',
        # 'DENSCHNE',
        # 'HATFLDD',
        # 'CUBE',
        # 'VESUVIOULS',
        # 'ECKERLE4LS',
        # 'CERI651ELS',
        # 'BIGGS6',
        # 'HATFLDFL',
        # 'STREG',
        # 'CERI651ALS',
            ]
    
    for i in range(len(prob)):
        print('Current Problem', prob[i], i)
        p = pycutest.import_problem(prob[i])
        l = p.n     
        x0 = generate_x0(x0Type, l)  
        print('Samples, Dimensions', p.m, p.n)
        obj = lambda x, control=None: get_obj_cutest(x, p, control)
        

        methods_all, record_all = run_algorithms(
                obj, x0, methods, algPara, HProp_all, mypath)            
        
        number_of_methods = len(methods_all)     
        pProfile_fi = np.zeros((1,number_of_methods))
        pProfile_gi = np.zeros((1,number_of_methods)) 
        
        for m in range(number_of_methods):
            record_matrices_i = record_all[m]
            pProfile_fi[0,m] = record_matrices_i[-1,0]
            pProfile_gi[0,m] = record_matrices_i[-1,1]
        if i == 0:     
            pProfile_f = pProfile_fi
            pProfile_g = pProfile_gi
            with open(os.path.join(mypath_pProfile, 'methods.txt'), \
                      'w') as myfile:
                for method in methods_all:
                    myfile.write('%s\n' % method)
        else:
            pProfile_f = np.append(pProfile_f, pProfile_fi, axis=0)
            pProfile_g = np.append(pProfile_g, pProfile_gi, axis=0)
            
        np.savetxt(os.path.join(mypath_pProfile, 'problem.txt'), \
                   prob[:i+1], delimiter="\n", fmt="%s")
        np.savetxt(os.path.join(mypath_pProfile, 'objVal.txt'), \
                   pProfile_f, delimiter=',')
        np.savetxt(os.path.join(mypath_pProfile, 'gradNorm.txt'), \
                   pProfile_g, delimiter=',')
       
    figsz = (6,4)
    mydpi = 200      
    
    fig1 = plt.figure(figsize=figsz)    
    pProfile(methods_all, pProfile_f, ylabel='F')
    fig1.savefig(os.path.join(mypath_pProfile, 'objVal'), dpi=mydpi)
    
    fig2 = plt.figure(figsize=figsz)    
    pProfile(methods_all, pProfile_g, ylabel='GradientNorm')
    fig2.savefig(os.path.join(mypath_pProfile, 'gradNorm'), dpi=mydpi)
    
    with open(os.path.join(mypath_pProfile, 'methods.txt'), 'w') as myfile:
        for method in methods_all:
            myfile.write('%s\n' % method)
    np.savetxt(os.path.join(mypath_pProfile, 'objVal.txt'), \
               pProfile_f, delimiter=',')
    np.savetxt(os.path.join(mypath_pProfile, 'gradNorm.txt'), \
               pProfile_g, delimiter=',')
    
        
        
def run_algorithms(obj, x0, methods, algPara, HProp_all, mypath):
    """
    Distribute all problems to its cooresponding optimisation methods.
    """
    record_all = []            
    record_txt = lambda filename, myrecord: np.savetxt(
            os.path.join(mypath, filename+'.txt'), myrecord, delimiter=',') 
                           
    if 'Newton_MR_invex' in methods:  
        print(' ')
        myMethod = 'Newton_MR_invex'
        for i in range(len(HProp_all)):
            HProp = HProp_all[i]
            if HProp != 1:
                obj_subsampled = lambda x, control=None: obj(x, control, HProp)
                myMethod = 'ss' + myMethod + '_%s%%' % (int(HProp_all[i]*100))
            else:
                obj_subsampled = obj
            print(myMethod)
            record_all.append(myMethod)
            HProp = HProp_all[i]
            x, record = Newton_MR_invex(
                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, 
                    algPara.funcEvalMax, algPara.innerSolverTol, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.beta, algPara.show)
            np.savetxt(os.path.join(mypath, myMethod+'.txt'), record, delimiter=',')
            record_all.append(record)
            
    if 'L_BFGS' in methods:
        print(' ')
        myMethod = 'L_BFGS'
        print(myMethod)
        record_all.append(myMethod)
        x, record = L_BFGS(
                obj, x0, algPara.mainLoopMaxItrs, algPara.funcEvalMax, 
                algPara.lineSearchMaxItrs, algPara.gradTol, algPara.L, algPara.beta, 
                algPara.beta2, algPara.show, record_txt)
        np.savetxt(os.path.join(mypath, 'L_BFGS.txt'), record, delimiter=',')
        record_all.append(record)
            
    if 'Newton_CR' in methods:  
        print(' ')
        myMethod = 'Newton_CR'        
        for j in range(len(HProp_all)):
            HProp = HProp_all[j]
            if HProp != 1:
                obj_subsampled = lambda x, control=None: obj(x, control, HProp)
                myMethod = 'ss' + myMethod + '_%s%%' % (int(HProp*100))
            else:
                obj_subsampled = obj
            print(myMethod)
            record_all.append(myMethod)
            x, record = Newton_CR(
                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, 
                    algPara.funcEvalMax, algPara.innerSolverTol1, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.beta, algPara.beta2, algPara.epsilon, 
                    algPara.show)
            np.savetxt(os.path.join(mypath, myMethod+'.txt'), record, delimiter=',')
            record_all.append(record)
    
    if 'Newton_CG' in methods:  
        print(' ')
        myMethod = 'Newton_CG_Pert'
        for j in range(len(HProp_all)):
            HProp = HProp_all[j]
            if HProp != 1:
                obj_subsampled = lambda x, control=None: obj(x, control, HProp)
                myMethod = 'ss' + myMethod + '_%s%%' % (int(HProp*100))
            else:
                obj_subsampled = obj
            print(myMethod)
            record_all.append(myMethod)
            x, record = Damped_Newton_CG(
                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, 
                    algPara.funcEvalMax, algPara.innerSolverTol1, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.beta, algPara.epsilon, 
                    algPara.show, record_txt)
            record_txt(myMethod, record)
            record_all.append(record)
    
    if 'Newton_CG_TR' in methods:  
        print(' ')
        myMethod = 'Newton_CG_TR'
        for j in range(len(HProp_all)):
            HProp = HProp_all[j]
            if HProp != 1:
                obj_subsampled = lambda x, control=None: obj(x, control, HProp)
                myMethod = 'ss' + myMethod + '_%s%%' % (int(HProp*100))
            else:
                obj_subsampled = obj
            print(myMethod)
            record_all.append(myMethod)
            x, record = Newton_CG_TR(
                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, 
                    algPara.funcEvalMax, algPara.innerSolverTol1, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.Delta, algPara.Delta_max, 
                    algPara.gama1, algPara.gama2, algPara.eta,
                    algPara.show, record_txt)
            record_txt(myMethod, record)
            record_all.append(record)
    
    if 'Newton_CG_TR_Pert' in methods:  
        print(' ')
        myMethod = 'Newton_CG_TR_Pert'
        for j in range(len(HProp_all)):
            HProp = HProp_all[j]
            if HProp != 1:
                obj_subsampled = lambda x, control=None: obj(x, control, HProp)
                myMethod = 'ss' + myMethod + '_%s%%' % (int(HProp*100))
            else:
                obj_subsampled = obj
            print(myMethod)
            record_all.append(myMethod)
            x, record = Newton_CG_TR_Pert(
                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, 
                    algPara.funcEvalMax, algPara.innerSolverTol1, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.epsilon, 
                    algPara.Delta, algPara.Delta_max, 
                    algPara.gama1, algPara.gama2, algPara.eta,
                    algPara.show, record_txt)
            record_txt(myMethod, record)
            record_all.append(record)
            
    methods_all = record_all[::2]
    record_all = record_all[1::2]
    
    return methods_all, record_all

    
def generate_x0(x0Type, l, zoom=1):    
    """
    Generate different type starting point.
    """
    if x0Type == 'randn':
        x0 = torch.randn(l)/zoom
    if x0Type == 'rand':
        x0 = torch.rand(l)/zoom
    if x0Type == 'ones':
        x0 = torch.ones(l)/zoom
    if x0Type == 'zeros':
        x0 = torch.zeros(l)
    return x0