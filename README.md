# CUTEst-test
A framework of testing algorithms with CUTEst test problems.

## CUTEst:
In short, The Constrained and Unconstrained Environment, safe threads, (CUTEst) is a set of test problems for linear and nonlinear optimization. For detailes, see, 
https://www.cuter.rl.ac.uk/Problems/mastsif.shtml

Reference: 
"CUTEst: a constrained and unconstrained testing environment with safe threads for mathematical optimization"
Authurs: Gould, Nicholas IM and Orban, Dominique and Toint, Philippe L

## Pre-installation: 
Pytorch, Linux systems (pycutest only supports Mac and Linux), CUTEst test problems, pycutest.

To successfully install CUTEst test problems/pycutest interface, one can use the guide following, 
https://github.com/jfowkes/pycutest, 
https://jfowkes.github.io/pycutest/_build/html/index.html

## Execution:
One can execute the framework with main.py. In particular, the codes can be executed in 2 ways:
### Input as a single problem
E.g., prob == ['PALMER7C'], then the codes will generate the comparison between solvers for a single CUTEst problem. The corresponding example plots can be found in general_plot_example folder.
### Input as a problem-set
E.g., prob == ['Unconstrained'], then this codes will generate the performance profile plots, which is the CDF of the performance, over the problem-set, of every algorithms. The problem-set can be manually changed/set at Line 130, initialize.py. Currently, as you can see in initialize.py, this set only selected 7 CUTEst problems for playing around. The corresponding example plots can be found in pProfile folder.
