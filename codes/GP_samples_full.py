import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

from likelihood import *
from utils import *
import kernel as ker

#Sampling parameters
nb_reps = 1000                      # np of random replicates
n = 160                              # nb of evaluations of the functional inputs
N = np.array([10, 20, 40, 80, 160]) # nb of basis functions used in the approximation
d = 500                             # nb of points to approximate the L2 norm
param0 = np.array([1., 0.5])        # GP covariance parameters
param_init = np.array([0.2, 0.2])   # GP covariance parameters
param_lb = np.array([1e-1, 1e-1])   # parameters' lower bound
param_ub = np.array([10., 10.])     # parameters' upper bound
jitter = 1e-12                      # jitter to ensure numerical stability for inversions
multistart = 5                      # nb of multistarts

print_landscape = False # to plot the log-likelihood landscapes
verbose = False         # to print the optimal parameters per iterations

expName = "./data_samples/GP_samples/GP_samples_" 
print("n = ", n)
for k in range(np.size(N)):
    print("N = ", N[k])
    results = np.zeros((nb_reps, 8))

    for l in range(nb_reps):
        if l % 100 == 0:
            print("Replicate:", l)
        ## Non random samples
        x = np.linspace(0, 1, N[k]) # vector of sample points
        np.random.seed(k+l)
        U = np.random.rand(n, 2)
        f = np.zeros((n, N[k]+1)) # functional inputs
        for j in range(n):
            f[j] = np.concatenate(([j/n], np.cos(U[j,0]*x)*np.exp(-U[j,1]*x)))
    
        ## Generating GP samples
        distf = ker.dmatrix(f) # distances between functional inputs
        K0 = ker.kernel(param0, ker.dmatrix(f)) # covariance matrix
        np.random.seed(l)
        samples = ker.sample(0, K0, jitter, N=1)[:,0] # matrix with samples
        # M = np.sqrt(n)*np.real(sqrtm(ker.acov(2,f,0.0001,param0)))
        M = np.real(sqrtm(ker.acov2(2, f, param0)))

        # MLE
        opt_res = maximum_likelihood(param_init, param_lb, param_ub, [0, 1], jitter,
                                     ker.kernel, distf, samples, multistart, opt_method = "Powell")
        results[l, :] = np.append(opt_res["hat_theta"], [M[0,0],M[0,1],M[1,0],M[1,1],n, N[k]])
        if verbose:
            print(results[l, :])
        
        ## Computing the log-likelihood landscape
        if print_landscape:
            nbgrid = 50
            landscape(modified_log_likelihood, ker.kernel, distf, samples,
                      jitter, nbgrid, param0, param_lb, param_ub, opt_res["hat_theta"])
    
    results = np.vstack((np.append(param0, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]), results)) # stacking the ground truth params
    np.save((expName  + "nbReps" + str(nb_reps) + "_n" + str(n) + "_N" + str(N[k])), results)
    # results_load = np.load((expName + "N" + str(N[k]) + ".npy"))
    
