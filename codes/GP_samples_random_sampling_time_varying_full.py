import numpy as np
import matplotlib.pyplot as plt

from likelihood import *
import kernel as ker

def partition(t,X):
    i = 0
    
    if t < X[0]:
        return 0
    
    while i < (np.size(X)-1):
        if X[i]<= t and t < X[i+1]:
            return X[i]
        i = i +1
    
    return 1

#Sampling parameters
nb_reps = 10                        # np of random replicates
n = 100                             # nb of evaluations of the functional inputs
N = np.array([50, 100])             # nb of basis functions used in the approximation
d = 500                             # nb of points to approximate the L2 norm
param0 = np.array([2, 1])           # GP covariance parameters
param_init = np.array([0.2, 0.2])   # GP covariance parameters
param_lb = np.array([1e-1, 1e-1])   # parameters' lower bound
param_ub = np.array([3., 2.])       # parameters' upper bound
jitter = 1e-12                      # jitter to ensure numerical stability for inversions
multistart = 1                      # nb of multistarts
T = np.linspace(0, 1, d)            # integrating variable

print_landscape = False # to plot the log-likelihood landscapes
verbose = True # to print the optimal parameters per iterations

expName = "./data_samples/GP_samples_rnd_sampling_time_vary_" 

for k in range(np.size(N)):
    print("N = ", N[k])
    results = np.zeros((nb_reps, 3))

    for l in range(nb_reps):
        print("Replicate:", l, end = " ")
        # Time varying Samples
        np.random.seed(k+l)
        U = np.random.rand(n,2)
        f = np.zeros((n,d+1))
        for j in range(n):
            np.random.seed(k+l+j) 
            x = np.random.rand(N[k], 1) # vector of sample points
            Y = np.array([partition(t, np.sort(x, axis=None)) for t in T])
            f[j] = np.concatenate(([j/n], np.cos(U[j,0]*Y)*np.exp(-U[j,1]*Y)))
    
        ## Generating GP samples
        distf = ker.dmatrix(f) # distances between functional inputs
        K0 = ker.kernel(param0, ker.dmatrix(f)) # covariance matrix
        np.random.seed(k+l)
        samples = ker.sample(0, K0, jitter, N=1)[:,0] # matrix with samples
            
        # MLE
        opt_res = maximum_likelihood(param_init, param_lb, param_ub, [0, 1], jitter,
                                     ker.kernel, distf, samples, multistart, opt_method = "Powell")
        results[l, :] = np.append(opt_res["hat_theta"], np.array(N[k]))
        if verbose:
            print(results[l, :])
        print()
        
        ## Computing the log-likelihood landscape
        if print_landscape:
            nbgrid = 50
            landscape(modified_log_likelihood, ker.kernel, distf, samples,
                      jitter, nbgrid, param0, param_lb, param_ub, opt_res["hat_theta"])

    np.save((expName + "n" + str(n) + "_N" + str(N[k])), results)
    # results_load = np.load((expName + "N" + str(N[k]) + ".npy"))