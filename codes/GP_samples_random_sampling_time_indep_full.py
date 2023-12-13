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
nb_reps = 10        # np of random replicates
n = 100             # nb of evaluations of the functional inputs
N = np.array([50, 100]) # nb of basis functions used in the approximation
d = 500             # nb of points to approximate the L2 norm
param0 = np.array([1, 0.7])         # GP covariance parameters
param_init = np.array([0.2, 0.2])   # GP covariance parameters
param_lb = np.array([1e-6, 1e-6])   # parameters' lower bound
param_ub = np.array([5., 5.]) # parameters' upper bound
jitter = 1e-12  # jitter to ensure numerical stability for inversions
multistart = 1  # nb of multistarts
T = np.linspace(0, 1, d) # integrating variable

print_landscape = False # to plot the log-likelihood landscapes
verbose = True # to print the optimal parameters per iterations

expName = "GP_samples_rnd_sampling_time_indep_" 

# results = np.zeros((np.size(N)*nb_reps, 3))
# idx = 0
for k in range(np.size(N)):
    print("N = ", N[k])
    results = np.zeros((nb_reps, 3))

    for l in range(nb_reps):
        print("Replicate:", l, end = " ")
        # Time independent Samples
        np.random.seed(k+l)
        x = np.random.rand(N[k], 1) # vector of sample points 
        Y = np.array([partition(t, np.sort(x, axis=None)) for t in T])
        np.random.seed(k+l)
        U = np.random.rand(n, 2)
        f = np.zeros((n, d+1))
        for j in range(n):
            f[j] = np.concatenate(([j/n], np.cos(U[j,0]*Y)*np.exp(-U[j,1]*Y)))
    
        ## Generating GP samples
        distf = ker.dmatrix(f) # distances between functional inputs
        K0 = ker.kernel(param0, ker.dmatrix(f)) # covariance matrix
        np.random.seed(k+l)
        samples = ker.sample(0, K0, jitter, N=1)[:,0] # matrix with samples
        
        ## Computing the log-likelihood landscape
        if print_landscape:
            print("#### Computing the log-likelihood landscape ####")
            nb_theta1 = nb_theta2 = 50 # nb grid evaluations
            theta1_vect = np.linspace(0.1, 2, nb_theta1)
            theta2_vect = np.linspace(0.1, 2, nb_theta2)
            Theta1, Theta2 = np.meshgrid(theta1_vect, theta2_vect)
            
            loglike_Mat = np.zeros((nb_theta1, nb_theta2))
            for i in range(nb_theta1):
                # print(i)
                for j in range(nb_theta2):
                    param = [theta1_vect[i], theta2_vect[j]]
                    K = ker.kernel(param, distf)
                    loglike_Mat[i,j] = modified_log_likelihood(K, samples, jitter)
    
            # plot
            fig, ax = plt.subplots()
            cnt = ax.contourf(Theta1, Theta2, loglike_Mat)
            cbar = ax.figure.colorbar(cnt, ax = ax)
            cnt = ax.contour(Theta1, Theta2, loglike_Mat, np.max(loglike_Mat)*np.linspace(0.98, 1.01, 10), colors = "k", linewidths = 0.5)
            ax.clabel(cnt, cnt.levels, inline = True, fontsize = 10)
            cbar.ax.set_ylabel("Log likelihood", rotation = -90, va = "bottom")
            ax.scatter(param0[0], param0[1]) # true parameters
            idxOpt = np.argmax(loglike_Mat)
            ax.scatter(Theta1.flatten()[idxOpt], Theta2.flatten()[idxOpt]) # estimated 
            
        # MLE
        opt_res = maximum_likelihood(param_init, param_lb, param_ub, [0, 1], jitter,
                                     ker.kernel, distf, samples, multistart, opt_method = "Powell")
        results[l, :] = np.append(opt_res["hat_theta"], np.array(N[k]))
        if verbose:
            print(results[l, :])
        print()
        idx += 1

    np.save((expName + "N" + str(N[k])), results)
    # results_load = np.load((expName + "N" + str(N[k]) + ".npy"))