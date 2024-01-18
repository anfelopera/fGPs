import numpy as np
from scipy.linalg import sqrtm

import kernel as ker

#Sampling parameters
nb_reps = 1000                      # np of random replicates
tab_n = [10,20,40,80,160]           # nb of evaluations of the functional inputs
N = np.array([10,20,40,80,160])     # nb of basis functions used in the approximation
param0 = np.array([1., 0.5])        # GP covariance parameters
jitter = 1e-12                      # jitter to ensure numerical stability for inversions


expName = "./data_samples/GP_samples/GP_samples_" 


for n in tab_n:
    print("n = ", n)
    for k in range(np.size(N)):
        print("N = ", N[k])
        new_results = np.zeros((nb_reps+1, 8))
    
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
        
            ## Generating covariance matrices
            distf = ker.dmatrix(f) # distances between functional inputs
            M = np.sqrt(n)*np.real(sqrtm(ker.acov(2,f,0.0001,param0)))
            
            results = np.load((expName  + "nbReps" + str(nb_reps) + "_n" + str(n) + "_N" + str(N[k])+ ".npy"))
            new_results[0] = np.append(param0, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            for i in range(1,new_results.shape[0]):
                new_results[i] = np.concatenate((results[i,:2],[M[0,0],M[0,1],M[1,0],M[1,1],n, N[k]]))  
            np.save((expName  + "nbReps" + str(nb_reps) + "_n" + str(n) + "_N" + str(N[k])), new_results)
