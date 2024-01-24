import numpy as np
from scipy.linalg import sqrtm

import kernel as ker

#Sampling parameters
nb_reps = 1000                      # np of random replicates
tab_n = [40]       # nb of evaluations of the functional inputs
N = np.array([160]) # nb of basis functions used in the approximation
param0 = np.array([1., 0.5])        # GP covariance parameters
jitter = 1e-12                      # jitter to ensure numerical stability for inversions

expIdx = 0
if expIdx == 0:
    expName = "./data_samples/GP_samples/GP_samples_" 
elif expIdx == 1:
    expName = "./data_samples/GP_samples_Bernstein/GP_samples_Bernstein_" 
elif expIdx == 2:
    expName = "./data_samples/GP_samples_rnd_sampling_time_indep/GP_samples_rnd_sampling_time_indep_" 
elif expIdx == 3:
    expName = "./data_samples/GP_samples_rnd_sampling_time_vary/GP_samples_rnd_sampling_time_vary_" 

for n in tab_n:
    print("n = ", n)
    for k in range(np.size(N)):
        print("N = ", N[k])
        new_results = np.zeros((nb_reps+1, 8))
        param0[0] = param0[0]**2
        new_results[0,:] = np.append(param0, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        results = np.load((expName  + "nbReps" + str(nb_reps) + "_n" + str(n) + "_N" + str(N[k])+ ".npy"))
        results[:,0] = results[:,0]**2
        
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
            # M = np.real(sqrtm(ker.acov(2,f,0.0001,param0)))
            M = np.real(sqrtm(ker.acov2(2, f, param0)))
            #print(M)
            
            new_results[l+1,:] = np.concatenate((results[l+1,:2],[M[0,0],M[0,1],M[1,0],M[1,1],n, N[k]]))            
            if l % 100 == 0:
                np.save((expName  + "nbReps" + str(nb_reps) + "_n" + str(n) + "_N" + str(N[k])), new_results)
                
        np.save((expName  + "nbReps" + str(nb_reps) + "_n" + str(n) + "_N" + str(N[k])), new_results)