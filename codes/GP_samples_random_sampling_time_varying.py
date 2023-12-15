import kernel as ker
import numpy as np

import matplotlib.pyplot as plt

from likelihood import *

def partition(t,X):
    i = 0
    
    if t < X[0]:
        return 0
    
    while i < (np.size(X)-1):
        if X[i]<= t and t < X[i+1]:
            return X[i]
        i = i +1
    
    return 1


#Sapmling parameters
n = 50
# N = [10,20,30,40,50,100]
N = [100]
d = 500
y = np.linspace(0,1,d) # integrating variable

# Simulating GP samples
# GP params
param = np.array([2, 0.3])         # GP covariance parameters
param_init = np.array([0.2, 0.2])   # GP covariance parameters
param_lb = np.array([1e-2, 1e-2])   # parameters' lower bound
param_ub = np.array([3., 1.]) # parameters' upper bound
jitter = 1e-12  # jitter to ensure numerical stability for inversions
multistart = 5  # nb of multistarts

samples = np.zeros((np.size(N),n))
for i in range(np.size(N)):

    # Time varying Samples
    U = np.random.rand(n,2)
    f = np.zeros((n,d+1))
    for j in range(n):
        x = np.random.rand(N[i],1) # vector of sample points
        Y = np.array([partition(t, np.sort(x, axis=None)) for t in y])
        f[j] = np.concatenate(([j/n],np.cos(U[j,0]*Y)*np.exp(-U[j,1]*Y)))
        
    # GP samples
    distf = ker.dmatrix(f)
    K = ker.kernel(param, distf)
    jitter = 1e-15  # small number to ensure numerical stability (eigenvalues of K can decay rapidly)
    samples[i] = ker.sample(0, K, jitter, N=1)[:,0]
    
nb_theta1 = nb_theta2 = 50
theta1_vect = np.linspace(param_lb[0], param_ub[0], nb_theta1)
theta2_vect = np.linspace(param_lb[1], param_ub[1], nb_theta2)

loglike_Mat = np.zeros((nb_theta1, nb_theta2))
for i in range(nb_theta1):
    print(i)
    for j in range(nb_theta2):
        param0 = [theta1_vect[i], theta2_vect[j]]
        K0 = ker.kernel(param0, distf)
        loglike_Mat[i, j] = modified_log_likelihood(K0, samples[-1], jitter)

Theta2, Theta1 = np.meshgrid(theta2_vect, theta1_vect)
fig, ax = plt.subplots()
cnt = ax.contourf(Theta1, Theta2, loglike_Mat)
cbar = ax.figure.colorbar(cnt, ax = ax)
cnt = ax.contour(Theta1, Theta2, loglike_Mat, np.max(loglike_Mat)*np.linspace(0.98, 1.01, 10), colors = "k", linewidths = 0.5)
ax.clabel(cnt, cnt.levels, inline = True, fontsize = 10)
cbar.ax.set_ylabel("Log likelihood", rotation = -90, va = "bottom")

ax.scatter(param[0], param[1])
idxOpt = np.argmax(loglike_Mat)
ax.scatter(Theta1.flatten()[idxOpt], Theta2.flatten()[idxOpt])
print(Theta1.flatten()[idxOpt], Theta2.flatten()[idxOpt])

opt_res = maximum_likelihood(param_init, param_lb, param_ub, [0, 1],
                             jitter, ker.kernel, distf, samples[-1], multistart, opt_method = "Powell")
print(np.append(opt_res["hat_theta"], np.array(N[-1])))
