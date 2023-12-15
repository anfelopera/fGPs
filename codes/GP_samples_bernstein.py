import kernel as ker
import numpy as np
from math import comb

import matplotlib.pyplot as plt

from likelihood import *

def bernstein(t, k, i, combination_coeff):
    return combination_coeff[i][k]*t**k*(1-t)**(N[i]-k)

#Sapmling parameters
n = 50 # nb of point evaluations of the functional inputs
N = [100] # nb of Bernstein basis functions used in the approximation
d = 500 # nb of points to approximate the L2 norm
T = np.linspace(0, 1, d) # integrating variable

# Computing combination coefficients
combination_coeff = [[comb(i,k) for k in range(i+1)] for i in N]

# Simulating GP samples
# GP params
param = np.array([0.7, 0.3])         # GP covariance parameters
param_init = np.array([0.2, 0.2])   # GP covariance parameters
param_lb = np.array([1e-2, 1e-2])   # parameters' lower bound
param_ub = np.array([3., 1.]) # parameters' upper bound
jitter = 1e-12  # jitter to ensure numerical stability for inversions
multistart = 5  # nb of multistarts

samples = np.zeros((np.size(N),n))
for i in range(np.size(N)):

    # Time independent Samples
    U = np.random.rand(n, 2)
    f = np.zeros((n, d+1))
    for j in range(n):
        f[j] = np.concatenate(([j/n], [np.sum([np.cos(U[j,0]*k/N[i])*np.exp(-U[j,1]*k/N[i])*bernstein(t,k,i,combination_coeff) for k in range(N[i]+1)]) for t in T]))
        
    # GP samples
    distf = ker.dmatrix(f)
    K = ker.kernel(param, distf)
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
