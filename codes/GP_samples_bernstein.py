import kernel as ker
import numpy as np
from math import comb

import matplotlib.pyplot as plt

from likelihood import *

def bernstein(t, k, i, combination_coeff):
    return combination_coeff[i][k]*t**k*(1-t)**(N[i]-k)

#Sapmling parameters
n = 50 # nb of point evaluations of the functional inputs
N = [300] # nb of Bernstein basis functions used in the approximation
d = 500 # nb of points to approximate the L2 norm
T = np.linspace(0, 1, d) # integrating variable

# Computing combination coefficients
combination_coeff = [[comb(i,k) for k in range(i+1)] for i in N]

# Simulating GP samples
# GP params
param = [2, 2] # parameters of the GP
jitter = 1e-10  # small number to ensure numerical stability (eigenvalues of K can decay rapidly)

samples = np.zeros((np.size(N),n))
for i in range(np.size(N)):

    # Time independent Samples
    U = np.random.rand(n, 2)
    f = np.zeros((n, d+1))
    for j in range(n):
        f[j] = np.concatenate(([j/n], [np.sum([np.cos(U[j,0]*k/N[i])*np.exp(-U[j,1]*k/N[i])*bernstein(t,k,i,combination_coeff) for k in range(N[i]+1)]) for t in T]))
        
    # GP samples
    K = ker.kernel(param, ker.dmatrix(f))
    samples[i] = ker.sample(0, K, jitter, N=1)[:,0]

nb_theta1 = nb_theta2 = 20
theta1_vect = np.linspace(0.2, 2.2, nb_theta1)
theta2_vect = np.linspace(0.2, 2.2, nb_theta2)

loglike_Mat = np.zeros((nb_theta1, nb_theta2))
for i in range(nb_theta1):
    print(i)
    for j in range(nb_theta2):
        param0 = [theta1_vect[i], theta2_vect[j]]
        K0 = ker.kernel(param0, ker.dmatrix(f))
        loglike_Mat[i,j] = modified_log_likelihood(K0, samples[-1], jitter)


Theta1, Theta2 = np.meshgrid(theta1_vect, theta2_vect)
fig, ax = plt.subplots()
cnt = ax.contourf(Theta1, Theta2, loglike_Mat)
cbar = ax.figure.colorbar(cnt, ax = ax)
cnt = ax.contour(Theta1, Theta2, loglike_Mat, np.max(loglike_Mat)*np.linspace(0.98, 1.01, 10), colors = "k", linewidths = 0.5)
ax.clabel(cnt, cnt.levels, inline = True, fontsize = 10)
cbar.ax.set_ylabel("Log likelihood", rotation = -90, va = "bottom")

ax.scatter(param[0], param[1])
idxOpt = np.argmax(loglike_Mat)
ax.scatter(Theta1.flatten()[idxOpt], Theta2.flatten()[idxOpt])

