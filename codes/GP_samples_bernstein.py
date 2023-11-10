import kernel as ker
import numpy as np
from math import comb

def bernstein(t,k,i,combination):
    return combination[i][k]*t**k*(1-t)**(N[i]-k)

#Sapmling parameters
n = 12
N = [10,20,30,40,50,100]
d = 500
y = np.linspace(0,1,d) # integrating variable

# Computing combination coefficients
combination = [[comb(i,k) for k in range(i+1)] for i in N]

# Simulating GP samples
# GP params
param = [1, 1] # parameters of the GP

samples = np.zeros((np.size(N),n))
for i in range(np.size(N)):

    # Time independent Samples
    U = np.random.rand(n,2)
    f = np.zeros((n,d+1))
    for j in range(n):
        f[j] = np.concatenate(([j/n],[np.sum([np.cos(U[j,0]*k/N[i])*np.exp(-U[j,1]*k/N[i])*bernstein(t,k,i,combination) for k in range(N[i]+1)]) for t in y]))
        
    # GP samples
    K = ker.kernel(param, ker.dmatrix(f))
    jitter = 1e-10  # small number to ensure numerical stability (eigenvalues of K can decay rapidly)
    samples[i] = ker.sample(0, K, jitter, N=1)[:,0]
