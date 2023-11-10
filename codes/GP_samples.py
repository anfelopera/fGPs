import kernel as ker
import numpy as np

#Sapmling parameters
n = 12
N = [10,20,30,40,50,100]
d = 500

# Simulating GP samples
# GP params
param = [1, 1] # parameters of the GP

samples = np.zeros((np.size(N),n))
for i in range(np.size(N)):

    # Non random Samples
    x = np.linspace(0, 1, N[i]) # vector of sample points
    U = np.random.rand(n,2)
    f = np.zeros((n,N[i]+1))
    for j in range(n):
        f[j] = np.concatenate(([j/n],np.cos(U[j,0]*x)*np.exp(-U[j,1]*x)))
        
    # GP samples
    K = ker.kernel(param, ker.dmatrix(f))
    jitter = 1e-10  # small number to ensure numerical stability (eigenvalues of K can decay rapidly)
    samples[i] = ker.sample(0, K, jitter, N=1)[:,0]
