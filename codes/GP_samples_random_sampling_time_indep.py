import kernel as ker
import numpy as np

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
n = 12
N = [10,20,30,40,50,100]
d = 500
y = np.linspace(0,1,d) # integrating variable

# Simulating GP samples
# GP params
param = [1, 1] # parameters of the GP

samples = np.zeros((np.size(N),n))
for i in range(np.size(N)):

    # Time independent Samples
    x = np.random.rand(N[i],1) # vector of sample points 
    Y = np.array([partition(t, np.sort(x, axis=None)) for t in y])
    U = np.random.rand(n,2)
    f = np.zeros((n,d+1))
    for j in range(n):
        f[j] = np.concatenate(([j/n],np.cos(U[j,0]*Y)*np.exp(-U[j,1]*Y)))
        
    # GP samples
    K = ker.kernel(param, ker.dmatrix(f))
    jitter = 1e-10  # small number to ensure numerical stability (eigenvalues of K can decay rapidly)
    samples[i] = ker.sample(0, K, jitter, N=1)[:,0]
