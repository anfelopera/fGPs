import kernel as ker
import numpy as np
import matplotlib.pyplot as plt

#Sapmling parameters
n = 12
N = [10,20,30,40,50,100]

# Simulating GP samples
# GP params
param = [1, 1] # parameters of the GP

samples = np.zeros((np.size(N),n))
for i in range(np.size(N)):
    x = np.linspace(0, 1, N[i]) # vector of inputs

    # Samples
    n = 12
    U = np.random.rand(n,2)
    f = np.zeros((n,N[i]))
    for j in range(n):
        f[j] = np.cos(U[j,0]*x)*np.exp(-U[j,1]*x)
        
    # GP samples
    K = ker.kernel(param, ker.dmatrix(f))
    #np.random.seed(9) # (10, 1), (12, 9),
    jitter = 1e-10  # small number to ensure numerical stability (eigenvalues of K can decay rapidly)
    samples[i] = ker.sample(0, K, jitter, N=1)[:,0]


# plot
#plt.plot(range(nsamples), samples);
# plt.scatter(X, Y, color = "black"); plt.axis('off')
# plt.savefig(r'GPsamples.pdf', bbox_inches='tight', dpi=300);
#plt.show()
