import numpy as np
# from scipy.spatial.distance import cdist

# kernel functions
def kernel(theta, r):
    return theta[0]**2*np.exp(-np.power(r, 2)/theta[1]**2)

def dmatrix(f):
    n, p = f.shape
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i,j] = (f[i,1] - f[j,1])**2 + (1/p)*np.sum([(f[i,k] - f[j,k])**2 for k in range(2,p)])
    return np.sqrt(D)

# function for generating GP samples
def sample(mu, var, jitter, N):
    """Generate N samples from a multivariate Gaussian \mathcal{N}(mu, var)"""
    L = np.linalg.cholesky(var + jitter*np.eye(var.shape[0])) # cholesky decomposition (square root) of covariance matrix
    f_post = mu + np.dot(L, np.random.normal(size=(var.shape[1], N)))
    return f_post
