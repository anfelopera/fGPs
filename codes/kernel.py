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

# computes the asymptotic covariance matrix
def acov(p,f,h,param0):
    M = np.zeros((p,p))
    n = f.shape[0]

    for i in range(p):
        for j in range(p):
            M[i,j] = (1/(2*n))*np.trace(prodmat(p,i,j,f,h,param0))
    return M

#computes each element of the asymptotic covariance matrix    
def prodmat(p,i,j,f,h,param0):
    dmat = dmatrix(f)           #distance matrix of the sample points 
    R0 = kernel(param0, dmat)   #R_{theta_0}
    jitter = 1e-5

    inv = np.linalg.inv(R0 + jitter*np.eye(R0.shape[0]))

    off = np.zeros(p)   
    off[i] = h
    Ri = kernel(param0+off, dmat) #Increrment in direction i   

    off = np.zeros(p)
    off[j] = h
    Rj = kernel(param0+off, dmat) #Increrment in direction j
    
    dRi = (Ri - R0)/h #First order derivative in direction i
    dRj = (Rj - R0)/h #First order derivative in direction j

    return np.matmul(inv,np.matmul(dRj,np.matmul(inv,dRi)))
