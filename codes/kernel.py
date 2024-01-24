import numpy as np
from scipy.linalg import cholesky, cho_solve, eig, solve_triangular
# from scipy.spatial.distance import cdist

# kernel functions
def kernel(theta, r):
    # return theta[0]**2*np.exp(-np.power(r, 2)/theta[1]**2)
    return theta[0]*np.exp(-np.power(r, 2)/theta[1]**2)


def dkernel(theta, r, idx_dtheta):
    if idx_dtheta == 0:
        # return 2 * kernel(theta, r) / theta[0]
        return kernel(theta, r) / theta[0]
    else:
        return 2 * kernel(theta, r) * np.power(r, 2) / theta[1]**3  

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

def acov2(p, f, param0):
    dmat = dmatrix(f)           #distance matrix of the sample points 
    R0 = kernel(param0, dmat)   #R_{theta_0}
    
    # computing the Cholesky matrix and the derivatives of R0
    jitter = 1e-9
    cR0 = cholesky(R0 + jitter*np.eye(R0.shape[0]), lower = True)
    dR = [dkernel(param0, dmat, 0), dkernel(param0, dmat, 1)]
    
    # Computing M via Cholesky and solve triangular 
    cR0t = np.transpose(cR0)
    P = [solve_triangular(cR0, dR[0], lower=True),
         solve_triangular(cR0, dR[1], lower=True)]
    Q = [solve_triangular(cR0t, P[0]),
         solve_triangular(cR0t, P[1])]
    
    M = np.zeros((p,p))
    n = f.shape[0]
    for i in range(p):
        for j in range(p):
            M[i,j] = (1/(2*n))*np.trace(np.matmul(Q[i], Q[j]))
    return M

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
    jitter = 1e-6

    n = R0.shape[0]
    # inv2 = np.linalg.inv(R0 + jitter*np.eye(R0.shape[0]))
    # print(min(eig(R0 + jitter*np.eye(n))[0]))
    # print(np.max(R0 - np.transpose(R0)))
    cR0 = cholesky(R0 + jitter*np.eye(n), lower = True)
    #inv = cho_solve((cR0, True), np.eye(n))

    off = np.zeros(p)   
    off[i] = h
    Ri = kernel(param0+off, dmat) #Increrment in direction i   

    off = np.zeros(p)
    off[j] = h
    Rj = kernel(param0+off, dmat) #Increrment in direction j
    
    dRi = (Ri - R0)/h #First order derivative in direction i
    dRj = (Rj - R0)/h #First order derivative in direction j

    # return np.matmul(inv,np.matmul(dRj,np.matmul(inv,dRi)))
    
    cR0t = np.transpose(cR0)
    Pj = solve_triangular(cR0, dRj, lower=True)
    Pi = solve_triangular(cR0, dRi, lower=True)
    Oj = solve_triangular(cR0t, Pj)
    Oi = solve_triangular(cR0t, Pi)
    Q = np.matmul(Oi, Oj)
    
    return(Q)
    
    
