import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# kernel functions
def SEKernel(x, y, param):
    """ Squared exponential kernel
    input:
      x,y: input vectors
      param: parameters (sigma,theta)
    output:
      covariance matrix cov(x,y)
    """
    sigma2, theta = param[0], param[1]
    dist = cdist(x, y)/theta
    return sigma2*np.exp(-0.5*dist**2)

def matern52Kernel(x, y, param):
    """Matern 5/2 kernel
    input:
      x,y: input vectors
      param: parameters (sigma,theta)
    output:
      covariance matrix cov(x,y)
    """
    sigma2, theta = param[0], param[1]
    dist = cdist(x, y)/theta
    return sigma2*(1+np.sqrt(5)*np.abs(dist)+(5/3)*dist**2)*np.exp(-np.sqrt(5)*np.abs(dist))

def matern32Kernel(x, y, param):
    """Matern 3/2 kernel
    input:
      x,y: input vectors
      param: parameters (sigma,theta)
    output:
      covariance matrix cov(x,y)
    """
    sigma2, theta = param[0], param[1]
    dist = cdist(x, y)/theta
    return sigma2*(1+np.sqrt(3)*np.abs(dist))*np.exp(-np.sqrt(3)*np.abs(dist))

def exponentialKernel(x, y, param):
    """Exponential kernel
    input:
      x,y: input vectors
      param: parameters (sigma,theta)
    output:
      covariance matrix cov(x,y)
    """
    sigma2, theta = param[0], param[1]
    dist = cdist(x, y)/theta
    return sigma2*np.exp(-np.abs(dist))


# function for generating GP samples
jitter = 1e-10  # small number to ensure numerical stability (eigenvalues of K can decay rapidly)
def sample(mu, var, jitter, N):
    """Generate N samples from a multivariate Gaussian \mathcal{N}(mu, var)"""
    L = np.linalg.cholesky(var + jitter*np.eye(var.shape[0])) # cholesky decomposition (square root) of covariance matrix
    f_post = mu + np.dot(L, np.random.normal(size=(var.shape[1], N)))
    return f_post

# Simulating GP samples
# GP params
param = [1, 0.2] # parameters of the GP
x = np.linspace(0, 1, 500).reshape(-1,1) # vector of inputs

# Conditional GP samples
nsamples = 12
K = SEKernel(x, x, param)
np.random.seed(9) # (10, 1), (12, 9), 
samples = sample(0.*x, K, jitter, N=nsamples)

# plot
plt.plot(x, samples);
# plt.scatter(X, Y, color = "black"); plt.axis('off')
# plt.savefig(r'GPsamples.pdf', bbox_inches='tight', dpi=300);
