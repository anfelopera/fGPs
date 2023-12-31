# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:54:47 2023

@author: allopera
"""

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import nlopt

from likelihood import *

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
jitter = 1e-12  # small number to ensure numerical stability (eigenvalues of K can decay rapidly)
def sample(mu, var, jitter, N):
    """Generate N samples from a multivariate Gaussian \mathcal{N}(mu, var)"""
    L = np.linalg.cholesky(var + jitter*np.eye(var.shape[0])) # cholesky decomposition (square root) of covariance matrix
    f_post = mu + np.dot(L, np.random.normal(size=(var.shape[1], N)))
    return f_post

# Simulating GP samples
## GP params
nobs = 100
param = np.array([1, 0.4]) # parameters of the GP
x = np.linspace(0, 1, nobs).reshape(-1,1) # vector of inputs

## GP sample
# nsamples = 1
K = SEKernel(x, x, param)
np.random.seed(0)
y = sample(0.*x, K, jitter, N = 1)
plt.scatter(x, y);

# Covariance parameter estimation
param0 = np.array([1, 0.5]) # initial parameters of the GP
K2 = SEKernel(x, x, param0)
multistart = 10 # nb of multistarts

## checking the likelihood
print(modified_log_likelihood(K2, y, jitter))

## optimizing the likelihood
opt_res = maximum_likelihood(param0, np.array([1e-6, 1e-6]), np.array([10., 10.]), [0, 1],
                             jitter, cov_matrix_function, x, y, multistart, opt_method = "Powell")
print(opt_res["hat_theta"])
