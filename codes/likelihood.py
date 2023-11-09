# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:55:48 2023

@author: allopera
"""

import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize

# functions used for computing the conditional mean and covariance functions
def cond_mean(x, X, Y, kern, param):
    """Conditional GP mean vector
    input:
      x: vector of prediction points
      X: DoE vector
      Y: vector of responses
      kern: kernel function
      param: parameters of the covariance
    output:
      conditional mean
    """
    k_xX = kern(x, X, param)
    k_XX = kern(X, X, param)
    return(np.dot(k_xX, np.matmul(np.linalg.inv(k_XX + 1e-9*np.eye(k_XX.shape[0])), Y)))

def cond_cov(x, X, Y, kern, param):
    """Conditional GP covariance matrix 
    input:
      x: vector of prediction points
      X: DoE vector
      Y: vector of responses
      kern: kernel function
      param: parameters of the covariance
    output:
      conditional covariance
    """
    k_xx = kern(x, x, param)
    k_xX = kern(x, X, param)
    k_XX = kern(X, X, param)
    return(k_xx - np.matmul(k_xX, np.matmul(np.linalg.inv(k_XX+1e-9*np.eye(k_XX.shape[0])), k_xX.transpose())))

def modified_log_likelihood(R, y_obs, nugget):
    """
    Compute -2*log(likelihood) for a Gaussian vector with mean zero.

    Parameters:
    - R (numpy.ndarray): Covariance matrix.
    - y_obs (numpy.ndarray): Observation vector.
    - nugget (float): Value to add to the covariance matrix of the observations:
                     nugget * largest diagonal element * identity.

    Returns:
    - float: The modified log likelihood value.
    """
    n = len(y_obs)
    R = R + np.max(np.diag(R)) * nugget * np.eye(R.shape[0])
    cR = cholesky(R, lower=False)
    iR = np.linalg.inv(cR) #cho_solve((cR, False), np.eye(n))
    return 2 * np.sum(np.log(np.diag(cR))) + np.sum(y_obs * np.dot(iR, y_obs))


def maximum_likelihood_descent(init_theta, theta_inf, theta_sup, ind_active, nugget, cov_matrix_function, mq_obs, y_obs):
    """
    Perform maximum likelihood estimation using gradient descent.

    Parameters:
    - init_theta (numpy.ndarray): Starting point of the gradient descent.
    - theta_inf (numpy.ndarray): Vector of lower bounds for the optimization.
    - theta_sup (numpy.ndarray): Vector of upper bounds for the optimization.
    - ind_active (list): List of indices of covariance parameters with respect to which we optimize.
    - nugget (float): Numerical nugget variance for matrix inversion.
    - cov_matrix_function (function): Covariance matrix function taking inputs (theta, mq).
    - mq_obs (numpy.ndarray): n*d matrix of quantile values for input points. n=number of inputs, d=number of quantile values.
    - y_obs (numpy.ndarray): Vector of observations of size n.

    Returns:
    - dict: {'hat_theta': The vector hat_theta of the maximum likelihood estimator, 'val': The value of the modified log likelihood at the optimizer.}
    """
    def to_min(param_active, ind_active, init_param, nugget, cov_matrix_function, mq_obs, y_obs):
        param = np.copy(init_param)
        param[ind_active] = param_active
        R = cov_matrix_function(param, mq_obs)
        return -modified_log_likelihood(R, y_obs, nugget)

    # Gradient descent
    res_opt = minimize(
        fun=to_min,
        x0=init_theta[ind_active],
        method="COBYLA",
        bounds=list(zip(theta_inf, theta_sup)),
        args=(ind_active, init_theta, nugget, cov_matrix_function, mq_obs, y_obs),
    )

    hat_param = np.copy(init_theta)
    hat_param[ind_active] = res_opt.x
    return {"hat_theta": hat_param, "val": -res_opt.fun}

# Example usage:
# result = maximum_likelihood_descent(init_theta, theta_inf, theta_sup, ind_active, nugget, cov_matrix_function, mq_obs, y_obs)
# hat_theta = result["hat_theta"]
# val = result["val"]


def maximum_likelihood(init_theta, theta_inf, theta_sup, ind_active, nugget, cov_matrix_function, mq_obs, y_obs, k):
    """
    Perform maximum likelihood estimation using the best of k gradient descents.

    Parameters:
    - init_theta (numpy.ndarray): Starting point of the gradient descent.
    - theta_inf (numpy.ndarray): Vector of lower bounds for the optimization.
    - theta_sup (numpy.ndarray): Vector of upper bounds for the optimization.
    - ind_active (list): List of indices of covariance parameters with respect to which we optimize.
    - nugget (float): Numerical nugget variance for matrix inversion.
    - cov_matrix_function (function): Covariance matrix function taking inputs (theta, mq).
    - mq_obs (numpy.ndarray): n*d matrix of quantile values for input points. n=number of inputs, d=number of quantile values.
    - y_obs (numpy.ndarray): Vector of observations of size n.
    - k (int): Number of gradient descents.

    Returns:
    - dict: {'hat_theta': The vector hat_theta of the maximum likelihood estimator,
             'val': The criterion value at the maximum likelihood estimator,
             'm_hat_theta': The matrix of hat_theta for the different gradient descents,
             'v_val': The vector of optimum values for the different gradient descents.}
    """
    m_hat_theta = np.empty((k, len(init_theta)))
    v_val = np.zeros(k)

    for i in range(k):
        init_theta[ind_active] = theta_inf[ind_active] + np.random.uniform(
            size=len(theta_inf[ind_active])
        ) * (theta_sup[ind_active] - theta_inf[ind_active])

        res_opt = maximum_likelihood_descent(
            init_theta, theta_inf, theta_sup, ind_active, nugget, cov_matrix_function, mq_obs, y_obs
        )

        m_hat_theta[i, :] = res_opt["hat_theta"]
        v_val[i] = res_opt["val"]

    min_index = np.argmin(v_val)
    hat_theta = m_hat_theta[min_index, :]
    val = v_val[min_index]

    return {"m_hat_theta": m_hat_theta, "v_val": v_val, "hat_theta": hat_theta, "val": val}

# Example usage:
# result = maximum_likelihood(init_theta, theta_inf, theta_sup, ind_active, nugget, cov_matrix_function, mq_obs, y_obs, k)
# m_hat_theta = result["m_hat_theta"]
# v_val = result["v_val"]
# hat_theta = result["hat_theta"]
# val = result["val"]






