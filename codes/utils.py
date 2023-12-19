# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 16:42:01 2023

@author: allopera
"""

import matplotlib.pyplot as plt
from likelihood import *

## Computing the log-likelihood landscape
def landscape(likelihood, kernel, distf, samples,  jitter, 
              nbgrid, param0, param_lb, param_ub, paramOptMLE):
    print("#### Computing the log-likelihood landscape ####")
    nb_theta1 = nb_theta2 = nbgrid # nb grid evaluations
    theta1_vect = np.linspace(param_lb[0], param_ub[0], nb_theta1)
    theta2_vect = np.linspace(param_lb[1], param_ub[1], nb_theta2)
    Theta2, Theta1 = np.meshgrid(theta2_vect, theta1_vect)
    
    loglike_Mat = np.zeros((nb_theta1, nb_theta2))
    for i in range(nb_theta1):
        # print(i)
        for j in range(nb_theta2):
            param = [theta1_vect[i], theta2_vect[j]]
            K = kernel(param, distf)
            loglike_Mat[i,j] = modified_log_likelihood(K, samples, jitter)
    
    idxOpt = np.argmax(loglike_Mat)
    paramOpt = [Theta1.flatten()[idxOpt], Theta2.flatten()[idxOpt]]
    #print(paramOpt)
    
    # plot
    fig, ax = plt.subplots()
    cnt = ax.contourf(Theta1, Theta2, loglike_Mat)
    cbar = ax.figure.colorbar(cnt, ax = ax)
    cnt = ax.contour(Theta1, Theta2, loglike_Mat,
                     np.max(loglike_Mat)*np.linspace(0.98, 1, 5),
                     colors = "k", linewidths = 0.5)
    ax.clabel(cnt, cnt.levels, inline = True, fontsize = 10)
    cbar.ax.set_ylabel("Log likelihood", rotation = -90, va = "bottom")
    ax.scatter(param0[0], param0[1], label='Ground truth') # true parameters
    ax.scatter(paramOptMLE[0], paramOptMLE[1], label='MLE (optim)') # estimated parameters via optim
    ax.scatter(paramOpt[0], paramOpt[1], label='MLE (grid)') # estimated parameters    
    #plt.legend()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5)
    plt.show();