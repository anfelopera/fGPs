import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

mpl.rc('text', usetex = True) #needed for TeX support in graphs
sns.set_theme(style="ticks") #Seaborn style - not needed ?

os.chdir("./data_samples") #simulation within this folder must be stored as "GP_samples_Nk.npy" with k an integer
#and only contain simulation results for one type (either fixed sampling points, random but time independent sampling points,
# time varying sampling points or Bernstein)
files = os.listdir()

data = []

for s in files: #loads the simulation results
    N = (s.split("_"))[-1][:-4][1:] #eliminates the part 'GP_samples_N' as well as the extension   
    data = data + [np.load(s)]
data = np.concatenate(data)

df = pd.DataFrame(data, columns=['theta_1', 'theta_2', 'N']) #dataframe storing all the simulation result

for i,N in enumerate(df['N'].unique()): #for each value of N, we plot a boxplot for both theta_1 and theta_2 and we save it
    plt.figure()
    plt.title('N = '+str(N))
    ax = sns.boxplot((df.loc[df['N'] == N])[['theta_1','theta_2']], width=.15, whis=(10,90))
    ax.set_xticklabels([r'$\theta _1$',r'$\theta _2$'])
    ax.set_xlabel("Covariance parameters")
    ax.set_ylabel("Estimated paramaters")
    ax.yaxis.grid('True')
    ax.axhline(y = 0.7,xmin = 0, xmax = 1, ls='--', c='black',lw=1)
    ax.axhline(y = 1,xmin = 0, xmax = 1, ls='--', c='black',lw=1)
    #plt.savefig('boxplot_N_'+str(N)+'.png')
    plt.show()
    
for i,N in enumerate(df['N'].unique()):#for each value of N, we plot a histogram for both theta_1 and theta_2 and we save it
    plt.figure()
    plt.title('N = '+str(N))
    ax = sns.histplot((df.loc[df['N'] == N])['theta_1'])
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel("Frequency density")
    ax.axvline(x = 1,ymin = 0, ymax = 1, ls='--', c='black',lw=1) 
    #plt.savefig('histplot_theta_1_N_'+str(N)+'.png')
    plt.show()
    
    plt.figure()
    plt.title('N = '+str(N))
    ax = sns.histplot((df.loc[df['N'] == N])['theta_2'])
    ax.set_xlabel(r'$\theta_2$')
    ax.set_ylabel("Frequency density")
    ax.axvline(x = 0.7,ymin = 0, ymax = 1, ls='--', c='black',lw=1) 
    #plt.savefig('histplot_theta_2_N_'+str(N)+'.png')
    plt.show()
    

        
        
        
    


