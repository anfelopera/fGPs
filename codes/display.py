import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

mpl.rc('text', usetex = True)   # needed for TeX support in graphs
sns.set_theme(style = "ticks")  # Seaborn style - not needed ?
palette = sns.color_palette()   # To use the default color palette

#os.chdir("./data_samples") #simulation within this folder must be stored as "GP_samples_Nk.npy" with k an integer
#and only contain simulation results for one type (either fixed sampling points, random but time independent sampling points,
# time varying sampling points or Bernstein)

dir_names = os.listdir("./data_samples/")

expIdx = 3
data_folder = "./data_samples/" + dir_names[expIdx] + "/"
files = os.listdir(data_folder)

data = []
for s in files: # loads the simulation results
    N = (s.split("_"))[-1][:-4][1:] # eliminates the part 'GP_samples_N' as well as the extension   
    n = (s.split("_"))[-2][1:]
    data = data + [np.load(data_folder + s)]
data = np.concatenate(data)
df = pd.DataFrame(data, columns=['theta_1', 'theta_2', 'n', 'N']) #dataframe storing all the simulation result

nlist = np.sort(df['n'].dropna().unique())
Nlist = np.sort(df['N'].dropna().unique())
for j,n in enumerate(nlist): # for each value of n and N, we plot a boxplot for both theta_1 and theta_2 and we save it   
    for i,N in enumerate(Nlist): 
        df_N = df.loc[(df['n'] == n) & (df['N'] == N) & (df['theta_1'] < 3)]        
        df_N2 = df.loc[(df['n'] == n) & (df['N'] == N) & (df['theta_1'] >= 3)]
        plt.figure()
        plt.title(r'$n = ' + str(int(n)) +  ', N = ' + str(int(N)) + '$')
        # ax = sns.boxplot(df_N[['theta_1','theta_2']], width=.15, whis=(5,95))
        ax = sns.violinplot(data=df_N[['theta_1', 'theta_2']], alpha = 0.7)
        sns.stripplot(data=df_N2[['theta_1', 'theta_2']], jitter = 0.02, color='black', alpha=0.3)
        ax.set_xticklabels([r'$\theta _1$',r'$\theta _2$'])
        # ax.set_xlabel("Covariance parameters")
        # ax.set_ylabel("Estimated paramaters")
        ax.set_ylim([0, 4])
        ax.yaxis.grid('True')
        ax.axhline(y = data[0, 0], xmin = 0.02, xmax = 0.48, ls='--', c='red', lw = 1.2)
        ax.axhline(y = data[0, 1], xmin = 0.52, xmax = 0.98, ls='--', c='red', lw = 1.2)
        #plt.savefig('boxplot_n' + n + '_N' + str(N) + '.png')
        plt.show()
    
for j,n in enumerate(nlist): # for each value of n and N, we plot a boxplot for both theta_1 and theta_2 and we save it   
    for i,N in enumerate(Nlist): 
        df_N = df.loc[(df['n'] == n) & (df['N'] == N)]
        # df_N = df.loc[(df['n'] == n) & (df['N'] == N) & (df['theta_1'] <= 4)]
        median_vals = df_N.median()
    
        plt.figure()
        plt.title(r'$n = ' + str(int(n)) +  ', N = ' + str(int(N)) + '$')
        # ax = sns.histplot((df.loc[df['N'] == N])['theta_1'])
        ax = sns.kdeplot(data = df_N, x = "theta_1", fill = True, alpha = 0.3, lw = 1.5)
        ax.axvline(x = median_vals[0], ymin = 0, ymax = 1, ls = '--', lw = 1.5) 
        # sns.kdeplot(data = df_N, color = palette[1], fill = True, alpha = 0.3, lw = 1.5)
        ax.axvline(x = data[0, 0], ymin = 0, ymax = 1, color = palette[1], lw = 1.5) 
        ax.set_xlim([-2, 4])
        ax.set_xlabel(r'$\theta_1$')
        ax.set_ylabel("")
        #plt.savefig('kdeplot_theta_1_n' + n + '_N' + str(N) + '.png')
        plt.show()
    
        plt.figure()
        plt.title(r'$n = ' + str(int(n)) +  ', N = ' + str(int(N)) + '$')
        # ax = sns.histplot((df.loc[df['N'] == N])['theta_1'])
        ax = sns.kdeplot(data = df_N, x = "theta_2", fill = True, alpha = 0.3, lw = 1.5)
        ax.axvline(x = median_vals[1], ymin = 0, ymax = 1, ls='--', lw = 1.5) 
        # sns.kdeplot(data = df_N, color = palette[1], fill = True, alpha = 0.3, lw = 1.5)
        ax.axvline(x = data[0, 1], ymin = 0, ymax = 1, color = palette[1], lw = 1.5) 
        ax.set_xlim([0, 1])
        ax.set_xlabel(r'$\theta_2$')
        ax.set_ylabel("")
        #plt.savefig('kdeplot_theta_2_n' + n + '_N' + str(N) + '.png')
        plt.show()
    
        # plt.figure()
        # plt.title('N = ' + str(N))
        # # ax = sns.histplot((df.loc[df['N'] == N])['theta_1'])
        # ax = sns.kdeplot(data = df_N, x = "theta_1", fill = True, alpha = 0.3, lw = 1.5)
        # ax.axvline(x = median_vals[0], ymin = 0, ymax = 1, ls = '--', lw = 1.5) 
        # ax.axvline(x = data[0, 0], ymin = 0, ymax = 1, color = palette[2], lw = 1.5) 
        # sns.kdeplot(data = df_N, x = "theta_2", color = palette[1], fill = True, alpha = 0.3, lw = 1.5)
        # ax.axvline(x = median_vals[1], ymin = 0, ymax = 1, ls = '--', color = palette[1], lw = 1.5) 
        # ax.axvline(x = data[0, 1], ymin = 0, ymax = 1, color = palette[3], lw = 1.5) 
        # ax.set_xlabel(r'$\theta$')
        # ax.set_ylabel("")
        #plt.savefig('kdeplot_theta_n' + n + '_N' + str(N) + '.png')
        # plt.show()
    
    

        
        
        
    


