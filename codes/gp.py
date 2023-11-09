import numpy as np
import matplotlib.pylab as plt

n=1000

def kernel(theta,r):
    return theta[1]**2*np.exp(-r**2/theta[2]**2)

def dmatrix(f):
    n,p = f.shape
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D[i,j] = (f[i,1] - f[j,1])**2+(1/p)*np.sum([(f[i,k] - f[j,k])**2 for k in range(2,p)])
    return np.sqrt(D)

def GP(r,n):
    power_spectrum = np.fft.rfft(r)
    gaussian_white_noise = np.random.normal(0, 1 ,n)
    return np.fft.irfft(np.sqrt(power_spectrum)*np.fft.rfft(gaussian_white_noise))

r = np.zeros(n)
for i in range(n):
    r[i] = np.exp(-((i-n/2)/n)**2/2)  



plt.figure(1)
plt.plot(field)
plt.show()



R=kernel(dmatrix(),(1,1))
