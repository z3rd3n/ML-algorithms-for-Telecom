import numpy as np


def em(y, X_init, pX_init, sigma2_init, delta_init, num_iter):
    
    N = y.size 
    M = X_init.size 
    
    y = y.reshape((N,1))
    X = X_init.reshape((1,M))
    pX = pX_init.reshape((1,M))
    sigma2 = sigma2_init
    delta = delta_init

    for _ in range(num_iter):

        tmp = pX * circularGauss(sigma2, y, X, delta)
        Q_xy = tmp / np.sum( tmp, axis=1 ).reshape(-1,1) 

        pX = np.sum(Q_xy, axis=0) / N

        delta = np.sum( np.sum( Q_xy * (y) * np.conjugate(X) ) ) / np.sum( np.sum( Q_xy * (np.abs(X))**2 ) )

        sigma2 = np.sum( np.sum( Q_xy * np.abs(y-delta*X)**2 )) / (2*N)
    
    data = {'X': X, 'pX': pX, 'sigma2': sigma2, 'delta': delta}
    return data    

def circularGauss(var, y, x, delta):
    return np.exp(-(np.abs(y-delta*x)**2)/var)/(var*np.pi)


M=4
X_init = np.array([-0.707-0.707j, -0.707+0.707j, 0.707-0.707j, 0.707+0.707j])
pX_init = np.ones(M)/M 
sigma2_init = 1
delta_init = 1
num_iter = 200
np.random.seed(0)
y = np.load('EM_data.npy')
params_out = em(y, X_init, pX_init, sigma2_init, delta_init, num_iter)
print(params_out)
