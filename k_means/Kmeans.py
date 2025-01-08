import numpy as np
import matplotlib.pyplot as plt
import scipy


def kmeans(y, C0, iters):
    """
    Calculates the clusters of the observations stored in 'y' performing
    'iters' iterations using an initial cluster C0.
    
    Returns a matrix of dimension (iters+1 x K) with the calculated 
    centroids for all iterations
    """

    N = y.size
    K = C0.size

    y = y.reshape(N,1) # make it Nby1
    C = np.zeros([iters+1, K])
    C[0, :] = C0 # make it something by 1 and assign to the first iter

    for it in range(1,iters+1):
        
        # Decision Step
        tmp = np.abs(y - C[it-1, :].reshape(1,-1)) # y is (N,1) and C is (1,K), since it is a row vector, we convert it
        assignment_idx = np.argmin(tmp,axis=1)

        # M-Step
        for j in range(M):
            idx = (assignment_idx == j)
            if np.sum(idx) > 0:
                C[it, j] = np.sum(y[idx]) / np.sum(idx)

    return C

M = 8
N = 10000
SNRdB = 15
SNR = 10**(SNRdB/10
           )
X = np.arange(-M+1, M+1, 2)
nu = 0.0641
pX = np.ones(M)/M
x = np.random.choice(X, p=pX, size=N)
P = np.mean((x)**2)
sigma2 = P/SNR

y = x +np.random.randn(N)*np.sqrt(sigma2)

def awgn_pY(y,X,pX,sigma2):
    X = X.reshape((1,X.size))
    pX = pX.reshape((1,pX.size))
    y=y.reshape((y.size,1))

    pY = np.sum(pX*scipy.stats.norm.pdf(y,X,np.sqrt(sigma2)), axis=1)
    return pY

yrange = np.linspace(np.min(y), np.max(y),1000)
py = awgn_pY(yrange,X,pX,sigma2)
plt.plot(yrange,py)
plt.show()


# run kmeans
C0_init = np.linspace(-1,1,M)
num_iter = 20
C = kmeans(y,C0_init, num_iter)

for k in range(M):
    plt.scatter(range(num_iter+1), C[:,k])

plt.grid()
plt.show()