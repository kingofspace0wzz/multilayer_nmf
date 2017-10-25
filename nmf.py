import numpy as np
from scipy import linalg as la
import torch
from torch.autograd import Variable
from sklearn.decomposition import nmf

def semi_nmf(x, iter=1000):
    '''
    Semi Nonnegative Matrix Factorization.
    It returns a feature matrix F and a representation matrix G by minimizing
    frobenius norm ||X - FG^T||^2. The only contraint is that elements in G have to be positive.

    Args:
        x: input matrix X
        int r: dimension of the latent structure embedding in X
        int iter: number of iterations of optimization algorithm

    Return:
        f: feature matrix F
        g: representation matrix G
    '''
    x = x.numpy()
    f, gT, p = svd_initialization(x)
    for i in range(iter):
        f = np.dot(x, np.dot(gT.T, la.inv(np.dot(gT, gT.T))))
        for j in range(gT.shape[0]):
            for k in range(gT.shape[1]):
                gT[j,k] = gT[j,k] * np.sqrt( ( ( abs(np.dot(x.T, f))[j,k] + np.dot(x.T, f)[j,k] )/2  +  np.dot(gT.T, (abs(np.dot(f.T, f)) - np.dot(f.T, f))/2)[j,k] )
                    /  ( ( abs(np.dot(x.T, f))[j,k] - np.dot(x.T, f)[j,k] )/2  +  np.dot(gT.T, (abs(np.dot(f.T, f)) + np.dot(f.T, f))/2)[j,k] )  ) # BUG to be fixed

    return torch.from_numpy(f), torch.from_numpy(gT.T)

def svd_initialization(x):
    '''
    SVD based initialization for feature matrix F and representation matrix G

    Args:
        x: input matrix X

    Returns:
        F: initialized feature matrix
        G: initialized representation matrix
        p: rank of Factorization
    '''
    p, sum_p = 0, 0
    U, s, Vh = la.svd(x)
    sum_r = sum(s)
    for i in range(len(s)):
        if sum_p/sum_r < 0.9:
            sum_p = sum_p + s[i]
            p+=1

    sigma = np.zeros((p, x.shape[1]))
    for i in range(p):
        sigma[i,i] = s[i]

    return abs(U[:, 0:p]), np.dot(sigma, Vh), p


------------------test------------------

def main():



if __name__ == '__main__':
    main()
