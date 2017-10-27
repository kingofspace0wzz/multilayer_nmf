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
        int iter: number of iterations of optimization algorithm

    Return:
        f: feature matrix F
        g: representation matrix G
    '''
    x = x.numpy()
    f, g, p = svd_initialization(x)

    for i in range(iter):

        f = np.dot(x, np.dot(g, la.pinv(np.dot(g.T, g))))
        f = np.nan_to_num(f)
        for j in range(g.shape[0]):

            for k in range(g.shape[1]):

                g[j,k] = g[j,k] * np.sqrt( ( ( abs(np.dot(x.T, f))[j,k] + np.dot(x.T, f)[j,k] )/2
                    +  np.dot(g, (abs(np.dot(f.T, f)) - np.dot(f.T, f))/2)[j,k] )
                    /  ( ( abs(np.dot(x.T, f))[j,k] - np.dot(x.T, f)[j,k] )/2
                    +  np.dot(g, (abs(np.dot(f.T, f)) + np.dot(f.T, f))/2)[j,k] )  ) # BUG to be fixed

        g = np.nan_to_num(g)


    return torch.from_numpy(f), torch.from_numpy(g)

def convex_nmf(x, iter=1000):
    '''
    Convex Nonnegative Matrix Factorization. (enhance sparsity)
    It returns a feature matrix F and a representation matrix G by minimizing
    frobenius norm ||X - XWG^T||^2. The only contraint is that elements in G and W have to be positive.

    Args:
        x: input matrix X
        int iter: number of iterations of optimization algorithm

    Return:
        w: feature matrix F
        g: representation matrix G
    '''


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

    return abs(U[:, 0:p]), np.dot(sigma, Vh).T, p


#------------------test------------------

def main():

    x = torch.randn(10, 10)
    f, g = semi_nmf(x)
    print(f, '\n')
    print(g, '\n')

if __name__ == '__main__':
    main()
