import numpy as np
from scipy import linalg as la
import torch
from torch.autograd import Variable
from sklearn.decomposition import nmf
import sys

def semi_nmf(x, iter=30, init='svd', p=0):
    '''
    Semi Nonnegative Matrix Factorization.
    It returns a feature matrix F and a representation matrix G by minimizing
    frobenius norm ||X - FG^T||^2. The only contraint is that elements in G to be positive.

    Args:
        x: input matrix X
        int iter: number of iterations of optimization algorithm

    Return:
        f: feature matrix F
        g: representation matrix G
    '''

    # x = x.numpy()   # n * m
    # f, g, p = svd_initialization(x)
    #
    # if  < 2:
    #     raise ValueError("The number of components (r) has to be >=2.")
    #
    #
    # for i in range(iter):
    #
    #     f = np.dot(x, np.dot(g, la.pinv(np.dot(g.T, g))))
    #
    #     f = np.nan_to_num(f)
    #
    #     Ap = (abs(np.dot(x.T, f)) + np.dot(x.T, f))/2   #m * r
    #     An = (abs(np.dot(x.T, f)) - np.dot(x.T, f))/2
    #     Bp = (abs(np.dot(g, np.dot(f.T, f))) + np.dot(g, np.dot(f.T, f)))/2
    #     Bn = (abs(np.dot(g, np.dot(f.T, f))) - np.dot(g, np.dot(f.T, f)))/2
    #
    #     C = An + Bp
    #     for m in range(C.shape[0]):
    #         for n in range(C.shape[1]):
    #             if C[m, n] is 0:
    #                 C[m, n] += 0.0001
    #
    #     for j in range(g.shape[0]):
    #         for k in range(g.shape[1]):
    #             g[j, k] = g[j, k] * np.sqrt( (Ap+Bn)[j,k]/(An+Bp)[j,k] )
    #
    # g = np.nan_to_num(g)
    #
    # return torch.from_numpy(f), torch.from_numpy(g)
    if init is 'svd':
        f, g, p = svd_initialization(x)

        if p < 2:
            raise ValueError("The number of components (r) has to be >=2.")

        x = x.cuda()
        for i in range(iter):

            f = torch.mm(x, torch.mm(g, torch.inverse(torch.mm(torch.t(g), g))))

            Ap = (torch.abs(torch.mm(torch.t(x), f)) + torch.mm(torch.t(x), f))/2   #m * r
            An = (torch.abs(torch.mm(torch.t(x), f)) - torch.mm(torch.t(x), f))/2
            Bp = (torch.abs(torch.mm(g, torch.mm(torch.t(f), f))) + torch.mm(g, torch.mm(torch.t(f), f)))/2
            Bn = (torch.abs(torch.mm(g, torch.mm(torch.t(f), f))) - torch.mm(g, torch.mm(torch.t(f), f)))/2

            C = An + Bp

            for m in range(C.size()[0]):
                for n in range(C.size()[1]):
                    if C[m, n] is 0:
                        C[m, n] += 0.0001

            for j in range(g.size()[0]):
                for k in range(g.size()[1]):
                    if ((An+Bp)[j,k] == 0):
                        g[j,k] = 0.5

                    else:
                        # print((An+Bp)[j,k], 'i=', i, 'j=', j, 'k=', k)
                        g[j, k] = g[j, k] * np.sqrt( (Ap+Bn)[j,k]/(An+Bp)[j,k] )

        return f, g
    else:
        if p == 0:
            p = x.size()[1] - 2
        if p < 2:
            raise ValueError("The number of components (r) has to be >=2.")
        f = torch.randn(x.size()[0], p).cuda()
        g = torch.randn(x.size()[1], p).cuda()

        x = x.cuda()
        for i in range(iter):

            f = torch.mm(x, torch.mm(g, torch.inverse(torch.mm(torch.t(g), g))))

            Ap = (torch.abs(torch.mm(torch.t(x), f)) + torch.mm(torch.t(x), f))/2   #m * r
            An = (torch.abs(torch.mm(torch.t(x), f)) - torch.mm(torch.t(x), f))/2
            Bp = (torch.abs(torch.mm(g, torch.mm(torch.t(f), f))) + torch.mm(g, torch.mm(torch.t(f), f)))/2
            Bn = (torch.abs(torch.mm(g, torch.mm(torch.t(f), f))) - torch.mm(g, torch.mm(torch.t(f), f)))/2

            C = An + Bp

            for m in range(C.size()[0]):
                for n in range(C.size()[1]):
                    if C[m, n] is 0:
                        C[m, n] += 0.0001

            for j in range(g.size()[0]):
                for k in range(g.size()[1]):
                    if ((An+Bp)[j,k] == 0):
                        g[j,k] = 0
                        print('ZEROS!!!')
                    else:
                        print((An+Bp)[j,k], 'i=', i, 'j=', j, 'k=', k)
                        g[j, k] = g[j, k] * np.sqrt( (Ap+Bn)[j,k]/(An+Bp)[j,k] )

        return f, g

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
    # p, sum_p = 0, 0
    # U, s, Vh = la.svd(x)
    # sum_r = sum(s)
    #
    # for i in range(len(s)):
    #
    #     if sum_p/sum_r < 0.9:
    #         sum_p = sum_p + s[i]
    #         p+=1
    #
    # sigma = np.zeros((p, x.shape[1]))
    #
    # for i in range(p):
    #     sigma[i,i] = s[i]
    #
    # return abs(U[:, 0:p]), np.dot(sigma, Vh).T, p
    p, sum_p = 0, 0
    U, s, V = torch.svd(x)
    sum_r = sum(s)

    for i in range(len(s)):

        if sum_p/sum_r < 0.9:
            sum_p = sum_p + s[i]
            p+=1

    sigma = torch.zeros(p, x.size()[1])

    for i in range(p):
        sigma[i,i] = s[i]

    return torch.abs(U[:, 0:p]).cuda(), torch.t(torch.mm(sigma, torch.t(V))).cuda(), p

def appr_seminmf(M, r):


    if r < 2:
        raise ValueError("The number of components (r) has to be >=2.")

    A, S, B = svds(M, r-1)
    S = np.diag(S)
    A = np.dot(A, S)

    m, n = M.shape

    for i in range(r-1):
        if B[i, :].min() < (-B[i, :]).min():
            B[i, :] = -B[i, :]
            A[:, i] = -A[:, i]


    if r == 2:
        U = np.concatenate([A, -A], axis=1)
    else:
        An = -np.sum(A, 1).reshape(A.shape[0], 1)
        U = np.concatenate([A, An], 1)

    V = np.concatenate([B, np.zeros((1, n))], 0)

    if r>=3:
        V -= np.minimum(0, B.min(0))
    else:
        V -= np.minimum(0, B)

    return U, V


# solve NMF by hierarchical alternating least squares, returns the approximation matrices and the residue
# alpha keeps W from becoming too big, beta keeps H sparse
def HALS(A, k, epsilon = 0.01, alpha = 0.0, beta = 0.01):
    '''
    solve NMF by hierarchical alternating least squares, returns the approximation matrices and the residue
    alpha keeps W from becoming too big, beta keeps H sparse
    '''
    W = np.random.random_sample((A.shape[0], k))
    H = np.random.random_sample((A.shape[1], k))

    proGradientW = 2 * W.dot(H.T).dot(H) - 2 * A.dot(H)
    proGradientH = 2 * H.dot(W.T).dot(W) - 2 * A.T.dot(W)

    for i in range(proGradientW.shape[0]):
        for j in range(proGradientW.shape[1]):

            if proGradientW[i, j] >= 0 or W[i, j] <= 0:
                proGradientW[i, j] = 0

    for i in range(proGradientH.shape[0]):
        for j in range(proGradientH.shape[1]):

            if proGradientH[i, j] >= 0 or H[i, j] <= 0:
                proGradientH[i, j] = 0

    initalDelta = np.sqrt( np.square(la.norm(proGradientW, 'fro')) + np.square(la.norm(proGradientH, 'fro')) )

    while True:

        for i in range(k):
            temp = W[:, i].reshape((A.shape[0], 1))
            temp = W[:, i].reshape((A.shape[0], 1)) * H.T.dot(H)[i, i] / (H.T.dot(H)[i, i] + alpha) + ( (A.dot(H))[:, i].reshape((A.shape[0], 1)) - (W.dot(H.T).dot(H))[:, i].reshape((A.shape[0], 1)) ) / (H.T.dot(H)[i, i] + alpha)
            W[:, i] = temp.reshape((1, A.shape[0]))

            for j in range(len(W[:, i])):
                if W[:, i][j] < 0:
                    W[:, i][j] = 0

        for i in range(k):
            temp = H[:, i].reshape((A.shape[1], 1))
            temp = H[:, i].reshape((A.shape[1], 1)) + ( (A.T.dot(W))[:, i].reshape((A.shape[1], 1)) - (H.dot(W.T.dot(W) + beta))[:, i].reshape((A.shape[1], 1)) ) / (W.T.dot(W)[i, i] + beta)
            H[:, i] = temp.reshape((1, A.shape[1]))

            for j in range(len(H[:, i])):
                if H[:, i][j] < 0:
                    H[:, i][j] = 0

        proGradientW = 2 * W.dot(H.T).dot(H) - 2 * A.dot(H)
        proGradientH = 2 * H.dot(W.T).dot(W) - 2 * A.T.dot(W)

        for i in range(proGradientW.shape[0]):
            for j in range(proGradientW.shape[1]):

                if proGradientW[i, j] >= 0 or W[i, j] <= 0:
                    proGradientW[i, j] = 0

        for i in range(proGradientH.shape[0]):
            for j in range(proGradientH.shape[1]):

                if proGradientH[i, j] >= 0 or H[i, j] <= 0:
                    proGradientH[i, j] = 0

        delta = np.sqrt( np.square(la.norm(proGradientW, 'fro')) + np.square(la.norm(proGradientH, 'fro')) )

        if delta / initalDelta <= epsilon:
            break

    residue = la.norm(A - W.dot(H.T), 2)

    return W, H, residue



def test_HALS(A, k, alpha = 0.0, beta = 0.0):

    W = np.random.random_sample((A.shape[0], k))
    H = np.random.random_sample((A.shape[1], k))


    for e in range(100):

        for i in range(k):
            temp = W[:, i].reshape((A.shape[0], 1))
            temp = W[:, i].reshape((A.shape[0], 1)) * H.T.dot(H)[i, i] / (H.T.dot(H)[i, i] + alpha) + ( (A.dot(H))[:, i].reshape((A.shape[0], 1)) - (W.dot(H.T).dot(H))[:, i].reshape((A.shape[0], 1)) ) / (H.T.dot(H)[i, i] + alpha)
            W[:, i] = temp.reshape((1, A.shape[0]))

            for j in range(len(W[:, i])):
                if W[:, i][j] < 0:
                    W[:, i][j] = 0

        for i in range(k):
            temp = H[:, i].reshape((A.shape[1], 1))
            temp = H[:, i].reshape((A.shape[1], 1)) + ( (A.T.dot(W))[:, i].reshape((A.shape[1], 1)) - (H.dot(W.T.dot(W) + beta))[:, i].reshape((A.shape[1], 1)) ) / (W.T.dot(W)[i, i] + beta)
            H[:, i] = temp.reshape((1, A.shape[1]))

            for j in range(len(H[:, i])):
                if H[:, i][j] < 0:
                    H[:, i][j] = 0



    residue = la.norm(A - W.dot(H.T), 2)

    print(W, '\n', H, '\n', residue, '\n')
    print(W.dot(H.T))

#------------------test------------------

def test1():
    x = torch.randn(7,9)
    # semi_nmf(x)
    m = x.numpy()
    w1,h1 = appr_seminmf(m, 6)
    w2,h2 = appr_seminmf(h1, 5)
    w3,h3 = appr_seminmf(h2, 4)

    re_x1 = np.dot(w1, h1)
    re_x2 = np.dot(w1, np.dot(w2, h2))
    re_x3 = np.dot(w1, np.dot(w2, np.dot(w3, h3)))
    # print(x)
    # print(torch.from_numpy(re_x))
    print('h1:', '\n', torch.from_numpy(h1))
    print('h2:', '\n', torch.from_numpy(h2))
    print('h3:', '\n', torch.from_numpy(h3))
    print(x)
    print('re_x1: ', '\n', torch.from_numpy(re_x1))
    print('re_x2: ', '\n', torch.from_numpy(re_x2))
    print('re_x3: ', '\n', torch.from_numpy(re_x3))

def test2():
    # x = torch.randn(4,4).numpy()
    # f, g, p = svd_initialization(x)
    #
    # Ap = (abs(np.dot(x.T, f)) + np.dot(x.T, f))/2   #m * r
    # An = (abs(np.dot(x.T, f)) - np.dot(x.T, f))/2
    # Bp = (abs(np.dot(g, np.dot(f.T, f))) + np.dot(g, np.dot(f.T, f)))/2
    # Bn = (abs(np.dot(g, np.dot(f.T, f))) - np.dot(g, np.dot(f.T, f)))/2
    #
    # print('An','\n', An, '\n', 'Bp' '\n', Bp)
    # print('dot(x.T, f)', '\n', np.dot(x.T, f), '\n', 'dot(g, dot(f.T, f))' '\n', np.dot(g, np.dot(f.T, f)))
    # for j in range(g.shape[0]):
    #     for k in range(g.shape[1]):
    #         g[j, k] = g[j, k] * np.sqrt( (Ap+Bn)[j,k]/C[j,k] )


    # y = torch.randn(10,10)
    # f, g = semi_nmf(y,20)
    # re_y = torch.from_numpy(np.dot(f.numpy(), g.numpy().T))
    # torch.set_printoptions(threshold=sys.maxsize)
    # print('Y: ', '\n', y,'\n')
    # print('re_y', '\n', re_y, '\n')
    # print('f', '\n', f, '\n', 'g', '\n', g)

    y = torch.randn(20,20)
    f, g = semi_nmf(y,40, init='r')
    re_y = torch.mm(f, torch.t(g))
    torch.set_printoptions(threshold=sys.maxsize)
    print('Y: ', '\n', y,'\n')
    print('re_y', '\n', re_y, '\n')
    print('f', '\n', f, '\n', 'g', '\n', g)

def main():

    x = torch.randn(10, 10)
    f, g = semi_nmf(x)
    print(f, '\n')
    print(g, '\n')

if __name__ == '__main__':
    # main()
    # test1()
    test2()
