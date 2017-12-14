import numpy as np
from scipy import linalg as la
import torch
from torch.autograd import Variable
from sklearn.decomposition import nmf
import sys
from scipy.sparse.linalg import svds

from tqdm import tqdm

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

        x = x.double().cuda()
        for i in range(iter):

            f = torch.mm(x, torch.mm(g, torch.inverse(torch.mm(torch.t(g), g))))
            # f = f/torch.norm(f, 2)
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
            # g = g/torch.norm(g, 2)
        return f, g
    else:
        if p == 0:
            p = x.size()[1] - 2
        if p < 2:
            raise ValueError("The number of components (r) has to be >=2.")
        f = torch.randn(x.size()[0], p).double().cuda()
        g = torch.randn(x.size()[1], p).double().cuda()

        x = x.double().cuda()
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

def semi_nmf_numpy(x, iter=30):

    x = x.numpy()   # n * m
    f, g, p = svd_initialization_numpy(x)

    if p < 2:
        raise ValueError("The number of components (r) has to be >=2.")


    for i in tqdm(range(iter)):

        f = np.dot(x, np.dot(g, la.pinv(np.dot(g.T, g))))

        f = np.nan_to_num(f)

        Ap = (abs(np.dot(x.T, f)) + np.dot(x.T, f))/2   #m * r
        An = (abs(np.dot(x.T, f)) - np.dot(x.T, f))/2
        Bp = (abs(np.dot(g, np.dot(f.T, f))) + np.dot(g, np.dot(f.T, f)))/2
        Bn = (abs(np.dot(g, np.dot(f.T, f))) - np.dot(g, np.dot(f.T, f)))/2

        C = An + Bp
        for m in range(C.shape[0]):
            for n in range(C.shape[1]):
                if C[m, n] is 0:
                    C[m, n] += 0.0001

        for j in range(g.shape[0]):
            for k in range(g.shape[1]):
                g[j, k] = g[j, k] * np.sqrt( (Ap+Bn)[j,k]/(An+Bp)[j,k] )

    g = np.nan_to_num(g)

    return torch.from_numpy(f), torch.from_numpy(g)


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

    return torch.abs(U[:, 0:p]).double().cuda(), torch.t(torch.mm(sigma, torch.t(V))).double().cuda(), p

def svd_initialization_numpy(x):
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


    y = torch.rand(50,50)
    y = y/torch.norm(y, 2)
    f, g = semi_nmf(y,5, init='svd')
    re_y = torch.mm(f, torch.t(g))
    torch.set_printoptions(threshold=sys.maxsize)
    print('Y: ', '\n', y,'\n')
    print('re_y', '\n', re_y, '\n')
    print('f', '\n', f, '\n', 'g', '\n', g)

def test3():

    y = torch.randn(100, 100)
    f, g = semi_nmf_numpy(y,20)
    re_y = torch.mm(f, torch.t(g))
    # torch.set_printoptions(threshold=sys.maxsize)
    print('Y: ', '\n', y,'\n')
    print('re_y', '\n', re_y, '\n')
    # print('f', '\n', f, '\n', 'g', '\n', g)

def main():

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread('images/me.png')
    img0, img1, img2 = img[:,:,0], img[:,:,1], img[:,:,2]

    print(img0.shape)
    f0,g0 = appr_seminmf(img0, 100)
    f1,g1 = appr_seminmf(img1, 100)
    f2,g2 = appr_seminmf(img2, 100)
    re_img0 = np.dot(f0, g0)
    re_img1 = np.dot(f1, g1)
    re_img2 = np.dot(f2, g2)
    re_img = np.zeros(img.shape)
    re_img[:,:,0] = re_img0
    re_img[:,:,1] = re_img1
    re_img[:,:,2] = re_img2
    re_img[:,:,3] = img[:,:,3]



    # print("f:", '\n', torch.from_numpy(f))
    # print("g:", '\n', torch.from_numpy(g))
    print(torch.from_numpy(img))
    print(torch.from_numpy(re_img))
    plt.figure(1)
    plt.subplot(421)
    plt.imshow(img)
    plt.title('Original image')
    # plt.show()
    plt.subplot(422)
    plt.imshow(re_img)
    plt.title('Reconstructed image')
    # plt.show()

    # plt.figure(2)
    plt.subplot(423)
    plt.imshow(f0)
    plt.subplot(424)
    plt.imshow(g0)
    plt.subplot(425)
    plt.imshow(f1)
    plt.subplot(426)
    plt.imshow(g1)
    plt.subplot(427)
    plt.imshow(f2)
    plt.xlabel('features')
    plt.subplot(428)
    plt.imshow(g2)
    plt.xlabel('neural activity')
    plt.show()

    # torch.set_printoptions(threshold=sys.maxsize)

def main2():
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread('images/queen01.png')
    img0, img1, img2 = img[:,:,0], img[:,:,1], img[:,:,2]

    print(img0.shape)
    f0,g0 = appr_seminmf(img0, 100)
    f1,g1 = appr_seminmf(img1, 100)
    f2,g2 = appr_seminmf(img2, 100)

    f01,g01 = appr_seminmf(g0, min(g0.shape)-20)
    f11,g11 = appr_seminmf(g1, min(g0.shape)-20)
    f21,g21 = appr_seminmf(g2, min(g0.shape)-20)

    re_g0 = np.dot(f01, g01)
    re_g1 = np.dot(f11, g11)
    re_g2 = np.dot(f21, g21)

    re_img0 = np.dot(f0, re_g0)
    re_img1 = np.dot(f1, re_g1)
    re_img2 = np.dot(f2, re_g2)

    re_img = np.zeros(img.shape)
    re_img[:,:,0] = re_img0
    re_img[:,:,1] = re_img1
    re_img[:,:,2] = re_img2
    re_img[:,:,3] = img[:,:,3]

    print(torch.from_numpy(img))
    print(torch.from_numpy(re_img))

    plt.figure(1)

    plt.subplot(421)
    plt.imshow(img)
    plt.title('Original image')
    # plt.show()
    plt.subplot(422)
    plt.imshow(re_img)
    plt.title('Reconstructed image, 3 layer NMF')
    # plt.show()

    # plt.figure(2)
    plt.subplot(423)
    plt.imshow(f0)
    plt.subplot(424)
    plt.imshow(g0)
    plt.subplot(425)
    plt.imshow(f01)
    plt.subplot(426)
    plt.imshow(g01)
    plt.subplot(427)
    plt.imshow(f11)
    plt.xlabel('features')
    plt.subplot(428)
    plt.imshow(g11)
    plt.xlabel('neural activity')
    plt.show()

if __name__ == '__main__':
     main2()
    # test1()
    #test3()
    # test3()
