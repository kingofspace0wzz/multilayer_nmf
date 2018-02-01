import numpy as np
from scipy import linalg as la
import torch
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import visdom
from nmf import *



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

def main3():
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread('images/queen01.png')
    img0, img1, img2, img3 = img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3]

    plt.imshow(img0+img1+img2+img3)
    plt.show()

if __name__ == '__main__':
    main3()
