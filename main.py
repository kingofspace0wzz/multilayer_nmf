import numpy as np
from scipy import linalg as la
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import visdom
from nmf import *

def experiment_1():
  img = mpimg.imread('images/stinkbug.png')
  img0, img1, img2 = img[:,:,0], img[:,:,1], img[:,:,2]
  f0,g0 = appr_seminmf(img0, 100)
  f1,g1 = appr_seminmf(img1, 100)
  f2,g2 = appr_seminmf(img2, 100)
  re_img0 = np.dot(f0, g0)
  re_img1 = np.dot(f1, g1)
  re_img2 = np.dot(f2, g2)

  corruptedF = f0
  print(torch.from_numpy(f0))
  # corruptedF[:,3], corruptedF[:,19] = f0[:,19], f0[:,3]
  corruptedF[:, 19] = 0
  corruptedImg = np.dot(corruptedF, g0)
  part1 = np.outer(f0[:, 19], g0[19, :])
  part2 = np.outer(f0[:,-1], g0[-1,:])

  plt.figure(1)
  plt.subplot(231)
  plt.imshow(re_img0)
  plt.subplot(232)
  plt.imshow(f0)
  plt.subplot(233)
  plt.imshow(part2)
  plt.subplot(234)
  plt.imshow(part1)
  plt.subplot(235)
  plt.imshow(corruptedImg)
  # plt.imshow(corruptedImg)
  plt.show()

  for i in range(f0.shape[1]):
    if f0[0,i] >10:
        print(i)


def experiment_2():

    x = np.random.rand(100,100)
    f,g = appr_seminmf(x, 80)
    re_x = np.dot(f,g)
    plt.figure(1)
    plt.subplot(221)
    plt.title('Original image')
    plt.imshow(x)
    plt.subplot(222)
    plt.title('Reconstructed image')
    plt.imshow(re_x)
    plt.subplot(223)
    plt.xlabel('features')
    plt.imshow(f)
    plt.subplot(224)
    plt.xlabel('neural activity')
    plt.imshow(g)
    plt.show()

# experiment_1()
experiment_2()
