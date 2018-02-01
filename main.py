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



if __name__ == '__main__':


# Image Preprocessing
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    for i, data in tqdm(enumerate(trainloader, 0)):
        images, labels = data
        features = torch.FloatTensor(images.size()[0],images.size()[1],images.size()[2],20)
        coefficients = torch.FloatTensor(images.size()[0],images.size()[1],20,32)
        for j in range(3):

            features[0,j] = torch.from_numpy(appr_seminmf(images[0,j].numpy(), 20)[0])
            coefficients[0,j] = torch.from_numpy(appr_seminmf(images[0,j].numpy(), 20)[1])

        # torchvision.utils.save_image(features, 'images/cifar10/%s_%d.png' %(classes[labels[0]], i))
        torchvision.utils.save_image(coefficients, 'images/cifar10/coefficients/%s_%d.png' %(classes[labels[0]], i))
