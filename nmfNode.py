import numpy as np
import pymf
import torch
from torch.autograd import Variable

class nmfNode(torch.autograd.Function):
    """
    NMF node that takes Semi-NMF as its pseudo-activation
    """
    def forward(self, input, int rank, int iter):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the representation H obtained by Semi-nonnegative-matrix-decomposition
        operation.

        Args:
            input: data
            rank: rank of the representation matrix H, or the dimension of the latent structure embedding 
                in the original high-dimensional space
            iter: number of iterations of Semi-NMF optimization algorithm
        """
        data = input.numpy()
        nmf_mdl = pymf.SNMF(data, num_bases=rank, niter=iter)   # Semi-NMF
        nmf_mdl.initialization()
        # Could find some matrix to normalize W
        nmf_mdl.W = W
        nmf_mdl.H = H
        nmf_mdl.factorize()

        return torch.from_numpy(H)

    def backword():
        """
        NMF layer does not need to store gradients
        """
        return None
