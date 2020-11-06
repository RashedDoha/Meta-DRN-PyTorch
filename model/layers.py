import torch.nn.functional as F
import numpy as np


def conv2d(input, weight, bias=None, device='cpu', stride=1, padding=0,
           dilation=1, groups=1):
    return F.conv2d(input, weight.to(device), bias.to(device), stride, padding,
                    dilation, groups)


def batchnorm(input, weight=None, bias=None, device='cpu', running_mean=None,
              running_var=None, training=True, eps=1e-5, momentum=0.1):
    ''' momentum = 1 restricts stats to the current mini-batch '''
    # This hack only works when momentum is 1 and avoids needing to track
    # running stats by substuting dummy variables
    running_mean = torch.zeros(
        np.prod(np.array(input.data.size()[1]))).to(device)
    running_var = torch.ones(
        np.prod(np.array(input.data.size()[1]))).to(device)
    return F.batch_norm(input, running_mean, running_var, weight, bias,
                        training, momentum, eps)


def leaky_relu(input, negative_slope=0.01):
    return F.leaky_relu(input, negative_slope, True)


def pixel_shuffle(input, scale):
    return F.pixel_shuffle(input, scale)
