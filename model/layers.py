import numpy as np
import torch
import torch.nn.functional as F
from common_types import Tensor, Union, _float, _int
from torch import Tensor
from torch.types import Device, _size

_opt_arg = Union[_int, _size]
_opt_tensor = Union[Tensor, None]


def conv2d(x: Tensor,
           weight: Tensor,
           bias: _opt_tensor = None,
           device: Device = 'cpu',
           stride: _opt_arg = 1,
           padding: _opt_arg = 0,
           dilation: _opt_arg = 1,
           groups: _int = 1) -> Tensor:

  return F.conv2d(x, weight.to(device), bias.to(device), stride, padding,
                  dilation, groups)


def batchnorm(x: Tensor,
              weight: Tensor = None,
              bias: _opt_tensor = None,
              device: Device = 'cpu',
              running_mean: _opt_tensor = None,
              running_var: _opt_tensor = None,
              training: bool = True,
              eps: _float = 1e-5,
              momentum: _float = 0.1) -> Tensor:
  ''' momentum = 1 restricts stats to the current mini-batch '''
  # This hack only works when momentum is 1 and avoids needing to track
  # running stats by substuting dummy variables
  running_mean = torch.zeros(np.prod(np.array(x.data.size()[1]))).to(device)
  running_var = torch.ones(np.prod(np.array(x.data.size()[1]))).to(device)
  return F.batch_norm(x, running_mean, running_var, weight, bias, training,
                      momentum, eps)


def leaky_relu(x: Tensor, negative_slope: _float = 0.01) -> Tensor:
  return F.leaky_relu(x, negative_slope, True)


def pixel_shuffle(x: Tensor, scale: _int):
  return F.pixel_shuffle(x, scale)
