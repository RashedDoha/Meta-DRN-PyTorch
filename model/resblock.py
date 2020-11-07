""" Residual components of the network"""
from config.model_config import cfg
from torch import nn
from torch.nn.parameter import Parameter
from typings.common_types import Dict, Tensor

from .layers import conv2d


class Resblock(nn.Module):
  """[summary]

  Args:
      nn ([type]): [description]
  """

  def __init__(self, block_id: int):
    super().__init__()
    self.block_id = 'resblock%d' % block_id

    # print(**(cfg['resblocks'][self.block_id]['conv1']))
    self.add_module('conv1',
                    nn.Conv2d(**cfg['resblocks'][self.block_id]['conv1']))
    self.add_module('conv2',
                    nn.Conv2d(**(cfg['resblocks'][self.block_id]['conv2'])))
    self.add_module('reducer', nn.Conv2d(**cfg['reducer'][self.block_id]))

  def forward(self, x: Tensor, weights: Dict[str, Parameter] = None):
    if weights is None:
      x_init = x
      for i, block in enumerate(self.children()):
        if i == 2:
          break
        x = block(x)
      x = x_init + x
    else:
      x_init = x

      # conv1
      x = conv2d(x,
                 weights[''.join(['resblocks', self.block_id, 'conv1.weight'])],
                 weights[''.join(['resblocks', self.block_id, 'conv1.bias'])],
                 str(self.device))

      # conv2
      x = conv2d(x,
                 weights[''.join(['resblocks', self.block_id, 'conv2.weight'])],
                 weights[''.join(['resblocks', self.block_id, 'conv2.bias'])],
                 str(self.device))

      # reducer
      x = conv2d(
          x, weights[''.join(['resblocks', self.block_id, 'reducer.weight'])],
          weights[''.join(['resblocks', self.block_id, 'reducer.bias'])],
          str(self.device))

      # add
      x = x_init + x
    return x
