from collections import OrderedDict

import torch
from config.global_config import cfg as gcfg
from config.model_config import cfg as mcfg
from torch import nn
from torch._C import ParameterDict
from typings.common_types import Callable, Optional, Tensor, _int

from .layers import batchnorm, conv2d, leaky_relu, pixel_shuffle
from .resblock import Resblock


class MetaDRN(nn.Module):

  def __init__(self,
               loss_fn: Callable[[Tensor, Tensor], Tensor],
               seed: _int = gcfg["seed"]):
    super(MetaDRN, self).__init__()
    # Definet the network
    self.head = nn.Sequential()
    self.head.add_module("conv1", nn.Conv2d(**mcfg["head"]["conv1"]))
    self.head.add_module("bn1", nn.BatchNorm2d(**mcfg["head"]["bn1"]))
    self.head.add_module("lr1", nn.LeakyReLU())
    self.head.add_module("conv2", nn.Conv2d(**mcfg["head"]["conv2"]))
    self.head.add_module("bn2", nn.BatchNorm2d(**mcfg["head"]["bn2"]))
    self.head.add_module("lr2", nn.LeakyReLU())

    self.resblocks = nn.Sequential()
    self.resblocks.add_module("resblock1", Resblock(1))
    self.resblocks.add_module("resblock2", Resblock(2))
    self.resblocks.add_module("resblock3", Resblock(3))

    self.degrid = nn.Sequential()
    self.degrid.add_module("conv1", nn.Conv2d(**mcfg["degrid"]["conv1"]))
    self.degrid.add_module("conv2", nn.Conv2d(**mcfg["degrid"]["conv2"]))

    self.upsample = nn.Sequential(
        OrderedDict([("conv1", nn.Conv2d(**mcfg["upsample"]["conv"])),
                     ("pixel_shuffle",
                      nn.PixelShuffle(**mcfg["upsample"]["pixel_shuffle"]))]))

    # define loss fn
    self.loss_fn = loss_fn

    # init weights
    self._init_weights(seed)

  def forward(self, x: Tensor, weights=Optional[ParameterDict]) -> Tensor:
    if weights is None:
      x = self.head(x)
      x = self.resblocks(x)
      x = self.degrid(x)
      x = self.upsample(x)
    else:
      # head

      x = conv2d(x, weights["head.conv1.weight"], weights["head.conv1.bias"],
                 str(self.device))
      x = batchnorm(x, weights["head.bn1.weight"], weights["head.bn1.bias"],
                    str(self.device))
      x = leaky_relu(x)
      x = conv2d(x, weights["head.conv2.weight"], weights["head.conv2.bias"],
                 str(self.device))
      x = batchnorm(x, weights["head.bn2.weight"], weights["head.bn2.bias"],
                    str(self.device))
      x = leaky_relu(x)

      # resblocks
      x = self.resblocks(x, weights=weights)

      # upsample

      x = conv2d(x, weights["degrid.conv1.weight"],
                 weights["degrid.conv1.bias"], str(self.device))
      x = conv2d(x, weights["degrid.conv2.weight"],
                 weights["degrid.conv2.bias"], str(self.device))

      x = conv2d(x, weights["upsample.conv1.weight"],
                 weights["upsample.conv1.bias"], str(self.device))
      x = pixel_shuffle(x, mcfg["upscale_factor"])

    return x

  def _init_weights(self, seed: _int) -> None:
    "Set weights to Gaussian, biases to zero"
    torch.manual_seed(seed)
    print("init weights")
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, a=0.01),
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def copy_weights(self, net: "MetaDRN") -> None:
    """ Set this module"s weights to be the same as those of "net" """
    if type(self) == type(net):
      self.load_state_dict(net.state_dict())


def main() -> None:
  return


if __name__ == "__main__":
  main()
