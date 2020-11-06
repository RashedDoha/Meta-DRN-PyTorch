from collections import OrderedDict

import torch
from config.global_config import cfg as gcfg
from config.model_config import cfg
from torch import nn

from .layers import batchnorm, conv2d, leaky_relu
from .resblock import Resblock


class MetaDRN(nn.Module):
    def __init__(self, loss_fn, seed=gcfg['seed']):
        super(MetaDRN, self).__init__()
        print(**(cfg['head']['conv1']))
        # Definet the network
        self.head = nn.Sequential()
        self.head.add_module('conv1', nn.Conv2d(**(cfg['head']['conv1'])))
        self.head.add_module('bn1', nn.BatchNorm2d(**cfg['head']['bn1']))
        self.head.add_module('lr1', nn.LeakyReLU())
        self.head.add_module('conv2', nn.Conv2d(**(cfg['head']['conv2'])))
        self.head.add_module('bn2', nn.BatchNorm2d(**(cfg['head']['bn2'])))
        self.head.add_module('lr2', nn.LeakyReLU())

        self.resblocks = nn.Sequential()
        self.resblocks.add_module('resblock1', Resblock(1))
        self.resblocks.add_module('resblock2', Resblock(2))
        self.resblocks.add_module('resblock3', Resblock(3))

        self.degrid = nn.Sequential()
        self.degrid.add_module('conv1', nn.Conv2d(**cfg['degrid']['conv1']))
        self.degrid.add_module('conv2', nn.Conv2d(**cfg['degrid']['conv2']))

        self.upsample = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(**cfg['upsample']['conv'])),
            ('pixel_shuffle', nn.PixelShuffle(
                **cfg['upsample']['pixel_shuffle']))
        ]))

        # define loss fn
        self.loss_fn = loss_fn

        # init weights
        self._init_weights(seed)

    def forward(self, x, weights=None):
        if weights is None:
            x = self.head(x)
            x = self.resblocks(x)
            x = self.degrid(x)
            x = self.upsample(x)
        else:
            # head

            x = conv2d(x, weights['head.conv1.weight'],
                       weights['head.conv1.bias'], self.device)
            x = batchnorm(x, weights['head.bn1.weight'],
                          weights['head.bn1.bias'], self.device)
            x = leaky_relu(x)
            x = conv2d(x, weights['head.conv2.weight'],
                       weights['head.conv2.bias'], self.device)
            x = batchnorm(x, weights['head.bn2.weight'],
                          weights['head.bn2.bias'], self.device)
            x = leaky_relu(x)

            # resblocks
            x = self.resblocks(x, weights=weights)

            # upsample

            x = conv2d(x, weights['degrid.conv1.weight'],
                       weights['degrid.conv1.bias'], self.device)
            x = conv2d(x, weights['degrid.conv2.weight'],
                       weights['degrid.conv2.bias'], self.device)

            x
        return x

    def _init_weights(self, seed):
        "Set weights to Gaussian, biases to zero"
        torch.manual_seed(seed)
        print('init weights')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.01),
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def main():
    return


if __name__ == '__main__':
    main()
