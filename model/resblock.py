from torch import nn

from config import model_config as cfg
from .layers import conv2d


class Resblock(nn.Module):
    def __init__(self, id):
        super(Resblock, self).__init__()
        self.block_id = 'resblock%d' % id

        # print(**(cfg['resblocks'][self.block_id]['conv1']))
        self.add_module('conv1', nn.Conv2d(**(cfg['resblocks']
                                              [self.block_id]['conv1'])))
        self.add_module('conv2', nn.Conv2d(
            **(cfg['resblocks'][self.block_id]['conv2'])))
        self.add_module('reducer',
                        nn.Conv2d(**cfg(['reducer'][self.block_id])))

    def forward(self, x, weights=None):
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
            x = conv2d(x, weights[''.join(['resblocks', self.block_id,
                                           'conv1.weight'])],
                       weights[''.join(['resblocks', self.block_id,
                                        'conv1.bias'])], self.device)

            # conv2
            x = conv2d(x, weights[''.join(['resblocks', self.block_id,
                                           'conv2.weight'])],
                       weights[''.join(['resblocks', self.block_id,
                                        'conv2.bias'])], self.device)

            # reducer
            x = conv2d(x, weights[''.join(['resblocks', self.block_id,
                                           'reducer.weight'])],
                       weights[''.join(['resblocks', self.block_id,
                                        'reducer.bias'])], self.device)

            # add
            x = x_init + x
        return x
