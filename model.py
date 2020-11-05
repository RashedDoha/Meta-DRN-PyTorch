import torch
from torch import nn
from resblock import Resblock
from config import model_config as cfg
from torchmeta.modules import MetaModule, MetaSequential,MetaConv2d, MetaBatchNorm2d

class MetaDRN(MetaModule):
    def __init__(self):
        super(MetaDRN,self).__init__()
        self.head = nn.Sequential()
        self.head.add_module('conv1', MetaConv2d(**cfg['head']['conv1']))
        self.head.add_module('bn1', MetaBatchNorm2d(**cfg['head']['bn1']))
        self.head.add_module('lr1', nn.LeakyReLU())
        self.head.add_module('conv2', MetaConv2d(**cfg['head']['conv2']))
        self.head.add_module('bn2', MetaBatchNorm2d(**cfg['head']['bn2']))
        self.head.add_module('lr2', nn.LeakyReLU())

        self.resblocks = nn.Sequential()
        self.resblocks.add_module('resblock1', Resblock(1))
        self.resblocks.add_module('resblock2', Resblock(2))
        self.resblocks.add_module('resblock3', Resblock(3))

        self.degrid = nn.Sequential()
        self.degrid.add_module('conv1', MetaConv2d(**cfg['degrid']['conv1']))
        self.degrid.add_module('conv2', MetaConv2d(**cfg['degrid']['conv2']))

        self.upsample = MetaSequential(
            MetaConv2d(**cfg['upsample']['conv']),
            nn.PixelShuffle(**cfg['upsample']['pixel_shuffle'])
        )

    def forward(self,x, params=None):
        return self.upsample(self.degrid(self.resblocks(self.head(x))))

def main():
    net = MetaDRN()
    # print(net._named_members)
    # for param in net.meta_named_parameters():
    #     print(f'Shape of parameters of {param[0]}: {param[1].shape}')
    img = torch.randn(1,3,128,256)
    print(net(img).shape)

if __name__ == '__main__':
    main()