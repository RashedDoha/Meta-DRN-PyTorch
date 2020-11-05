from torch import nn
from config import model_config as cfg
from torchmeta.modules import MetaModule,MetaSequential,MetaConv2d

class Resblock(MetaModule):
    def __init__(self, id):
        super(Resblock, self).__init__()
        block_id = 'resblock%d'%id

        self.resblocks = MetaSequential(
            MetaConv2d(**cfg['resblocks'][block_id]['conv1']),
            MetaConv2d(**cfg['resblocks'][block_id]['conv2']),
            MetaConv2d(**cfg['reducer'][block_id])
        )

    def forward(self, x):
        x_init = x
        for i,block in enumerate(self.resblocks.children()):
            if i == 2:
                break
            x = block(x)
        return block(x_init)+x