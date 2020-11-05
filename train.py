import pytorch_lightning as pl
from pytorch_lightning import Trainer
from model import MetaDRN

class MetaDRNModule(pl.LightningModule):
    def __init__(self,algo='maml'):
        super(MetaDRNModule,self).__init__()
        self.net = MetaDRN()

    def forward(self, x):
        return self.net(x)

    # def training_step(self,batch,batch_idx):
    #     (spt_x, spt_y), (qry_x, qry_y) = 