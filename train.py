import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import cfg
from dataset import get_dataloader
from model import MetaDRN

from pytorch_lightning.metrics.functional.classification import iou

n_epochs = cfg['train_config']['n_epochs']
learner_lr = cfg['train_config']['maml']['learner_lr']
meta_lr = cfg['train_config']['maml']['meta_lr']

net = MetaDRN(F.cross_entropy)

learner_optim = optim.AdamW(net.parameters(), lr=learner_lr)
meta_optim = optim.AdamW(net.parameters(), lr=meta_lr)

meta_scheduler = ReduceLROnPlateau(meta_optim, 'max')

train_dl = get_dataloader('maml')

for e in range(1, n_epochs + 1):
  for batch in tqdm(train_dl):
    pass
