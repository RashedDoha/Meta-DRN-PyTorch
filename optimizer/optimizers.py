from argparse import Namespace

from config.train_config import cfg
from torch import optim
from torch.optim import Optimizer
from typings.common_types import Dict, Module, Union


def get_optimizers(
    module: Module, args: Namespace
) -> Union[Dict[str, Union[Optimizer, object, str]], [Optimizer, object]]:
  meta_lr = cfg[args.algo]['meta_lr']
  n_epochs = cfg['n_epochs']
  optimizer = optim.AdamW(module.parameters(), meta_lr)
  if args.algo in ['maml', 'fomal', 'meta-sgd']:
    patience = cfg[args.algo]['halve_lr_every']
    lr_red = cfg[args.algo]['lr_reduction_factor']
    metric_to_watch = cfg[args.algo]['metric_to_watch']
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', lr_red,
                                                     patience)
    return {
        'optimizer': optimizer,
        'scheduler': scheduler,
        'monitor': metric_to_watch
    }

  else:
    final_meta_lr = cfg[args.algo]['final_meta_lr']
    slope = (meta_lr - final_meta_lr) / n_epochs
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch:
                                            (epoch + 1) * slope)
    return optimizer, scheduler
