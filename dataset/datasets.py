"""Helper functions for retrieving dataloaders"""

from typing import Tuple

from config.data_config import cfg
from torch import Tensor
from torch.utils.data import DataLoader
from utils import get_transforms

from .fewshot import FSSDataset


def get_dataset(algo: str, meta_split='train'):
  """Retrieves dataset corresponding to parameters
   defined in config module for the meta learning algorithm used.

  Args:
      algo (str): Meta learning algorithm from [maml, fomaml,\
       meta-sgd, reptile]
      meta_split (str, optional): 'train' or 'test' split of the data.\
       Defaults to 'train'.

  Returns:
      Dataset: An instance of the Dataset class.
  """
  transform = get_transforms()
  data_root = cfg['data_root']
  n_ways = cfg[algo]['n_ways']
  train_shots = cfg[algo]['train_shots']
  test_shots = cfg[algo]['test_shots']

  dataset = FSSDataset(data_root, n_ways, train_shots, test_shots, meta_split,
                       transform)
  return dataset


def get_dataloader(algo: str, meta_split='train'):
  """Retrieves a PyTorch dataloader. Parameters for the dataloader can be
    found in the data_config module

  Args:
      algo (str): Meta learning algorithm from [maml, fomaml,
       meta-sgd, reptile]
      meta_split (str, optional): 'train' or 'test' split of the data.
       Defaults to 'train'.

  Returns:
      DataLoader: An instance of DataLoader class.
  """
  batch_size = cfg['batch_size']
  num_workers = cfg['num_workers']
  dataset = get_dataset(algo, meta_split)
  shuffle = meta_split == 'train'
  return DataLoader(dataset,
                    batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers)


def split_batch(
    batch: Tensor,
    algo: str) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
  """Splits a batch of data into image and mask pairs from the support
  set and the query set.

  Args:
      batch (Tensor): A torch.Tensor returned by the dataloader
      algo (str): Meta learning algorithm from [maml, fomaml,
       meta-sgd, reptile]

  Returns:
      [[Tensor, Tensor], [Tensor, Tensor]: Four Tensors paired into two tuples.
      (support_images, support_targets), (query_images, query_targets)
  """
  n_ways = cfg[algo]['n_ways']
  train_shots = cfg[algo]['train_shots']
  return FSSDataset.break_batch(batch, train_shots, n_ways)
