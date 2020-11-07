from torch.utils.data import DataLoader
from .FSS import FSSDataset
from utils import get_transforms
from config.data_config import cfg


def get_dataset(algo: str, meta_split='train'):
  transform = get_transforms()
  data_root = cfg['data_root']
  n_ways = cfg[algo]['n_ways']
  train_shots = cfg[algo]['train_shots']
  test_shots = cfg[algo]['test_shots']

  dataset = FSSDataset(data_root, n_ways, train_shots, test_shots, meta_split,
                       transform)
  return dataset


def get_dataloader(algo: str, meta_split='train'):
  batch_size = cfg['batch_size']
  num_workers = cfg['num_workers']
  dataset = get_dataset(algo, meta_split)
  shuffle = meta_split == 'train'
  return DataLoader(dataset,
                    batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers)


def split_batch(batch, algo):
  n_ways = cfg[algo]['n_ways']
  train_shots = cfg[algo]['train_shots']
  return FSSDataset.break_batch(batch, train_shots, n_ways)
