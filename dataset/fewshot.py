"""Implementation of the torch dataset"""
import os
import random
import shutil
import zipfile

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensor
from config.data_config import cfg
from PIL import Image
from torch.utils.data import Dataset
from typings.common_types import Any, Optional, _int
from utils import download_file_from_google_drive


class FSSDataset(Dataset):
  """A subclass of torch.utils.data.Dataset that reads images
   from the  dataset folder into the appropriate support and query
   sets defined in data_config.
  """
  folder = cfg['dataset_dir']

  def __init__(self,
               root: str,
               ways: _int,
               shots: _int,
               test_shots: _int,
               meta_split: Optional[str] = 'train',
               transform: Optional[Any] = None,
               download: Optional[bool] = True):
    super().__init__()
    assert meta_split in ['train', 'val',
                          'test'], "meta-split must be either 'train',\
                 'val' or 'test'"

    self.ways = ways
    self.shots = shots
    self.transform = transform
    self.test_shots = test_shots
    self.meta_split = meta_split
    if transform is None:
      self.transform = A.Compose([
          A.Normalize(mean=cfg['normalize_mean'], std=cfg['normalize_std']),
          ToTensor()
      ])
    else:
      self.transform = transform
    if download:
      self.download(root)

    self.root = os.path.expanduser(os.path.join(root, self.folder))
    all_classes = os.listdir(self.root)

    if meta_split == 'train':
      self.classes = [all_classes[i] for i in range(cfg['n_train_classes'])]
    elif meta_split == 'val':
      self.classes = [
          all_classes[i]
          for i in range(cfg['n_train_classes'], cfg['n_train_classes'] +
                         cfg['n_val_classes'])
      ]
    else:
      self.classes = [
          all_classes[i]
          for i in range(cfg['n_train_classes'] +
                         cfg['n_val_classes'], cfg['n_classes'])
      ]

    self.num_classes = len(self.classes)

  def thresh_mask(self, mask, thresh=0.5):
    thresh = (mask.min() + mask.max()) * thresh
    mask = mask > thresh
    return mask.long()

  def make_batch(self, classes):
    shots = self.shots + self.test_shots
    batch = torch.zeros((shots, self.ways, 4, 224, 224))

    for i in range(shots):
      for j, cname in enumerate(classes):
        img_id = str(random.choice(list(range(1, 11))))
        img = Image.open(os.path.join(self.root, cname,
                                      img_id + '.jpg')).convert('RGB')
        mask = Image.open(os.path.join(self.root, cname,
                                       img_id + '.png')).convert('RGB')
        img, mask = np.array(img), np.array(mask)[:, :, 0]
        transformed = self.transform(image=img, mask=mask)
        batch[i, j, :3, :, :] = transformed['image']
        batch[i, j, 3:, :, :] = self.thresh_mask(transformed['mask'])

    return batch

  @staticmethod
  def break_batch(batch, shots, ways, shuffle=True):
    permute = torch.randperm(ways) if shuffle else torch.arange(ways)
    train_images, train_masks = batch[:, :shots, permute, :3, :, :],\
        batch[:, :shots, permute, 3:, :, :]

    test_images, test_masks = batch[:, shots:, :, :3, :, :],\
        batch[:, shots:, :, 3:, :, :]

    return (train_images, train_masks), (test_images, test_masks)

  def __getitem__(self, class_index):
    classes = [
        self.classes[i] for i in range(class_index, (class_index + self.ways) %
                                       self.num_classes)
    ]
    batch = self.make_batch(classes)
    return batch

  def __len__(self):
    return self.num_classes

  def download(self, root, remove_zip=True):
    filename = cfg['dataset_dir'] + '.zip'

    if os.path.exists(root):
      return

    file_id = cfg['gdrive_file_id']

    download_file_from_google_drive(file_id, filename)

    with zipfile.ZipFile(filename, 'r') as f:
      f.extractall()

    if remove_zip:
      os.remove(filename)

    shutil.move(cfg['dataset_dir'], root)
