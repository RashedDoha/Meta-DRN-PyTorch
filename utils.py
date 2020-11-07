"""Contains utlity functions"""

import time

import albumentations as A
import requests
from albumentations.pytorch.transforms import ToTensor
from requests.models import Response
from tqdm import tqdm

from config.utils_config import cfg
from typings.common_types import Callable, Module, Union, _float, _int


def download_file_from_google_drive(file_id: str, destination: str) -> None:
  print('Downloading ', destination.rpartition('/')[-1])
  url = 'https://docs.google.com/uc?export=download'

  session = requests.Session()

  response = session.get(url, params={'id': file_id}, stream=True)
  token = get_confirm_token(response)

  if token:
    params = {'id': file_id, 'confirm': token}
    response = session.get(url, params=params, stream=True)

  save_response_content(response, destination)


def get_confirm_token(response: Response) -> Union[str, None]:
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      return value

  return None


def save_response_content(response: Response, destination: str) -> None:
  chunk_size = 32768

  with open(destination, 'wb') as f:
    pbar = tqdm(total=None)
    progress = 0
    for chunk in response.iter_content(chunk_size):
      if chunk:  # filter out keep-alive new chunks
        progress += len(chunk)
        pbar.update(progress - pbar.n)
        f.write(chunk)
    pbar.close()


def count_parameters(model: Module) -> _int:
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_transforms() -> A.Compose:
  transforms = cfg['transforms']

  ts = []
  for t in transforms:
    trans = t['transform']
    params = trans.get('params', {})

    if hasattr(A, trans):
      if params is not None:

        ts.append(getattr(A, trans)(**params))
      else:
        ts.append(getattr(A, trans)(**params))

  transform = A.Compose([*ts, ToTensor()])
  return transform


def time_func(func: Callable[...]) -> _float:
  t1 = time.time()
  func()
  t2 = time.time()
  print('Time taken: %.2f %s' % (t2 - t1, 'seconds'))
  return t2 - t1


def main() -> None:
  return


if __name__ == '__main__':
  main()
