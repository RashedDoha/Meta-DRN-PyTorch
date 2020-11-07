""" Argument parser for obtaining command line parameters passed 
to the training script."""
import argparse
from argparse import Namespace
from typings.common_types import _int, Dict, Union
from pytorch_lightning import Trainer
from config.data_config import cfg as dcfg
from config.train_config import cfg as tcfg


def get_args() -> Namespace:
  """Returns a namespace containing the command line arguments passed to the 
  command line.

  Returns:
      Dict[str, Union[str, _int]]: A python dictionary where the keys 
      are the argument names and the values are either default or user provided.
  """
  parser = argparse.ArgumentParser(
      'Meta-DRN: Meta-Learning for 1-Shot Image Segmentation')
  parser = Trainer.add_argparse_args(parser)
  parser.add_argument('--algo',
                      type=str,
                      default='maml',
                      help='A meta learning algorithms to use out of maml,\
                        fomal, meta-sgd and reptile')
  parser.add_argument('--folder',
                      type=str,
                      default=dcfg['data_root'],
                      help='Path to the folder the data is downloaded to.')
  parser.add_argument('--n_epochs',
                      type=int,
                      default=tcfg['n_epochs'],
                      help='Number of epochs to train the meta learner')
  parser.add_argument('--num-workers',
                      type=int,
                      default=dcfg['num_workers'],
                      help='Number of workers for data loading\
                                 (default: 1).')
  parser.add_argument('--batch_size',
                      type=int,
                      default=dcfg['batch_size'],
                      help='Number of tasks in a mini-batch of tasks\
                        (default: 16).')
  parser.add_argument('--output-folder',
                      type=str,
                      default=None,
                      help='Path to the output folder for saving\
                                 the model (optional).')
  parser.add_argument('--mode',
                      type=str,
                      default='train',
                      help='Whether the network is used\
                        for training or testing.')

  args = parser.parse_args()
  return args
