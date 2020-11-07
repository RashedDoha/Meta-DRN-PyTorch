import builtins
from torch import Tensor
from torch.nn import Module
from typing import Callable
import builtins

_int = builtins.int


class MetaDRN(Module):

  def __init__(self, loss_fn: Callable[[Tensor], Tensor]) -> None:
    ...

  def __call__(self, *inputs: Tensor) -> Tensor:
    ...

  def init_weights(self, seed: _int) -> None:
    ...

  def copy_weights(self, net: MetaDRN) -> None:
    ...

  ...