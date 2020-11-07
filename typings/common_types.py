"""A list of common types used in multiple places.
"""
import builtins
from torch import Tensor
from typing import Callable, Dict, Tuple, Union, Optional, Type, Any, Mapping
from torch.types import Device
from torch.nn import Module

_int = builtins.int
_float = builtins.float
_tensor = Tensor

_opt_int = Optional[_int]
_opt_float = Optional[_float]
_opt_tensor = Optional[Tensor]

_inttuple = Tuple[_int, _int]
_floattuple = Tuple[_float, _float]
_strtuple = Tuple[str, str]
_ttuple = Tuple[Tensor, Tensor]
_opt_ttuple = Optional[_ttuple]

_intstr = Union[_int, str]
_intfloat = Union[_int, _float]
