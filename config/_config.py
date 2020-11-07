"""Utility module for retrieving configs"""
from .global_config import cfg as gcfg
from .model_config import cfg as mcfg
from .train_config import cfg as tcfg
from .data_config import cfg as dcfg
from .utils_config import cfg as ucfg
from .vis_config import cfg as vcfg

_merged_cfg = [gcfg, mcfg, tcfg, dcfg, ucfg, vcfg]
_cfg_names = [
    'global_config', 'model_config', 'train_config', 'data_config',
    'utils_config', 'vis_config'
]

_cfg = {}
for cname, cfg in zip(_cfg_names, _merged_cfg):
  _cfg[cname] = cfg
