from .global_config import gcfg
from .model_config import mcfg
from .train_config import tcfg
from .data_config import dcfg
from .utils_config import ucfg
from .vis_config import vcfg, vis_config

_merged_cfg = [gcfg, mcfg, tcfg, dcfg, ucfg, vcfg]
_cfg_names = ['global_config', 'model_config', 'train_config', 'data_config',
              'utils_config', vis_config]

_cfg = {}
for cname, cfg in zip(_cfg_names, _merged_cfg):
    _cfg[cname] = cfg
