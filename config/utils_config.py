"""Configs for utlities and transformations"""
from .data_config import cfg as data_cfg

cfg = {
    'transforms': [{
        'transform': 'Resize',
        'params': {
            'height': data_cfg['img_height'],
            'width': data_cfg['img_width']
        }
    }, {
        'transform': 'HorizontalFlip'
    }, {
        'transform': 'VerticalFlip'
    }, {
        'transform': 'ShiftScaleRotate',
        'params': {
            'shift_limit': 0,
            'rotate_limit': 0
        }
    }, {
        'transform': 'RandomBrightnessContrast'
    }, {
        'transform': 'Normalize',
        'params': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }]
}
