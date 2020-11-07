"""Configs for the model hyperparameters"""
cfg = {
    'head': {
        'conv1': {
            'in_channels': 3,
            'out_channels': 16,
            'kernel_size': 3,
            'stride': 2,
            'padding': 1,
            'dilation': 1
        },
        'bn1': {
            'num_features': 16
        },
        'conv2': {
            'in_channels': 16,
            'out_channels': 64,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'dilation': 1
        },
        'bn2': {
            'num_features': 64
        }
    },
    'resblocks': {
        'resblock1': {
            'conv1': {
                'in_channels': 64,
                'out_channels': 128,
                'kernel_size': 3,
                'stride': 2,
                'padding': 1,
                'dilation': 1
            },
            'conv2': {
                'in_channels': 128,
                'out_channels': 128,
                'kernel_size': 3,
                'stride': 1,
                'padding': 1,
                'dilation': 1
            }
        },
        'resblock2': {
            'conv1': {
                'in_channels': 128,
                'out_channels': 256,
                'kernel_size': 3,
                'stride': 1,
                'padding': 1,
                'dilation': 1
            },
            'conv2': {
                'in_channels': 256,
                'out_channels': 256,
                'kernel_size': 3,
                'stride': 1,
                'padding': 2,
                'dilation': 2
            }
        },
        'resblock3': {
            'conv1': {
                'in_channels': 256,
                'out_channels': 512,
                'kernel_size': 3,
                'stride': 1,
                'padding': 2,
                'dilation': 2
            },
            'conv2': {
                'in_channels': 512,
                'out_channels': 512,
                'kernel_size': 3,
                'stride': 1,
                'padding': 4,
                'dilation': 4
            }
        }
    },
    'reducer': {
        'resblock1': {
            'in_channels': 64,
            'out_channels': 128,
            'kernel_size': 1,
            'stride': 2,
            'padding': 0,
            'dilation': 1
        },
        'resblock2': {
            'in_channels': 128,
            'out_channels': 256,
            'kernel_size': 1,
            'stride': 1,
            'padding': 0,
            'dilation': 1
        },
        'resblock3': {
            'in_channels': 256,
            'out_channels': 512,
            'kernel_size': 1,
            'stride': 1,
            'padding': 0,
            'dilation': 1
        }
    },
    'degrid': {
        'conv1': {
            'in_channels': 512,
            'out_channels': 512,
            'kernel_size': 3,
            'stride': 1,
            'padding': 2,
            'dilation': 2
        },
        'conv2': {
            'in_channels': 512,
            'out_channels': 512,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'dilation': 1
        }
    },
    'upsample': {
        'conv': {
            'in_channels': 512,
            'out_channels': 32,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'dilation': 1
        },
        'pixel_shuffle': {
            'upscale_factor': 4
        }
    }
}
