model_config = {
    'head': {
        'conv1': {
            'in_channels':3,
            'out_channels':16,
            'kernel_size':3,
            'stride':2,
            'padding':1,
            'dilation':1
        },
        'bn1': {
            'num_features':16
        },
        'conv2': {
            'in_channels':16,
            'out_channels':64,
            'kernel_size':3,
            'stride':1,
            'padding':1,
            'dilation':1
        },
        'bn2': {
            'num_features':64
        }
    },
    'resblocks': {
        'resblock1': {
                'conv1': {
                'in_channels':64,
                'out_channels':128,
                'kernel_size':3,
                'stride':2,
                'padding':1,
                'dilation':1                
            },
            'conv2': {
                'in_channels':128,
                'out_channels':128,
                'kernel_size':3,
                'stride':1,
                'padding':1,
                'dilation':1             
            }
        },
        'resblock2': {
            'conv1': {
                'in_channels':128,
                'out_channels':256,
                'kernel_size':3,
                'stride':1,
                'padding':1,
                'dilation':1
            },
            'conv2': {
                'in_channels':256,
                'out_channels':256,
                'kernel_size':3,
                'stride':1,
                'padding':2,
                'dilation':2
            }

        },
        'resblock3': {
            'conv1': {
                'in_channels':256,
                'out_channels':512,
                'kernel_size':3,
                'stride':1,
                'padding':2,
                'dilation':2          
            },
            'conv2': {
                'in_channels':512,
                'out_channels':512,
                'kernel_size':3,
                'stride':1,
                'padding':4,
                'dilation':4
            }
        }
    },
    'reducer': {
        'resblock1': {
            'in_channels':64,
            'out_channels':128,
            'kernel_size':1,
            'stride':2,
            'padding':0,
            'dilation':1
        },
        'resblock2': {
            'in_channels':128,
            'out_channels':256,
            'kernel_size':1,
            'stride':1,
            'padding':0,
            'dilation':1
        },
        'resblock3': {
            'in_channels':256,
            'out_channels':512,
            'kernel_size':1,
            'stride':1,
            'padding':0,
            'dilation':1
        }
    },
    'degrid': {
        'conv1': {
            'in_channels': 512,
            'out_channels':512,
            'kernel_size':3,
            'stride':1,
            'padding':2,
            'dilation':2
        },
        'conv2': {
            'in_channels': 512,
            'out_channels':512,
            'kernel_size':3,
            'stride':1,
            'padding':1,
            'dilation':1            
        }
    },
    'upsample': {
        'conv': {
            'in_channels': 512,
            'out_channels':48,
            'kernel_size':3,
            'stride':1,
            'padding':1,
            'dilation':1            
        },
        'pixel_shuffle': {
            'upscale_factor': 4
        }
    }
}

data_config = {
    'dataset_name': 'FSS-1000',
    'gdrive_file_id': '16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI',
    'dataset_dir': 'fewshot_data',
    'img_size':224,
    'n_classes': 1000,
    'n_train_classes': 700,
    'n_val_classes': 60,
    'n_test_classes': 240,
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225]
}

train_config = {
    'max_epochs': 200,
    'maml': {
        'batch_size': 5,
        'learner_lr': 1e-3,
        'meta_lr': 1e-3,
        'train_steps': 1,
        'halve_lr_every': 8,

    },
    'fomaml': {
        'batch_size': 5,
        'learner_lr': 1e-3,
        'meta_lr': 1e-3,
        'train_steps': 1,       
        'halve_lr_every': 8,
        
    },
    'meta-sgd': {
        'batch_size': 5,
        'learner_lr': 1e-3,
        'meta_lr': 1e-3,
        'train_steps': 1,
        'halve_lr_every': 8,
    },
    'reptile': {
        'batch_size': 8,
        'learner_lr': 1e-3,
        'meta_lr': 3e-2,
        'train_steps': 5,
        'final_meta_lr': 3e-5
    }
}