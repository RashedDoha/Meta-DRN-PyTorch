{
    'ngpus': 1,
    'metrics': ['iou', 'learner_loss', 'meta_loss'],
    'n_epochs': 5,
    'maml': {
        'learner_lr': 1e-3,
        'meta_lr': 1e-3,


        'halve_lr_every': 8,
        'lr_reduction_factor': 0.5,
        'metric_to_watch': 'mIoU'
    },
    'fomaml': {
        'learner_lr': 1e-3,
        'meta_lr': 1e-3,
        'train_steps': 1,
        'halve_lr_every': 8,
        'lr_reduction_factor': 0.5,
        'metric_to_watch': 'mIoU'
    },
    'meta-sgd': {
        'learner_lr': 1e-3,
        'meta_lr': 1e-3,
        'train_steps': 1,
        'halve_lr_every': 8,
        'lr_reduction_factor': 0.5,
        'metric_to_watch': 'iou'
    },
    'reptile': {
        'learner_lr': 1e-3,
        'meta_lr': 3e-2,
        'train_steps': 5,
        'final_meta_lr': 3e-5
    }
}
