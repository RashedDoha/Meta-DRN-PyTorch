cfg = {
    'data_root': 'data',
    'dataset_name': 'FSS-1000',
    'gdrive_file_id': '16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI',
    'dataset_dir': 'fewshot_data',
    'img_height': 224,
    'img_width': 224,
    'n_classes': 1000,
    'n_train_classes': 700,
    'n_val_classes': 60,
    'n_test_classes': 240,
    'shuffle': True,
    'num_workers': 4,
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225],
    'batch_size': 8,
    'maml': {

        'test_shots': 5,
        'train_shots': 1,
        'n_ways': 4
    },
    'fomaml': {
        'test_shots': 5,
        'train_shots': 1,
        'n_ways': 5
    },
    'metasgd': {
        'test_shots': 5,
        'train_shots': 1,
        'n_ways': 5
    },
    'reptile': {
        'test_shots': 8,
        'train_shots': 5,
        'n_ways': 5
    }
}
