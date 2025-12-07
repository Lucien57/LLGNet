model_params = {
    "paths":     {
        "lmdb_path":     {
            "BCIC-IV-2a": 'lmdb/BCIC-IV-2a',
            "BCIC-IV-2b": 'lmdb/BCIC-IV-2b',
        },
    },
    "model":     "DeepConvNet",
    "model_args":     {
        "dropoutRate": 0.5,
        "batch_norm": True,
        "batch_norm_alpha": 0.1,
    },
    "train":     {
        "batch_size": 64,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "max_epochs": 100,
        "early_stopping_patience": 20,
        "num_workers": 4,
        "random_seed": 42,
    },
    "eval":     {
        "k_folds": 5,
    },
    "data":     {
        # "use_zscore_normalization": True,
        "Norm": "EA",
        "use_augmentation": False,
        "augmentation":     {
            "S": True,
            "R": True,
        },
    },
    "defaults":     {
        "dataset_shapes":     {
            "BCIC-IV-2a": (22, 1000),
            "BCIC-IV-2b": (3, 1000),
        },
    },
}
