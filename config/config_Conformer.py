model_params = {
    "paths":     {
        "lmdb_path":     {
            "BCIC-IV-2a": 'lmdb/BCIC-IV-2a',
            "BCIC-IV-2b": 'lmdb/BCIC-IV-2b',
        },
    },
    "model":     "Conformer",
    "model_args":     {
        "emb_size": 40,  # 原64
        "num_heads": 8,
        "depth": 6,
        "patch_dropout": 0.25,
        "drop_p": 0.25,
        "forward_drop_p": 0.25,
    },
    "train":     {
        "batch_size": 64,
        "learning_rate": 0.0002,
        "betas": (0.9, 0.999),
        "weight_decay": 0.001,
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
            "use_sr_augmentation": True,
            "sr_n_segments": 8,
        },
    },
    "defaults":     {
        "dataset_shapes":     {
            "BCIC-IV-2a": (22, 1000),
            "BCIC-IV-2b": (3, 1000),
        },
    },
}
