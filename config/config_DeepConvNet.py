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
         "norm_rate": 0.25,
        # Toggle adversarial head support without switching build_model entries
        "enable_adversarial_head": True,
        # 对抗头参数（仅 enable_adversarial_head=True 时生效）
        # "n_nuisance": 9,
        "adv_lambda": 0.01,
    },
    "train":     {
        "batch_size": 64,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "max_epochs": 100,
        "early_stopping_patience": 20,
        "num_workers": 4,
        "random_seed": 42,
        "two_stage": True,
        "two_stage_extra_epochs": 30,
        # 对抗损失权重（仅对抗版启用）
        "adv_lambda": 0.01,
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
