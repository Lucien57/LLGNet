model_params = {
    "paths":     {
        "lmdb_path":     {
            "BCIC-IV-2a": 'lmdb/BCIC-IV-2a',
            "BCIC-IV-2b": 'lmdb/BCIC-IV-2b',
        },
    },
    # Use EEGNet with Conditional Domain Adversarial head
    "model":     "EEGNet",
    "model_args":     {
        "kernLength": 64,
        "F1": 8,
        "D": 2,
        "F2": 16,
        "dropoutRate": 0.25,
        "norm_rate": 0.25,
        # Enable adversarial head and conditional variant
        "enable_adversarial_head": True,
        "enable_conditional_adversarial_head": True,
        # 对抗头参数（可选：默认会使用 subject count）
        # "n_nuisance": 9,
        "adv_lambda": 0.01,
    },
    "train":     {
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "max_epochs": 100,
        "early_stopping_patience": 20,
        "num_workers": 4,
        "random_seed": 57,
        "two_stage": True,
        "two_stage_extra_epochs": 30,
        # 对抗损失权重（训练时也可在 train 中设置）
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


