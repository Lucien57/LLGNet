model_params = {
    "paths":     {
        "lmdb_path":     {
            "BCIC-IV-2a": 'lmdb/BCIC-IV-2a',
            "BCIC-IV-2b": 'lmdb/BCIC-IV-2b',
        },
    },
    "model":     "ShallowConvNet",
    "model_args":     {
        "dropoutRate": 0.5,
        "batch_norm": True,
        "batch_norm_alpha": 0.1,
        "enable_adversarial_head": True,
        "adv_lambda": 0.01, 
    },
    "train":     {
        "batch_size": 64,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "max_epochs": 200,
        "early_stopping_patience": 20,
        "num_workers": 4,
        "random_seed": 42,
        "two_stage": False,             
        "two_stage_extra_epochs": 30,   # 第二阶段额外 epoch
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
            "PhysioNet-MI": (64, 800),
            "PhysioNet-MI-2class": (64, 800),
            "BCIC-IV-2a": (22, 800),
            "BCIC-IV-2a-2class": (22, 800),
            "BCIC-IV-2b": (3, 800),
            "FACED": (30, 100),
            "SEED_V": (62, 400),
            "Mumtaz2016": (19, 400),
        },
    },
}
