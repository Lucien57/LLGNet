"""EEGNet with adversarial subject head, single-stage training."""

from copy import deepcopy

from config.config_EEGNet import model_params as _base_cfg


def _build():
    cfg = deepcopy(_base_cfg)
    cfg["model_args"]["enable_adversarial_head"] = True
    cfg["train"]["adv_lambda"] = 0.01
    cfg["train"]["two_stage"] = False
    cfg["train"]["two_stage_extra_epochs"] = 0
    return cfg


model_params = _build()
