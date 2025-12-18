"""ShallowConvNet with adversarial head and two-stage training."""

from copy import deepcopy

from config.config_ShallowConvNet import model_params as _base_cfg


def _build():
    cfg = deepcopy(_base_cfg)
    cfg["model_args"]["enable_adversarial_head"] = True
    cfg["train"]["adv_lambda"] = cfg["train"].get("adv_lambda", 0.01)
    cfg["train"]["two_stage"] = True
    if cfg["train"].get("two_stage_extra_epochs", 0) <= 0:
        cfg["train"]["two_stage_extra_epochs"] = 30
    return cfg


model_params = _build()


