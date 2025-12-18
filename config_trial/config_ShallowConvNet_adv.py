"""ShallowConvNet with adversarial subject head, single-stage training."""

from copy import deepcopy

from config.config_ShallowConvNet import model_params as _base_cfg


def _build():
    cfg = deepcopy(_base_cfg)
    cfg["model_args"]["enable_adversarial_head"] = True
    # 使用 base config 里的 adv_lambda 作为默认权重
    cfg["train"]["adv_lambda"] = cfg["train"].get("adv_lambda", 0.01)
    cfg["train"]["two_stage"] = False
    cfg["train"]["two_stage_extra_epochs"] = 0
    return cfg


model_params = _build()


