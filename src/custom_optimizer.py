"""Modified build_optimizer that supports Muon optimizer while remaining backward-compatible."""

from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer as verl_build_optimizer


def build_optimizer(parameters, config: FSDPOptimizerConfig, module=None):
    if config.optimizer == "Muon":
        from src.custom_muon import Muon

        assert module is not None, "Muon optimizer requires module for named_parameters access"
        muon_params = [
            p for name, p in module.named_parameters()
            if p.ndim == 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p for name, p in module.named_parameters()
            if not (p.ndim == 2 and "embed_tokens" not in name and "lm_head" not in name)
        ]
        return Muon(
            lr=config.lr,
            wd=config.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )

    return verl_build_optimizer(parameters, config)
