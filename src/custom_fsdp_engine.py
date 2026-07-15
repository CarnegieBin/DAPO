"""Monkey-patch FSDPEngine._build_optimizer to pass module for Muon support."""

from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def _build_optimizer_with_module(self, module):
    from src.custom_optimizer import build_optimizer
    return build_optimizer(module.parameters(), self.optimizer_config, module=module)


FSDPEngine._build_optimizer = _build_optimizer_with_module
