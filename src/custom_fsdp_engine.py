"""Install the module-aware FSDP optimizer builder used by DAPO."""

from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def _build_optimizer_with_module(self, module):
    from src.custom_optimizer import build_optimizer

    return build_optimizer(module.parameters(), self.optimizer_config, module=module)


def install_custom_fsdp_optimizer_builder():
    """Install the custom builder in the process that creates the FSDP engine."""
    FSDPEngine._build_optimizer = _build_optimizer_with_module
