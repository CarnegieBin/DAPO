"""DAPO-specific Ray workers."""

from src.custom_fsdp_engine import install_custom_fsdp_optimizer_builder
from verl.workers.engine_workers import ActorRolloutRefWorker


class DAPOActorRolloutRefWorker(ActorRolloutRefWorker):
    """Actor worker that installs DAPO optimizer support in its Ray process."""

    def __init__(self, *args, **kwargs):
        install_custom_fsdp_optimizer_builder()
        super().__init__(*args, **kwargs)
