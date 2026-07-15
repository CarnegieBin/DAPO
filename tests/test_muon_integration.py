from types import SimpleNamespace

import torch
from torch import nn

import src.custom_fsdp_engine as custom_fsdp_engine
import src.custom_optimizer as custom_optimizer
import src.custom_worker as custom_worker
from src.custom_muon import Muon
from verl.workers.config.optimizer import FSDPOptimizerConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 4)
        self.embed_tokens = nn.Embedding(8, 3)
        self.lm_head = nn.Linear(3, 8, bias=False)
        self.rank_three = nn.Parameter(torch.zeros(2, 2, 2))


def test_muon_builder_uses_local_implementation_and_partitions_parameters():
    module = ToyModel()
    config = FSDPOptimizerConfig(
        optimizer="Muon",
        lr=1e-6,
        weight_decay=0.1,
    )

    optimizer = custom_optimizer.build_optimizer(module.parameters(), config, module=module)

    assert isinstance(optimizer, Muon)
    assert config.optimizer_impl == "torch.optim"
    assert optimizer._muon_param_ids == {id(module.proj.weight)}
    assert {id(param) for param in optimizer.param_groups[0]["params"]} == {
        id(param) for param in module.parameters()
    }
    assert optimizer.param_groups[0]["lr"] == config.lr
    assert optimizer.param_groups[0]["wd"] == config.weight_decay


def test_non_muon_optimizer_delegates_to_verl_builder(monkeypatch):
    parameters = tuple(ToyModel().parameters())
    config = FSDPOptimizerConfig(optimizer="AdamW")
    sentinel = object()
    calls = []

    def fake_build_optimizer(received_parameters, received_config):
        calls.append((received_parameters, received_config))
        return sentinel

    monkeypatch.setattr(custom_optimizer, "verl_build_optimizer", fake_build_optimizer)

    assert custom_optimizer.build_optimizer(parameters, config) is sentinel
    assert calls == [(parameters, config)]


def test_fsdp_installer_builds_muon_with_module(monkeypatch):
    module = ToyModel()
    config = FSDPOptimizerConfig(optimizer="Muon")
    engine = SimpleNamespace(optimizer_config=config)
    original_builder = FSDPEngine._build_optimizer
    monkeypatch.setattr(FSDPEngine, "_build_optimizer", original_builder)

    custom_fsdp_engine.install_custom_fsdp_optimizer_builder()
    optimizer = FSDPEngine._build_optimizer(engine, module)

    assert FSDPEngine._build_optimizer is custom_fsdp_engine._build_optimizer_with_module
    assert isinstance(optimizer, Muon)


def test_dapo_worker_installs_builder_before_base_initialization(monkeypatch):
    calls = []

    monkeypatch.setattr(
        custom_worker,
        "install_custom_fsdp_optimizer_builder",
        lambda: calls.append("install"),
    )
    monkeypatch.setattr(
        custom_worker.ActorRolloutRefWorker,
        "__init__",
        lambda self, *args, **kwargs: calls.append("base"),
    )

    custom_worker.DAPOActorRolloutRefWorker(object(), "actor_rollout")

    assert calls == ["install", "base"]


def test_dapo_task_runner_registers_custom_worker(monkeypatch):
    import src.main_dapo as main_dapo
    from verl.trainer.ppo.ray_trainer import Role

    runner = main_dapo.DAPOTaskRunner()
    config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(model={}))
    monkeypatch.setattr(main_dapo, "need_reference_policy", lambda _: False)
    monkeypatch.setattr(main_dapo.ray, "remote", lambda cls: ("remote", cls))

    actor_rollout_cls, _ = runner.add_actor_rollout_worker(config)

    assert actor_rollout_cls is custom_worker.DAPOActorRolloutRefWorker
    assert runner.role_worker_mapping[Role.ActorRollout] == (
        "remote",
        custom_worker.DAPOActorRolloutRefWorker,
    )
