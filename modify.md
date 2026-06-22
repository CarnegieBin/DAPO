# Muon 优化器集成修改说明

## 概述

将 Muon（MomentUm Orthogonalized by Newton-schulz）优化器集成到 DAPO 训练流程中，实现 Muon + AdamW 混合优化策略：
- **Muon**：用于所有 2D 权重矩阵（attention QKV、MLP up/down/gate 等）
- **AdamW**：用于 embedding、lm_head 和所有 1D 参数（bias、LayerNorm）

---

## 修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `dapo/muon.py` | 新建 | Muon 优化器核心实现 |
| `dapo/train_muon.sh` | 新建 | 使用 Muon 的训练启动脚本 |
| `verl/workers/config/optimizer.py` | 修改 | `build_optimizer` 支持 Muon |
| `verl/workers/engine/fsdp/transformer_impl.py` | 修改 | `_build_optimizer` 传递 module |

---

## 详细修改内容

### 1. `dapo/muon.py`（新建）

从 `toy_train.py` 中提取的 Muon 优化器独立模块，包含：

- **`zeropower_via_newtonschulz5(G, steps)`**：Newton-Schulz 五次迭代，将梯度矩阵正交化为最近正交矩阵，使用 bfloat16 在 GPU 上高效运行。
- **`Muon` 类**：继承 `torch.optim.Optimizer`，内部维护两套更新逻辑：
  - 对 `use_muon=True` 的参数：SGD-Momentum → Newton-Schulz 正交化 → 按矩阵尺寸缩放 lr → 应用更新
  - 对 `use_muon=False` 的参数：标准 AdamW 更新（含 bias correction）
- **`get_optimizer(optimizer_name, model, lr, wd)`**：辅助函数，按参数名自动分组

### 2. `verl/workers/config/optimizer.py`（修改）

**修改函数**：`build_optimizer`

**改动前**：
```python
def build_optimizer(parameters, config: FSDPOptimizerConfig):
```

**改动后**：
```python
def build_optimizer(parameters, config: FSDPOptimizerConfig, module=None):
```

**新增逻辑**（函数开头，早于原有逻辑）：
```python
if config.optimizer == "Muon":
    from dapo.muon import Muon

    assert module is not None, "Muon optimizer requires module for named_parameters access"
    muon_params = [
        p for name, p in module.named_parameters()
        if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
    ]
    adamw_params = [
        p for name, p in module.named_parameters()
        if not (p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name)
    ]
    return Muon(
        lr=config.lr,
        wd=config.weight_decay,
        muon_params=muon_params,
        adamw_params=adamw_params,
    )
```

**设计说明**：
- 通过 `config.optimizer == "Muon"` 判断是否使用 Muon，不影响原有 AdamW 等优化器的构建路径
- 需要 `module` 参数来访问 `named_parameters()`，因为 Muon 需要按参数名称分组
- 参数分组规则：`ndim >= 2` 且名称中不含 `embed_tokens` 和 `lm_head` 的参数使用 Muon，其余使用 AdamW

### 3. `verl/workers/engine/fsdp/transformer_impl.py`（修改）

**修改函数**：`FSDPEngine._build_optimizer`（第 441 行）

**改动前**：
```python
def _build_optimizer(self, module):
    from verl.workers.config.optimizer import build_optimizer
    optimizer = build_optimizer(module.parameters(), self.optimizer_config)
    return optimizer
```

**改动后**：
```python
def _build_optimizer(self, module):
    from verl.workers.config.optimizer import build_optimizer
    optimizer = build_optimizer(module.parameters(), self.optimizer_config, module=module)
    return optimizer
```

**设计说明**：
- 仅增加 `module=module` 关键字参数传递
- 对于非 Muon 优化器，`module` 参数不会被使用，完全向后兼容

### 4. `dapo/train_muon.sh`（新建）

基于 `run_dapo_deepseek_1.5b.sh` 复制，关键差异：

| 配置项 | 原值 | 新值 |
|--------|------|------|
| `exp_name` | `DAPO-DeepSeek-R1-Distill-Qwen-1.5B` | `DAPO-DeepSeek-R1-Distill-Qwen-1.5B-Muon` |
| `actor_rollout_ref.actor.optim.optimizer` | 未设置（默认 AdamW） | `Muon` |
| `actor_rollout_ref.actor.optim.optimizer_impl` | 未设置（默认 torch.optim） | `dapo.muon` |

其余参数（lr=1e-6, weight_decay=0.1, lr_warmup_steps=10 等）保持不变。

---

## 调用链路

```
train_muon.sh
  └─ python3 -m dapo.main_dapo (hydra config: optim.optimizer=Muon)
      └─ DAPOTaskRunner.run()
          └─ RayDAPOTrainer → ActorRolloutRefWorker.init_model()
              └─ TrainingWorker.__init__() → EngineRegistry.new(backend="fsdp")
                  └─ FSDPEngineWithLMHead.initialize()
                      └─ _build_model_optimizer()
                          └─ _build_optimizer(module)  ← 传入 FSDP-wrapped module
                              └─ build_optimizer(params, config, module=module)
                                  └─ config.optimizer == "Muon" → 分组构造 Muon 实例
```

---

## 使用方式

```bash
cd /path/to/DAPO
bash dapo/train_muon.sh
```

---

## 注意事项

1. **学习率**：Muon 的 lr 语义与 AdamW 不同（控制更新的谱范数），当前保持 1e-6 与原配置一致，实际使用中可能需要调大（Muon 论文推荐 0.02 级别用于 pretraining）
2. **LR Scheduler**：当前 scheduler 对所有 param_groups 统一调度，Muon 和 AdamW 部分共享同一 lr schedule
3. **Checkpoint 兼容**：Muon 继承 `torch.optim.Optimizer`，`state_dict()`/`load_state_dict()` 正常工作，但与 AdamW checkpoint 不互通
4. **FSDP 兼容**：FSDP wrapping 后 `named_parameters()` 仍保留原始参数名（含 `_fsdp_wrapped_module` 前缀），`embed_tokens`/`lm_head` 子串匹配不受影响
