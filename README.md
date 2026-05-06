# DAPO 复现

本项目基于 [verl](https://github.com/volcengine/verl) 框架，对 **DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization）** 算法进行复现。

DAPO 由字节跳动与清华大学联合提出，将 GRPO 的 Clip 机制解耦为高低两个阈值，并引入动态采样策略，在 Qwen2.5-32B 基座模型上以更少的训练步数超越 DeepSeek-R1-Zero-Qwen-32B，在 AIME 2024 上达到 **50% 准确率**。

- 论文：[arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
- 原始仓库：[BytedTsinghua-SIA/DAPO](https://github.com/BytedTsinghua-SIA/DAPO)

---

## 环境配置

### Docker 容器启动（含磁盘挂载）

服务器上使用 `verlai/verl:vllm011.latest` 镜像运行训练环境。将宿主机的 `/home`、`/ssd1`、`/ssd2`、`/ssd3` 全部挂载进容器，保证容器内路径与宿主机完全一致，训练脚本无需修改路径。

**创建并启动容器：**

```bash
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" \
  --cap-add=SYS_ADMIN \
  --entrypoint sleep \
  -v /home:/home \
  -v /ssd1:/ssd1 \
  -v /ssd2:/ssd2 \
  -v /ssd3:/ssd3 \
  --name verl \
  verlai/verl:vllm011.latest \
  infinity

docker start verl
docker exec -it verl bash
```

**重建容器（先停止并删除旧容器）：**

```bash
docker stop verl && docker rm verl
```

然后重新执行上方的 `docker create` 命令。

**验证挂载：**

```bash
ls /home
ls /ssd1
ls /ssd2
ls /ssd3
```

### 说明

| 参数 | 说明 |
|------|------|
| `--runtime=nvidia --gpus all` | 使用全部 GPU |
| `--net=host` | 使用宿主机网络，多节点训练时必须 |
| `--shm-size="10g"` | 共享内存，训练时建议不低于 10g |
| `--entrypoint sleep` + `infinity` | 覆盖镜像默认入口，保持容器常驻 |
| `-v /home:/home` 等 | 将宿主机磁盘挂载进容器，路径保持一致 |

---

## 快速开始

进入容器后，切换到项目目录启动训练：

```bash
cd /home/work/tcbian/DAPO
bash dapo/run_dapo_deepseek_1.5b.sh
```

训练脚本默认路径配置：

| 内容 | 路径 |
|------|------|
| 训练/验证数据集 | `/home/work/tcbian/ExpThink/data` |
| 模型权重 | `/ssd2/llm_models/DeepSeek-R1-Distill-Qwen-1.5B` |
| Checkpoint 输出 | 脚本同级目录 `ckpts/` 下 |

详细算法配置参见 [dapo/README.md](dapo/README.md)。
