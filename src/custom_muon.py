import math
import torch


# Polar Express coefficients from https://arxiv.org/pdf/2505.16932
_POLAR_EXPRESS_RAW = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
]
_SAFETY = 1.05
POLAR_EXPRESS_COEFFICIENTS = [
    (a / _SAFETY, b / _SAFETY**3, c / _SAFETY**5)
    for (a, b, c) in _POLAR_EXPRESS_RAW
]


@torch.compile
def gram_newton_schulz(G, steps=5):
    """
    Gram Newton-Schulz: operates on the n×n Gram matrix instead of n×m,
    with a restart after iteration 2 for numerical stability in float16.
    Reference: https://dao-lab.ai/blog/2026/gram-newton-schulz/
    """
    assert len(G.shape) == 2
    X = G.float()
    if X.size(0) > X.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    X = X.half()

    coeffs = POLAR_EXPRESS_COEFFICIENTS[:steps]
    n = X.size(0)
    I = torch.eye(n, device=X.device, dtype=X.dtype)
    R = X @ X.mT
    Q = None

    for i, (a, b, c) in enumerate(coeffs):
        if i == 2 and Q is not None:
            X = Q @ X
            R = X @ X.mT
            Q = None

        Z = torch.baddbmm(R, R, R, beta=b, alpha=c)
        if Q is None:
            Q = Z + a * I
        else:
            Q = torch.baddbmm(Q, Q, Z, beta=a, alpha=1.0)
        if i < len(coeffs) - 1 and (i + 1) != 2:
            RZ = torch.baddbmm(R, R, Z, beta=a, alpha=1.0)
            R = torch.baddbmm(RZ, Z, RZ, beta=a, alpha=1.0)

    X = Q @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


# =============================================================================
# H100/H800 (Hopper, SM90) 版本 — 使用 Quack 库的自定义对称 GEMM kernel
# 相比纯 cuBLAS 版本额外提速 ~25-30%（对称矩阵只计算下三角 + 转置写入上三角）
# 需要: pip install quack (https://github.com/Dao-AILab/quack)
# 硬件要求: NVIDIA Hopper (H100/H800) 或 Blackwell 架构，不兼容 Ampere (A100/A800)
# =============================================================================
#
# from gram_newton_schulz import GramNewtonSchulz, POLAR_EXPRESS_COEFFICIENTS
#
# # 初始化（仅需一次，ns_use_kernels=True 启用 Quack 对称 GEMM kernel）
# _gram_ns_hopper = GramNewtonSchulz(
#     ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
#     ns_use_kernels=True,  # 启用 CuTeDSL 对称 GEMM kernel (Hopper/Blackwell only)
#     gram_newton_schulz_reset_iterations=[2],
#     compile_kwargs={"fullgraph": True, "mode": "reduce-overhead"},
# )
#
# def gram_newton_schulz_hopper(G, steps=5):
#     """
#     H100/Hopper 专用版本，利用 Quack 对称 GEMM kernel 获得额外加速。
#     对比纯 PyTorch 版本:
#       - 对称 GEMM 只计算下三角，FLOP 减半
#       - 利用 Hopper TMA/Cluster/Ping-Pong Scheduling 硬件特性
#       - 矩阵尺寸 > 256 时自动启用 kernel，否则 fallback 到 cuBLAS
#     在 Kimi K2 配置下 (384 experts, hidden=7168) 比标准 NS 快 2x。
#     """
#     return _gram_ns_hopper(G)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Store muon param ids separately so FSDP state offload/reload won't lose them
        self._muon_param_ids = set()
        for p in muon_params:
            assert p.ndim == 2, p.ndim
            self._muon_param_ids.add(id(p))

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if id(p) in self._muon_param_ids]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = gram_newton_schulz(g, steps=group["ns_steps"])

                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if id(p) not in self._muon_param_ids]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]
        return Muon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    else:
        assert 0, "optimizer not supported"
