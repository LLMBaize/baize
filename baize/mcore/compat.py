"""
megatron-core 兼容补丁
======================

1. ``apply_compat_patches()``（正式训练也会调用）
   megatron-core 0.16.1 在没有安装 Transformer Engine 时，``moe_utils`` 引用了未定义的
   ``te_general_gemm``，MoE 路由前向直接 NameError。这里补成 None，走 torch 的 matmul 分支。

2. ``enable_cpu_smoke_mode()``（只用于没有 GPU 的冒烟测试 / CI）
   megatron-core 是为 GPU 写的：attention 的全局缓冲区、RNG tracker、DDP 梯度缓冲区等
   约 50 处写死了 ``torch.cuda.current_device()``。冒烟测试时把这些调用指向 CPU，
   让同一套代码路径能在 CPU + gloo 上跑通前向 / 反向 / 优化器 / 断点。
   **GPU 训练不要调用它**（性能与数值都没有意义）。
"""

import torch

_CPU_MODE = False


def apply_compat_patches():
    try:
        import megatron.core.transformer.moe.moe_utils as moe_utils
    except ImportError:
        return
    if not hasattr(moe_utils, "te_general_gemm"):
        moe_utils.te_general_gemm = None
    # multi_latent_attention 在 TE 缺失时仍把 HAVE_TE 当作真值，随后 isinstance(x, None) 报错
    import megatron.core.transformer.multi_latent_attention as mla

    import importlib.util

    if importlib.util.find_spec("transformer_engine") is None:  # TE 未安装：megatron 会塞入 MagicMock 占位
        mla.HAVE_TE = False


def enable_cpu_smoke_mode():
    """把 megatron-core 里写死的 CUDA 设备 / RNG 调用重定向到 CPU。幂等。"""
    global _CPU_MODE
    if _CPU_MODE:
        return
    if torch.cuda.is_available():
        raise RuntimeError("检测到 GPU：CPU 冒烟模式只用于无 GPU 环境，请去掉 --cpu_smoke")
    torch.cuda.current_device = lambda: torch.device("cpu")
    torch.cuda.get_rng_state = lambda device=None: torch.get_rng_state()
    torch.cuda.set_rng_state = lambda state, device=None: torch.set_rng_state(state)
    torch.cuda.manual_seed = lambda seed: None
    torch.cuda.manual_seed_all = lambda seed: None
    torch.cuda.synchronize = lambda device=None: None

    import megatron.core.tensor_parallel.random as mrandom

    mrandom._get_cuda_rng_state = lambda *a, **k: torch.get_rng_state()
    mrandom._set_cuda_rng_state = lambda state, *a, **k: torch.set_rng_state(state)

    # 下列模块里有 torch.zeros(..., device="cuda") 之类的字面量：给它们换一个把 cuda 映射到 cpu 的 torch 代理
    import importlib

    for name in ("megatron.core.pipeline_parallel.schedules", "megatron.core.optimizer.optimizer",
                 "megatron.core.optimizer.grad_scaler", "megatron.core.optimizer.clip_grads",
                 "megatron.core.optimizer.distrib_optimizer", "megatron.core.distributed.param_and_grad_buffer",
                 "megatron.core.transformer.moe.moe_utils", "megatron.core.utils"):
        importlib.import_module(name).torch = _CpuTorchProxy()

    # clip_grad_by_total_norm_fp32 断言梯度是 torch.cuda.FloatTensor：换成等价的 CPU 实现（范数仍由 megatron 计算）
    import megatron.core.optimizer.optimizer as mopt

    def _clip_cpu(parameters, max_norm, total_norm, use_decoupled_grad=False):
        coeff = max_norm / (total_norm + 1.0e-6)
        if coeff < 1.0:
            for p in parameters:
                g = getattr(p, "decoupled_grad", None) if use_decoupled_grad else p.grad
                if g is not None:
                    g.mul_(coeff)

    mopt.clip_grad_by_total_norm_fp32 = _clip_cpu

    # LanguageModule.setup_embeddings_and_output_layer 在 PP 下对词嵌入做 weight.data.cuda()：CPU 模式下 Tensor.cuda 是恒等函数
    torch.Tensor.cuda = lambda self, *a, **k: self
    _CPU_MODE = True


class _CpuTorchProxy:
    """除了把 device="cuda" 改成 cpu，其余属性全部透传给 torch。"""

    def __getattr__(self, name):
        attr = getattr(torch, name)
        if name in ("zeros", "ones", "empty", "tensor", "full", "arange", "randn", "rand"):
            def wrapped(*args, device=None, **kwargs):
                if device is not None and str(device).startswith("cuda"):
                    device = "cpu"
                return attr(*args, device=device, **kwargs)
            return wrapped
        return attr


def cpu_smoke_mode() -> bool:
    return _CPU_MODE
