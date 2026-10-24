"""
BaiZe 的 Megatron-core 版本（可选依赖：pip install megatron-core）。

    from baize.mcore import init_parallel, build_model
    init_parallel(tensor_parallel_size=2)
    model = build_model(BaiZeConfig(...), tensor_parallel_size=2)

训练入口：scripts/pretrain_mcore.py；权重互转：baize/mcore/convert.py。
"""

import os

import torch
import torch.distributed as dist

from .compat import apply_compat_patches, cpu_smoke_mode, enable_cpu_smoke_mode

apply_compat_patches()


def init_parallel(tensor_parallel_size=1, expert_parallel_size=1, pipeline_parallel_size=1, seed=42, cpu=False):
    """初始化进程组与 Megatron 并行状态（torchrun 启动或单进程均可）。返回 local_rank。

    cpu=True：用 gloo 后端并打开 CPU 冒烟模式（仅用于无 GPU 的测试）。
    """
    if cpu:
        enable_cpu_smoke_mode()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        if not cpu:
            torch.cuda.set_device(local_rank)
        dist.init_process_group("gloo" if cpu else "nccl", rank=rank, world_size=world)

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
            expert_model_parallel_size=expert_parallel_size,
        )
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    return local_rank


def build_model(cfg, *, tensor_parallel_size=1, expert_parallel_size=1, pipeline_parallel_size=1, bf16=False,
                n_loops_train=None, grad_checkpoint=False, moe_dispatcher="allgather"):
    """按 BaiZeConfig 构建 Megatron-core 模型（需先 init_parallel）。当前进程所在的流水线段决定构建哪些模块。"""
    from megatron.core import parallel_state

    from .model import BaiZeMCoreModel, build_transformer_configs

    dense_cfg, moe_cfg = build_transformer_configs(
        cfg, tensor_parallel_size=tensor_parallel_size, expert_parallel_size=expert_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size, bf16=bf16, n_loops_train=n_loops_train,
        grad_checkpoint=grad_checkpoint, moe_dispatcher=moe_dispatcher)
    model = BaiZeMCoreModel(cfg, dense_cfg, moe_cfg, pre_process=parallel_state.is_pipeline_first_stage(),
                            post_process=parallel_state.is_pipeline_last_stage())
    if bf16:
        model = model.to(torch.bfloat16)
    if not cpu_smoke_mode():
        model = model.cuda(torch.cuda.current_device())
    return model


__all__ = ["init_parallel", "build_model", "enable_cpu_smoke_mode", "cpu_smoke_mode"]
