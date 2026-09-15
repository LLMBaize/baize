"""训练工具函数：LR 调度、DDP 初始化、日志、权重 IO。"""

import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file


def save_weights(model, path, half=True):
    """以 safetensors 格式保存模型权重（不含优化器状态）。

    half=True 时权重以 fp16 落盘（训练权重为 fp32 主精度时缩小一半体积）。
    tie_word_embeddings 场景下 embed_tokens 与 lm_head 指向同一对象；
    safetensors 不允许重复 data_ptr，这里去重后只保存一份，加载时由
    load_weights 通过 strict=False + model.tie_weights() 恢复共享关系。
    """
    sd = model.state_dict() if hasattr(model, "state_dict") else model
    if half:
        sd = {k: v.half() for k, v in sd.items()}
    sd = {k: v.contiguous().cpu() for k, v in sd.items()}
    # 去重：相同 data_ptr 的张量只保留第一个 key
    seen, deduped = {}, {}
    for k, v in sd.items():
        ptr = v.data_ptr()
        if ptr not in seen:
            seen[ptr] = k
            deduped[k] = v
    sd = deduped
    if not path.endswith(".safetensors"):
        path = os.path.splitext(path)[0] + ".safetensors"
    save_file(sd, path)
    return path


def load_weights(model, path, strict=False):
    """加载 safetensors 权重；兼容传入 .safetensors 或旧 .pth 路径。

    旧 .pth 的 state_dict 中可能混入非张量项，这里只保留张量且形状匹配的项。
    """
    if not path.endswith(".safetensors") and not os.path.exists(path):
        st = os.path.splitext(path)[0] + ".safetensors"
        if os.path.exists(st):
            path = st
    if path.endswith(".safetensors"):
        weights = load_file(path)
    else:
        weights = torch.load(path, map_location="cpu")
        weights = {k: v for k, v in weights.items() if torch.is_tensor(v)}
    shapes = {k: v.shape for k, v in model.named_parameters()}
    weights = {k: v for k, v in weights.items() if k in shapes and v.shape == shapes[k]}
    model.load_state_dict(weights, strict=strict)
    return len(weights)


def build_scaler(enabled: bool):
    """GradScaler 构造（兼容新旧 torch：torch.amp 接口自 2.3 起，旧版回退到 torch.cuda.amp）。"""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content, flush=True)


def get_lr(current_step, total_steps, lr):
    """半余弦退火：lr_min = 0.1·lr_max。"""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_model_params(model):
    """打印总参数量与（MoE 时）激活参数量。"""
    total = sum(p.numel() for p in model.parameters()) / 1e6
    cfg = model.config
    n_routed = getattr(cfg, "n_experts", 0)
    n_active = getattr(cfg, "n_experts_per_tok", 0)
    if getattr(cfg, "use_moe", False) or n_routed > 0:
        expert = sum(p.numel() for n, p in model.named_parameters() if ".experts.0." in n) / 1e6
        base = total - expert * n_routed
        active = base + expert * n_active
        Logger(f"Model Params: {total:.2f}M-A{active:.2f}M")
    else:
        Logger(f"Model Params: {total:.2f}M")
