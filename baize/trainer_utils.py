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
    # 先按原始 data_ptr 去重（tie 的 embed/lm_head 是同一张量），再做类型转换；
    # 反过来做的话 .half() 会产生新张量导致去重失效、权重存两份
    seen, deduped = set(), {}
    for k, v in sd.items():
        ptr = v.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            deduped[k] = v
    sd = {k: (v.half() if half and v.is_floating_point() else v).contiguous().cpu()
          for k, v in deduped.items()}
    if not path.endswith(".safetensors"):
        path = os.path.splitext(path)[0] + ".safetensors"
    save_file(sd, path)
    return path


def load_weights(model, path, strict=True):
    """加载 safetensors 权重；兼容传入 .safetensors 或旧 .pth 路径，以及旧版逐专家 MoE 权重。

    strict=True（默认）时，缺失参数、形状不匹配都会直接报错并列出具体参数名，
    避免配置写错时静默加载出一个部分随机初始化的模型。strict=False 时只打印警告。
    返回成功加载的张量数。
    """
    from .model import convert_legacy_moe_state_dict

    if not path.endswith(".safetensors") and not os.path.exists(path):
        st = os.path.splitext(path)[0] + ".safetensors"
        if os.path.exists(st):
            path = st
    if path.endswith(".safetensors"):
        weights = load_file(path)
    else:
        weights = torch.load(path, map_location="cpu")
        weights = {k: v for k, v in weights.items() if torch.is_tensor(v)}
    weights = {k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in weights.items()}
    weights = convert_legacy_moe_state_dict(weights)

    shapes = {k: v.shape for k, v in model.named_parameters()}
    if "lm_head.weight" not in shapes:  # tie_word_embeddings：lm_head 与 embed_tokens 共享
        weights.pop("lm_head.weight", None)
    mismatched = [f"{k}: 文件 {tuple(v.shape)} ≠ 模型 {tuple(shapes[k])}"
                  for k, v in weights.items() if k in shapes and v.shape != shapes[k]]
    missing = sorted(k for k in shapes if k not in weights)
    unexpected = sorted(k for k in weights if k not in shapes)
    problems = []
    if mismatched:
        problems.append("形状不匹配:\n  " + "\n  ".join(mismatched))
    if missing:
        problems.append(f"缺失 {len(missing)} 个参数（将保持随机初始化）:\n  " + "\n  ".join(missing[:20])
                        + ("\n  ..." if len(missing) > 20 else ""))
    if problems:
        msg = f"加载 {path} 时发现问题（多半是 config 与权重不一致，如 vocab_size / hidden_size / 层数）：\n" \
              + "\n".join(problems)
        if strict:
            raise ValueError(msg)
        print(f"[warning] {msg}", flush=True)
    if unexpected:
        print(f"[warning] 忽略权重文件中多余的 {len(unexpected)} 个张量: {unexpected[:5]}", flush=True)

    weights = {k: v for k, v in weights.items() if k in shapes and v.shape == shapes[k]}
    model.load_state_dict(weights, strict=False)
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


def get_lr(current_step, total_steps, lr, warmup_steps=0):
    """线性 warmup + 半余弦退火：warmup 期间从 0 线性升到 lr_max，之后余弦降到 lr_min = 0.1·lr_max。"""
    if warmup_steps > 0 and current_step < warmup_steps:
        return lr * (current_step + 1) / warmup_steps
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def resolve_warmup(warmup_steps, total_steps):
    """warmup_steps 为 None 时自动取总步数的 1%（至少 1 步，最多 2000 步）。"""
    if warmup_steps is not None:
        return max(0, warmup_steps)
    return min(2000, max(1, total_steps // 100))


def init_distributed_mode():
    """torchrun 启动时初始化进程组：有 GPU 用 nccl，否则用 gloo（便于 CPU 上调试多进程）。"""
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        dist.init_process_group(backend="gloo")
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
    if n_routed > 0:
        # 合并存储的专家权重 [E, ...]：每个 MoE 层只有 k/E 的专家参数被激活
        expert_total = sum(p.numel() for n, p in model.named_parameters()
                           if n.endswith((".w_gate", ".w_up", ".w_down"))) / 1e6
        active = total - expert_total + expert_total * n_active / n_routed
        Logger(f"Model Params: {total:.2f}M-A{active:.2f}M")
    else:
        Logger(f"Model Params: {total:.2f}M")
