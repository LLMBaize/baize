"""Megatron-core 训练循环（--backend megatron 时由 scripts/pretrain.py 调用）。

与原版循环（baize/trainer.py）相同的数据、模型参数和训练语义（warmup+余弦、梯度累积、ACT 延后启用、
ponder cost、可续训），训练循环换成 Megatron-core：

    并行      parallel_state（TP × PP × DP，EP 位于 DP 组内），torchrun 启动
    模型      baize/mcore/model.py（TransformerBlock / SelfAttention / MoELayer / VocabParallel*）
    DDP       megatron.core.distributed.DistributedDataParallel（梯度 bucket，可选分布式优化器）
    优化器    megatron.core.optimizer.get_megatron_optimizer（AdamW，梯度裁剪，bf16 主权重）
    学习率    与原版完全一致的线性 warmup + 余弦退火（baize.trainer_utils.get_lr，逐步写入 param_groups）
    前向反向  megatron.core.pipeline_parallel.get_forward_backward_func（micro-batch 梯度累积）

产物：
    <save_dir>/mcore_ckpt_<save_weight>/step_XXXXXXX/tpXX_epXX_ppXX.pt   断点（--from_resume 1 续训）
    <save_dir>/<save_weight>.safetensors + config.json       训练结束导出的原版格式权重，
                                                             可直接用于 eval.py / demo.py / sft.py
"""

import functools
import json
import os
import shutil
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from baize import BaiZeTokenizer
from baize.data import ResumableSampler
from baize.mcore import build_model, cpu_smoke_mode, init_parallel
from baize.mcore.args import check_parallel_args
from baize.mcore.convert import (
    hf_to_mcore, mcore_to_hf, merge_parallel, merge_pp, model_tensors, shard_parallel, split_pp,
)
from baize.pretrain_common import build_dataset, config_from_args
from baize.trainer import WandbLogger
from baize.trainer_utils import Logger, get_lr, is_main_process, resolve_warmup, save_weights


def _ckpt_root(args):
    return os.path.join(args.save_dir, f"mcore_ckpt_{args.save_weight}")


def _rank_tag():
    """本 rank 在 (TP, EP, PP) 网格中的坐标，用作断点文件名。"""
    from megatron.core import parallel_state as ps

    return (f"tp{ps.get_tensor_model_parallel_rank():02d}_ep{ps.get_expert_model_parallel_rank():02d}"
            f"_pp{ps.get_pipeline_model_parallel_rank():02d}")


def _is_saver():
    """每个 (TP, EP, PP) 坐标只让一个 rank 写文件。专家参数只在 expert-DP 组内复制，所以用 expert-DP rank 0；
    dense DP 组包含 EP 维度，用它会漏掉 ep>0 的分片。"""
    from megatron.core import parallel_state as ps

    return ps.get_expert_data_parallel_rank() == 0


def save_checkpoint(args, step, model, optimizer, wandb_id):
    root = _ckpt_root(args)
    d = os.path.join(root, f"step_{step:07d}")
    tag = _rank_tag()
    if is_main_process():
        os.makedirs(d, exist_ok=True)
    dist.barrier()
    if args.use_distributed_optimizer:  # 集合通信：参数状态汇总到 DP rank 0
        optimizer.save_parameter_state(os.path.join(d, f"{tag}_distopt.pt"))
    if _is_saver():
        state = {"model": model_tensors(model.module), "optimizer": optimizer.state_dict(),
                 "step": step, "wandb_id": wandb_id}
        tmp = os.path.join(d, f"{tag}.pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, os.path.join(d, f"{tag}.pt"))
    dist.barrier()
    if is_main_process():
        tmp = os.path.join(root, "latest.json.tmp")
        with open(tmp, "w") as f:
            json.dump({"step": step, "dir": os.path.basename(d), "tp": args.tensor_parallel_size,
                       "ep": args.expert_parallel_size, "pp": args.pipeline_parallel_size}, f)
        os.replace(tmp, os.path.join(root, "latest.json"))
        for old in os.listdir(root):  # 新断点落盘后再删旧的
            if old.startswith("step_") and old != os.path.basename(d):
                shutil.rmtree(os.path.join(root, old), ignore_errors=True)
    dist.barrier()


def load_checkpoint(args, model, optimizer):
    """返回 (step, wandb_id)；没有断点时返回 (0, None)。"""
    latest = os.path.join(_ckpt_root(args), "latest.json")
    if not os.path.exists(latest):
        Logger(f"[warning] 未找到断点 {latest}，从头训练")
        return 0, None
    meta = json.load(open(latest))
    want = {"tp": args.tensor_parallel_size, "ep": args.expert_parallel_size, "pp": args.pipeline_parallel_size}
    if {k: meta.get(k, 1) for k in want} != want:
        raise ValueError(f"断点的并行方式 {({k: meta.get(k, 1) for k in want})} 与当前参数 {want} 不同")
    d = os.path.join(_ckpt_root(args), meta["dir"])
    tag = _rank_tag()
    state = torch.load(os.path.join(d, f"{tag}.pt"), map_location="cpu", weights_only=False)
    _load_model_tensors(model.module, state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if args.use_distributed_optimizer:  # 与 Megatron-LM 相同的顺序：先 state_dict，再参数状态
        optimizer.load_parameter_state(os.path.join(d, f"{tag}_distopt.pt"))
    # 注意不要调用 optimizer.reload_model_params()：bf16 时它会用 bf16 模型参数覆盖刚恢复的 fp32 主权重
    Logger(f"断点续训: {d} step={state['step']}")
    return state["step"], state.get("wandb_id")


def _load_model_tensors(module, sd):
    res = module.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if not k.endswith("_extra_state")]
    if missing or res.unexpected_keys:
        raise RuntimeError(f"权重不匹配：缺少 {missing[:8]}，多余 {res.unexpected_keys[:8]}")


def load_hf_weights(args, cfg, module):
    """从原版 safetensors 初始化（例如继续训练原版预训练权重），按 PP / TP / EP 切分后加载。"""
    from megatron.core import parallel_state as ps
    from safetensors.torch import load_file

    path = os.path.join(args.save_dir, f"{args.from_weight}.safetensors")
    full = hf_to_mcore(load_file(path), cfg, vocab_padded=module.vocab_padded)
    piece = split_pp(full, cfg, ps.get_pipeline_model_parallel_rank(), args.pipeline_parallel_size)
    piece = shard_parallel(piece, cfg, ps.get_tensor_model_parallel_rank(), args.tensor_parallel_size,
                           ps.get_expert_model_parallel_rank(), args.expert_parallel_size)
    _load_model_tensors(module, piece)
    Logger(f"从 {path} 加载原版权重并切分为 PP={args.pipeline_parallel_size} TP={args.tensor_parallel_size} "
           f"EP={args.expert_parallel_size}")


def export_hf(args, cfg, module):
    """收集所有 (PP, TP, EP) 分片 → 原版格式 safetensors + config.json（rank 0 写）。"""
    from megatron.core import parallel_state as ps

    payload = None
    if _is_saver():  # 每个 (PP,TP,EP) 位置只取一份
        payload = ((ps.get_pipeline_model_parallel_rank(), ps.get_tensor_model_parallel_rank(),
                    ps.get_expert_model_parallel_rank()), model_tensors(module))
    gathered = [None] * dist.get_world_size() if is_main_process() else None
    dist.gather_object(payload, gathered, dst=0)
    if is_main_process():
        grid = {}
        for item in gathered:
            if item is not None:
                (pp, tp, ep), sd = item
                grid.setdefault(pp, {}).setdefault(tp, {})[ep] = sd
        stages = []
        for pp in sorted(grid):
            tps = [grid[pp][t] for t in sorted(grid[pp])]
            stages.append(merge_parallel([[row[e] for e in sorted(row)] for row in tps], cfg))
        hf = mcore_to_hf(merge_pp(stages, cfg), cfg)
        path = save_weights(hf, os.path.join(args.save_dir, f"{args.save_weight}.safetensors"))
        cfg.save_pretrained(args.save_dir)
        Logger(f"训练完成，原版格式权重导出至 {path}")
    dist.barrier()


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------


def run_megatron(args):
    if args.cpu_smoke:
        args.bf16 = 0
    check_parallel_args(args)

    init_parallel(args.tensor_parallel_size, args.expert_parallel_size, args.pipeline_parallel_size,
                  seed=args.seed, cpu=bool(args.cpu_smoke))
    from megatron.core import parallel_state as ps
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.pipeline_parallel import get_forward_backward_func

    os.makedirs(args.save_dir, exist_ok=True)
    tokenizer = BaiZeTokenizer.from_pretrained(args.tokenizer)
    cfg = config_from_args(args, tokenizer.vocab_size)
    n_loops = cfg.n_loops_train
    device = torch.device("cpu") if cpu_smoke_mode() else torch.device("cuda", torch.cuda.current_device())

    # ---- 模型 → Megatron DDP ----
    model = build_model(cfg, tensor_parallel_size=args.tensor_parallel_size,
                        expert_parallel_size=args.expert_parallel_size,
                        pipeline_parallel_size=args.pipeline_parallel_size, bf16=bool(args.bf16),
                        n_loops_train=n_loops, grad_checkpoint=bool(args.grad_checkpoint),
                        moe_dispatcher=args.moe_dispatcher)
    model.enable_grad_checkpointing(bool(args.grad_checkpoint))
    if args.from_weight != "none" and not args.from_resume:
        load_hf_weights(args, cfg, model)
    n_params = sum(p.numel() for p in model.parameters())
    Logger(f"Model Params (TP 分片 0): {n_params / 1e6:.2f}M | TP={ps.get_tensor_model_parallel_world_size()} "
           f"DP={ps.get_data_parallel_world_size()} | loops={cfg.max_loop_iters}(train {n_loops}) "
           f"prelude={cfg.prelude_layers} coda={cfg.coda_layers} moe={cfg.use_moe} act={cfg.use_act}")
    ddp_config = DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False,
                                               use_distributed_optimizer=bool(args.use_distributed_optimizer))
    model = DistributedDataParallel(config=model.config, ddp_config=ddp_config, module=model)
    model.broadcast_params()
    unwrapped = model.module

    # ---- 数据 ----
    dataset = build_dataset(args, tokenizer)
    dp_rank, dp_size = ps.get_data_parallel_rank(), ps.get_data_parallel_world_size()
    sampler = ResumableSampler(len(dataset), shuffle=True, seed=args.seed, rank=dp_rank, world_size=dp_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                        drop_last=True, pin_memory=not cpu_smoke_mode())
    acc = args.accumulation_steps
    steps_per_epoch = sampler.num_samples // (args.batch_size * acc)
    if steps_per_epoch == 0:
        raise ValueError(f"数据太少：每个 DP rank {sampler.num_samples} 个样本不足一个优化步")
    total = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total = min(total, args.max_steps)
    warmup = resolve_warmup(args.warmup_steps, total)
    act_start = args.act_start_step if args.act_start_step is not None else total // 10

    # ---- 优化器 ----
    opt_cfg = OptimizerConfig(optimizer="adam", lr=args.learning_rate, min_lr=args.learning_rate * 0.1,
                              weight_decay=args.weight_decay, adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-8,
                              clip_grad=args.grad_clip, bf16=bool(args.bf16),
                              params_dtype=torch.bfloat16 if args.bf16 else torch.float32,
                              use_distributed_optimizer=bool(args.use_distributed_optimizer))
    optimizer = get_megatron_optimizer(opt_cfg, [model])

    # schedules 通过 model.config 拿这些钩子（与 Megatron-LM training.py 的做法一致）
    mcfg = unwrapped.config
    mcfg.no_sync_func = model.no_sync
    mcfg.finalize_model_grads_func = finalize_model_grads
    mcfg.grad_scale_func = optimizer.scale_loss

    step, wandb_id = (load_checkpoint(args, model, optimizer) if args.from_resume else (0, None))
    tracker = WandbLogger(args, cfg, args.save_weight, run_id=wandb_id)
    Logger(f"每 epoch {steps_per_epoch} 步，共 {total} 步，warmup {warmup} 步，每步 "
           f"{args.batch_size * acc * dp_size} 个样本，ACT 自第 {act_start} 步启用")

    def loss_func(loss_mask, output_tensor):
        ce = (output_tensor * loss_mask).sum() / loss_mask.sum().clamp(min=1)
        loss = ce
        ponder = unwrapped.recurrent.ponder_cost
        if cfg.use_act and cfg.act_ponder_coef > 0:
            loss = loss + cfg.act_ponder_coef * ponder
        # 日志值要 clone：ACT 关闭时 loss 与 ce 是同一个张量，schedule 会对它就地做 /= num_microbatches
        return loss, {"loss": ce.detach().clone(), "ponder": unwrapped.recurrent.ponder_log}

    def forward_step(data_iterator, model_chunk):
        x, y = next(data_iterator)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if not unwrapped.post_process:  # 第一段：输出隐状态交给下一段，loss 由最后一段计算
            hidden = model_chunk(x, labels=y, n_loops=n_loops)
            return hidden, functools.partial(loss_func, None)
        token_loss, loss_mask = model_chunk(x, labels=y, n_loops=n_loops)
        return token_loss, functools.partial(loss_func, loss_mask.float())

    forward_backward = get_forward_backward_func()
    act_enabled = cfg.use_act
    start_epoch = step // steps_per_epoch
    skip_micro = (step % steps_per_epoch) * acc
    t0, step0 = time.time(), step
    model.train()
    for epoch in range(start_epoch, args.epochs):
        if step >= total:
            break
        first = skip_micro if epoch == start_epoch else 0
        sampler.set_epoch(epoch, skip=first * args.batch_size)
        it = iter(loader)
        for _ in range((steps_per_epoch * acc - first) // acc):
            if step >= total:
                break
            cfg.use_act = act_enabled and step >= act_start
            lr = get_lr(step, total, args.learning_rate, warmup)
            for group in optimizer.param_groups:  # 与 OptimizerParamScheduler 相同的约定：lr_mult / wd_mult
                group["lr"] = lr * group.get("lr_mult", 1.0)
                group["weight_decay"] = args.weight_decay * group.get("wd_mult", 1.0)
            model.zero_grad_buffer()
            optimizer.zero_grad()
            micro = [next(it) for _ in range(acc)]
            logs = forward_backward(forward_step_func=forward_step, data_iterator=iter(micro), model=[model],
                                    num_microbatches=acc, seq_length=args.max_seq_len,
                                    micro_batch_size=args.batch_size, forward_only=False)
            ok, grad_norm, _ = optimizer.step()
            step += 1

            if step % args.log_interval == 0 or step == total:
                # 统计量只在最后一段产生：先在 DP 组内平均，再广播到整个流水线组（使第 0 号 rank 也能打印、记录）
                stats = torch.zeros(4, device=device)  # loss, ponder, rho(A), avg_loops
                if logs:
                    pair = torch.tensor([sum(float(d[k]) for d in logs) / len(logs) for k in ("loss", "ponder")],
                                        device=device)
                    dist.all_reduce(pair, group=ps.get_data_parallel_group())
                    stats[:2] = pair / dp_size
                    stats[2] = unwrapped.recurrent.injection.get_A().max().item()
                    stats[3] = float(unwrapped.recurrent.avg_loops)
                if args.pipeline_parallel_size > 1:
                    dist.broadcast(stats, src=ps.get_pipeline_model_parallel_last_rank(),
                                   group=ps.get_pipeline_model_parallel_group())
                vals, rho, loops = stats[:2], float(stats[2]), float(stats[3])
                elapsed = time.time() - t0
                eta = elapsed / max(step - step0, 1) * (total - step) / 60
                gn = float(grad_norm) if grad_norm is not None else float("nan")
                Logger(f"step:{step}/{total} loss:{vals[0]:.4f} ponder:{vals[1]:.3f} lr:{lr:.2e} gnorm:{gn:.2f} "
                       f"loops:{loops:.2f} ρ(A):{rho:.3f} eta:{eta:.1f}min" + ("" if ok else " [skipped]"))
                tracker.log({"train/loss": float(vals[0]), "train/ponder": float(vals[1]), "train/lr": lr,
                             "train/grad_norm": gn, "train/rho_A": rho, "act/avg_loops": loops,
                             "act/enabled": float(cfg.use_act),
                             "train/samples": step * args.batch_size * acc * dp_size,
                             "train/steps_per_sec": (step - step0) / max(elapsed, 1e-6)}, step=step)
            if step % args.save_interval == 0 and step < total:  # 最后一步由循环后的 save_checkpoint 统一保存
                save_checkpoint(args, step, model, optimizer, tracker.run_id)
            if args.exit_interval and step % args.exit_interval == 0 and step < total:
                if step % args.save_interval:
                    save_checkpoint(args, step, model, optimizer, tracker.run_id)
                Logger(f"到达 --exit_interval，已在第 {step} 步存断点并退出（用 --from_resume 1 继续）")
                tracker.finish()
                dist.barrier()
                dist.destroy_process_group()
                return

    cfg.use_act = act_enabled
    save_checkpoint(args, step, model, optimizer, tracker.run_id)
    export_hf(args, cfg, unwrapped)
    tracker.finish()
    dist.barrier()
    dist.destroy_process_group()
