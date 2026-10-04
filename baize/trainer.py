"""
BaiZe — 通用训练循环
====================
pretrain.py / sft.py / dpo.py 共用：

    - 单卡 / DDP / FSDP2（--fsdp 1）三种模式，torchrun 启动即多卡；无 GPU 时自动用 gloo（便于调试）
    - 线性 warmup + 余弦退火（--warmup_steps，默认总步数 1%）
    - ResumableSampler：单卡也打乱；断点续训从中断处的下一个 batch 继续，不重复、不遗漏
    - 续训在 torch.compile / DDP / FSDP 包装之前加载权重，避免 _orig_mod. / module. 前缀问题
    - 激活重计算（--grad_checkpoint 1）、ACT 延后启用（--act_start_step N）
    - 梯度累积时非最后一个 micro-batch 跳过梯度同步（DDP no_sync / FSDP set_requires_gradient_sync）
    - 断点文件先写临时文件再原子替换，中途被杀不会损坏旧断点
    - 可选 Weights & Biases 记录（--use_wandb 1）；续训时接着写同一个 run
"""

import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .data import ResumableSampler
from .trainer_utils import (
    Logger, build_scaler, get_lr, init_distributed_mode, is_main_process, resolve_warmup,
    save_weights, setup_seed,
)


def add_train_args(parser, learning_rate=5e-4, epochs=2):
    g = parser.add_argument_group("训练")
    g.add_argument("--save_dir", type=str, default="out")
    g.add_argument("--epochs", type=int, default=epochs)
    g.add_argument("--batch_size", type=int, default=8, help="每卡 micro-batch 大小")
    g.add_argument("--learning_rate", type=float, default=learning_rate)
    g.add_argument("--weight_decay", type=float, default=0.1, help="只作用于 ≥2 维的权重矩阵")
    g.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    g.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    g.add_argument("--num_workers", type=int, default=0, help="数据加载进程数；小数据集用 0 更快")
    g.add_argument("--accumulation_steps", type=int, default=1)
    g.add_argument("--grad_clip", type=float, default=1.0)
    g.add_argument("--log_interval", type=int, default=20)
    g.add_argument("--save_interval", type=int, default=200, help="断点保存间隔（覆盖式，只留最新）")
    g.add_argument("--snapshot_interval", type=int, default=0,
                   help="每 N 步额外保存一份 fp16 权重快照到 <save_dir>/snapshots/step_XXXXXX/，0=不保存")
    g.add_argument("--max_steps", type=int, default=None, help="最多训练多少个优化步（与 epochs 取较小者）")
    g.add_argument("--warmup_steps", type=int, default=None, help="学习率 warmup 步数，默认总步数的 1%%")
    g.add_argument("--from_resume", type=int, default=0, choices=[0, 1], help="从断点续训")
    g.add_argument("--use_compile", type=int, default=0, choices=[0, 1])
    g.add_argument("--grad_checkpoint", type=int, default=0, choices=[0, 1], help="激活重计算（省显存）")
    g.add_argument("--fsdp", type=int, default=0, choices=[0, 1], help="多卡时用 FSDP2 切分参数/梯度/优化器状态")
    g.add_argument("--act_start_step", type=int, default=None,
                   help="前 N 步关闭 ACT（跑满所有圈），之后再启用早停；默认总步数的 10%%，0 = 一开始就启用")
    g.add_argument("--seed", type=int, default=42)
    w = parser.add_argument_group("Weights & Biases")
    w.add_argument("--use_wandb", type=int, default=0, choices=[0, 1], help="把训练指标记录到 wandb（需 pip install wandb）")
    w.add_argument("--wandb_project", type=str, default="baize")
    w.add_argument("--wandb_entity", type=str, default=None, help="团队/用户名，默认用 wandb 登录账号")
    w.add_argument("--wandb_run_name", type=str, default=None, help="默认 <save_weight>-<时间>")
    w.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"],
                   help="offline：只写本地 wandb/ 目录，之后用 wandb sync 上传（无外网时用）")
    return g


class WandbLogger:
    """wandb 的薄封装：只在主进程启用；未安装或初始化失败时打印警告并降级为空操作，不影响训练。"""

    def __init__(self, args, config, save_weight, run_id=None):
        self.run = None
        self.run_id = run_id
        if not getattr(args, "use_wandb", 0) or not is_main_process():
            return
        try:
            import wandb
        except ImportError:
            Logger("[warning] --use_wandb 1 但未安装 wandb（pip install wandb），本次不记录")
            return
        name = args.wandb_run_name or f"{save_weight}-{time.strftime('%m%d-%H%M')}"
        try:
            self.run = wandb.init(
                project=args.wandb_project, entity=args.wandb_entity, name=name, mode=args.wandb_mode,
                id=run_id, resume="allow" if run_id else None, dir=args.save_dir,
                config={"stage": save_weight, "train": vars(args), "model": config.to_dict()},
            )
        except Exception as exc:  # 网络 / 鉴权问题不应中断训练
            Logger(f"[warning] wandb 初始化失败，本次不记录：{type(exc).__name__}: {exc}")
            return
        self.run_id = self.run.id
        Logger(f"wandb: {args.wandb_project}/{name}（id={self.run_id}，mode={args.wandb_mode}）")

    def log(self, metrics: dict, step: int):
        if self.run is not None:
            self.run.log(metrics, step=step)

    def summary(self, **kv):
        if self.run is not None:
            self.run.summary.update(kv)

    def finish(self):
        if self.run is not None:
            self.run.finish()
            self.run = None


class Trainer:
    def __init__(self, args, model, config, save_weight: str):
        self.args = args
        self.config = config
        self.save_weight = save_weight
        self.ckpt_path = os.path.join(args.save_dir, f"ckpt_{save_weight}.pt")

        # ---- 分布式 ----
        local_rank = init_distributed_mode()
        self.distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        if self.distributed:
            args.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        self.device = args.device
        self.use_fsdp = bool(args.fsdp) and self.distributed
        if args.fsdp and not self.distributed:
            Logger("[warning] --fsdp 1 需要 torchrun 多进程启动，单进程下忽略")
        setup_seed(args.seed + self.rank)
        os.makedirs(args.save_dir, exist_ok=True)

        # ---- 精度 ----
        device_type = "cuda" if "cuda" in self.device else "cpu"
        self.dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
        if self.use_fsdp and args.dtype == "float16":
            raise ValueError("FSDP 模式请使用 --dtype bfloat16 或 float32（float16 需要分片 GradScaler）")
        self.autocast = (torch.autocast(device_type=device_type, dtype=self.dtype)
                         if device_type == "cuda" and args.dtype != "float32" else nullcontext())
        self.scaler = build_scaler(enabled=(args.dtype == "float16" and device_type == "cuda"))

        # ---- 模型：先加载断点（裸模型）→ 再包装 ----
        self.raw = model.to(self.device)
        if args.grad_checkpoint:
            self.raw.enable_grad_checkpointing(True)
        self.act_enabled = bool(config.use_act)
        self.step = 0
        resume = None
        self.wandb_id = None
        if args.from_resume and os.path.exists(self.ckpt_path):
            resume = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            self.raw.load_state_dict(resume["model"], strict=False)
            self.step = resume["step"]
            self.wandb_id = resume.get("wandb_id")
            Logger(f"断点续训: {self.ckpt_path} step={self.step}")

        if self.use_fsdp:
            from torch.distributed.fsdp import fully_shard
            for blk in list(self.raw.model.prelude) + list(self.raw.model.coda):
                fully_shard(blk)
            fully_shard(self.raw.model.recurrent)
            fully_shard(self.raw)
            wrapped = self.raw
        elif self.distributed:
            find_unused = (not self.act_enabled) or args.act_start_step != 0  # ACT 头未参与计算时
            wrapped = DistributedDataParallel(
                self.raw, device_ids=[local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=find_unused)
        else:
            wrapped = self.raw
        self.model = torch.compile(wrapped) if args.use_compile else wrapped
        self.ddp = wrapped if isinstance(wrapped, DistributedDataParallel) else None

        # ---- 优化器：≥2 维权重做 weight decay，norm / bias / 1 维参数不做 ----
        decay, no_decay = [], []
        for p in self.raw.parameters():
            if p.requires_grad:
                (decay if p.dim() >= 2 else no_decay).append(p)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=args.learning_rate, betas=(0.9, 0.95))
        if resume is not None and resume.get("mode", "standard") != ("fsdp" if self.use_fsdp else "standard"):
            Logger("[warning] 断点与当前并行模式（FSDP / 非 FSDP）不同，只恢复模型权重与步数，优化器状态重新开始")
            resume["optimizer"] = None
        if resume is not None and resume["optimizer"] is not None:
            if self.use_fsdp:
                from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict
                set_optimizer_state_dict(self.raw, self.optimizer, resume["optimizer"],
                                         options=StateDictOptions(full_state_dict=True))
            else:
                self.optimizer.load_state_dict(resume["optimizer"])
        if resume is not None and resume.get("scaler"):
            self.scaler.load_state_dict(resume["scaler"])
        mode = "FSDP" if self.use_fsdp else ("DDP" if self.distributed else "单进程")
        Logger(f"训练模式: {mode} × {self.world_size}，设备 {self.device}，精度 {args.dtype}"
               f"{'，激活重计算' if args.grad_checkpoint else ''}")

    # ------------------------------------------------------------------
    def _to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(b) for b in batch)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        return batch.to(self.device, non_blocking=True) if torch.is_tensor(batch) else batch

    def _grad_sync_ctx(self, sync: bool):
        if self.ddp is not None and not sync:
            return self.ddp.no_sync()
        if self.use_fsdp:
            self.raw.set_requires_gradient_sync(sync)
        return nullcontext()

    def _spectral_radius(self):
        A = self.raw.model.recurrent.injection.get_A()
        if hasattr(A, "full_tensor"):  # FSDP 下为分片 DTensor（集合通信，所有 rank 都要调用）
            A = A.full_tensor()
        return A.max().item()

    def fit(self, dataset, loss_fn, collate_fn=None):
        """loss_fn(model, batch) -> (loss, logs: dict[str, float|Tensor])"""
        args = self.args
        sampler = ResumableSampler(len(dataset), shuffle=True, seed=args.seed,
                                   rank=self.rank, world_size=self.world_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn,
                            num_workers=args.num_workers, pin_memory="cuda" in self.device, drop_last=True)
        acc = args.accumulation_steps
        micro_per_epoch = sampler.num_samples // args.batch_size
        steps_per_epoch = micro_per_epoch // acc
        if steps_per_epoch == 0:
            raise ValueError(f"数据太少：每卡 {sampler.num_samples} 个样本不足一个优化步 "
                             f"（batch_size={args.batch_size} × accumulation_steps={acc}）")
        total = args.epochs * steps_per_epoch
        if args.max_steps is not None:
            total = min(total, args.max_steps)
        warmup = resolve_warmup(args.warmup_steps, total)
        # ACT 延后启用：前期跑满所有圈，让后面几圈先被训练出来。若一开始就启用，
        # 未训练的后几圈只会引入噪声，停机头会迅速学会在 1~2 圈停下，后几圈拿不到梯度，循环深度坍缩
        act_start = args.act_start_step if args.act_start_step is not None else total // 10
        samples_per_step = args.batch_size * acc * self.world_size
        Logger(f"每 epoch {steps_per_epoch} 步，共 {total} 步，warmup {warmup} 步，"
               f"每步 {samples_per_step} 个样本" + (f"，ACT 自第 {act_start} 步启用" if self.act_enabled else ""))

        self.wandb = WandbLogger(args, self.config, self.save_weight, run_id=self.wandb_id)
        self.wandb_id = self.wandb.run_id
        self.wandb.summary(total_steps=total, samples_per_step=samples_per_step, act_start_step=act_start,
                           params_m=sum(p.numel() for p in self.raw.parameters()) / 1e6)

        start_epoch = self.step // steps_per_epoch
        skip_micro = (self.step % steps_per_epoch) * acc
        start_time, start_step = time.time(), self.step
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        log_acc = {}
        for epoch in range(start_epoch, args.epochs):
            if self.step >= total:
                break
            first = skip_micro if epoch == start_epoch else 0
            sampler.set_epoch(epoch, skip=first * args.batch_size)
            micro = first
            for batch in loader:
                if micro >= steps_per_epoch * acc or self.step >= total:
                    break
                lr = get_lr(self.step, total, args.learning_rate, warmup)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr
                self.raw.config.use_act = self.act_enabled and self.step >= act_start

                last_micro = (micro + 1) % acc == 0
                batch = self._to_device(batch)
                with self._grad_sync_ctx(last_micro):
                    with self.autocast:
                        loss, logs = loss_fn(self.model, batch)
                    self.scaler.scale(loss / acc).backward()
                for k, v in logs.items():
                    log_acc[k] = log_acc.get(k, 0.0) + (v.item() if torch.is_tensor(v) else float(v)) / acc
                micro += 1
                if not last_micro:
                    continue

                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.raw.parameters(), args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1

                if self.step % args.log_interval == 0 or self.step == total:
                    rho = self._spectral_radius()
                    if hasattr(grad_norm, "full_tensor"):
                        grad_norm = grad_norm.full_tensor()
                    if is_main_process():
                        spend = time.time() - start_time
                        eta = spend / max(self.step - start_step, 1) * (total - self.step) / 60
                        loops = self.raw.model.recurrent.avg_loops
                        parts = " ".join(f"{k}:{v:.4f}" for k, v in log_acc.items())
                        Logger(f"step:{self.step}/{total} {parts} lr:{lr:.2e} gnorm:{float(grad_norm):.2f} "
                               f"loops:{float(loops) if loops is not None else 0:.2f} ρ(A):{rho:.3f} "
                               f"eta:{eta:.1f}min")
                        metrics = {f"train/{k}": v for k, v in log_acc.items()}
                        metrics.update({
                            "train/lr": lr, "train/grad_norm": float(grad_norm), "train/rho_A": rho,
                            "train/samples": self.step * samples_per_step,
                            "train/steps_per_sec": (self.step - start_step) / max(spend, 1e-6),
                            "act/enabled": float(self.raw.config.use_act),
                        })
                        if loops is not None:
                            metrics["act/avg_loops"] = float(loops)
                        self.wandb.log(metrics, step=self.step)
                log_acc = {}
                if self.step % args.save_interval == 0:
                    self.save_checkpoint()
                if args.snapshot_interval and self.step % args.snapshot_interval == 0:
                    self.save_snapshot()
        self.save_checkpoint()
        self.save_final()

    # ------------------------------------------------------------------
    def _full_state(self):
        """返回 (model_sd, optim_sd)；FSDP 下为集合通信，只有 rank0 拿到完整字典。"""
        if self.use_fsdp:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
            )
            opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
            return (get_model_state_dict(self.raw, options=opts),
                    get_optimizer_state_dict(self.raw, self.optimizer, options=opts))
        return self.raw.state_dict(), self.optimizer.state_dict()

    def save_checkpoint(self):
        model_sd, optim_sd = self._full_state()
        if is_main_process():
            tmp = self.ckpt_path + ".tmp"
            torch.save({"model": model_sd, "optimizer": optim_sd, "scaler": self.scaler.state_dict(),
                        "step": self.step, "config": self.config.to_dict(),
                        "wandb_id": getattr(self, "wandb_id", None),
                        "mode": "fsdp" if self.use_fsdp else "standard"}, tmp)
            os.replace(tmp, self.ckpt_path)
        if self.distributed:
            dist.barrier()

    def _model_state(self):
        """完整的模型 state dict；FSDP 下为集合通信（所有 rank 都要调用），只有 rank0 拿到内容。"""
        if self.use_fsdp:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
            return get_model_state_dict(self.raw, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        return self.raw

    def _save_config(self, directory):
        current = self.config.use_act
        self.config.use_act = self.act_enabled  # ACT 延后启用期间也按最终设置保存
        self.config.save_pretrained(directory)
        self.config.use_act = current

    def save_snapshot(self):
        """独立的权重快照（不含优化器状态，不会被覆盖），可直接用于推理 / SFT / 评测。"""
        state = self._model_state()
        if is_main_process():
            d = os.path.join(self.args.save_dir, "snapshots", f"step_{self.step:06d}")
            os.makedirs(d, exist_ok=True)
            save_weights(state, os.path.join(d, f"{self.save_weight}.safetensors"))
            self._save_config(d)
            Logger(f"权重快照 → {d}")
        if self.distributed:
            dist.barrier()

    def save_final(self):
        model_sd = self._model_state()
        if is_main_process():
            path = save_weights(model_sd, os.path.join(self.args.save_dir, f"{self.save_weight}.safetensors"))
            self._save_config(self.args.save_dir)
            Logger(f"训练完成，权重保存至 {path}")
        if getattr(self, "wandb", None) is not None:
            self.wandb.finish()
        if self.distributed:
            dist.barrier()
            dist.destroy_process_group()


def finetune_defaults(args, config):
    """SFT / DPO 从预训练权重继续训练时的默认值，返回训练圈数。

    - 圈数沿用预训练（config.n_loops_train），而不是 max_loop_iters：换圈数会改变模型的计算图，
      微调初期 loss 会先被抬高、还白白多花算力。
    - ACT 停机头已在预训练中学好，act_start_step 默认 0（一开始就启用），
      否则前 10% 步关掉 ACT、输出改取最后一圈，等启用时又切回加权输出，分布来回跳。
    """
    n_loops = args.n_loops_train or getattr(config, "n_loops_train", None) or config.max_loop_iters
    if getattr(args, "from_weight", "none") != "none" and args.act_start_step is None:
        args.act_start_step = 0
    Logger(f"训练圈数: {n_loops}（max_loop_iters={config.max_loop_iters}），ACT 自第 {args.act_start_step} 步启用")
    return n_loops


def lm_loss_fn(n_loops=None):
    """预训练 / SFT 的损失：交叉熵 + aux（MoE 均衡 + ACT ponder）。"""

    def fn(model, batch):
        x, y = batch
        out = model(x, labels=y, n_loops=n_loops)
        return out.loss + out.aux_loss, {"loss": out.loss.detach(), "aux": out.aux_loss.detach()}

    return fn
