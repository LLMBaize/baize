#!/usr/bin/env python3
"""
从训练断点 ckpt_*.pt 导出可直接推理 / SFT / 评测的权重快照（fp16 safetensors + config.json）。

断点每 save_interval 步覆盖一次，只保留最新的；用本脚本把各阶段导出成独立快照：

    # 导出当前断点一次
    python scripts/export_ckpt.py --ckpt out/ckpt_pretrain.pt

    # 监视模式：与训练并行运行，断点每更新一次就导出一份（不影响训练）
    nohup python scripts/export_ckpt.py --ckpt out/ckpt_pretrain.pt --watch > out/export.log 2>&1 &

快照目录：out/snapshots/step_001000/{pretrain.safetensors, config.json}
使用：python scripts/demo.py --save_dir out/snapshots/step_001000 --weight pretrain --tokenizer tokenizer
"""

import argparse
import os
import sys
import time

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baize import BaiZeConfig
from baize.trainer_utils import save_weights


def snapshot_dir(root: str, step: int) -> str:
    return os.path.join(root, f"step_{step:06d}")


def export(ckpt_path: str, out_root: str, name: str) -> int:
    """导出一次，返回断点步数。断点由训练端原子替换写入，读到的总是完整文件。"""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = int(ck["step"])
    d = snapshot_dir(out_root, step)
    if os.path.exists(os.path.join(d, f"{name}.safetensors")):
        return step
    os.makedirs(d, exist_ok=True)
    save_weights(ck["model"], os.path.join(d, f"{name}.safetensors"))
    BaiZeConfig(**ck["config"]).save_pretrained(d)
    print(f"[{time.strftime('%H:%M:%S')}] step {step} → {d}", flush=True)
    return step


def main():
    p = argparse.ArgumentParser(description="从训练断点导出权重快照")
    p.add_argument("--ckpt", default="out/ckpt_pretrain.pt")
    p.add_argument("--out_dir", default=None, help="快照根目录，默认 <ckpt 所在目录>/snapshots")
    p.add_argument("--name", default=None, help="权重文件名，默认取断点名，如 ckpt_pretrain.pt → pretrain")
    p.add_argument("--watch", action="store_true", help="持续监视断点，每次更新都导出")
    p.add_argument("--interval", type=int, default=60, help="监视模式的检查间隔（秒）")
    args = p.parse_args()

    out_root = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "snapshots")
    name = args.name or os.path.basename(args.ckpt).removeprefix("ckpt_").removesuffix(".pt")

    if not args.watch:
        export(args.ckpt, out_root, name)
        return

    print(f"监视 {args.ckpt}，每 {args.interval}s 检查一次；快照写入 {out_root}（Ctrl+C 退出）", flush=True)
    last_mtime = None
    while True:
        try:
            mtime = os.path.getmtime(args.ckpt)
            if mtime != last_mtime:
                export(args.ckpt, out_root, name)
                last_mtime = mtime
        except FileNotFoundError:
            pass  # 训练还没存第一个断点
        except Exception as exc:  # 偶发读取失败（如磁盘忙）下次再试
            print(f"[warning] 导出失败，稍后重试：{type(exc).__name__}: {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
