"""训练循环测试：warmup、采样器、断点续训一致性、DDP / FSDP 多进程（CPU gloo）。

运行：python tests/test_trainer.py   （也兼容 pytest）
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from baize import BaiZeConfig, BaiZeForCausalLM
from baize.data import ResumableSampler, TokenBinDataset
from baize.trainer import Trainer, add_train_args, lm_loss_fn
from baize.trainer_utils import get_lr


class RandomTokens(Dataset):
    def __init__(self, n=48, seq_len=16, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randint(5, 100, (n, seq_len), generator=g)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.data[i].clone()


def tiny_config():
    return BaiZeConfig(vocab_size=100, hidden_size=32, num_attention_heads=2, num_key_value_heads=1,
                       head_dim=16, max_loop_iters=3, intermediate_size=64, moe_intermediate_size=32,
                       n_experts=4, prelude_layers=1, coda_layers=1, max_position_embeddings=128)


def make_args(save_dir, *extra):
    p = argparse.ArgumentParser()
    add_train_args(p)
    return p.parse_args(["--save_dir", save_dir, "--epochs", "2", "--batch_size", "4",
                         "--accumulation_steps", "2", "--dtype", "float32", "--log_interval", "100",
                         "--save_interval", "4", "--device", "cpu", *extra])


def test_warmup_schedule():
    assert get_lr(0, 100, 1.0, warmup_steps=10) == 0.1
    assert abs(get_lr(9, 100, 1.0, warmup_steps=10) - 1.0) < 1e-9
    assert abs(get_lr(10, 100, 1.0, warmup_steps=10) - 1.0) < 1e-9
    assert abs(get_lr(100, 100, 1.0, warmup_steps=10) - 0.1) < 1e-9


def test_sampler_shuffles_and_skips():
    s = ResumableSampler(20, seed=3)
    s.set_epoch(0)
    full = list(s)
    assert sorted(full) == list(range(20)) and full != list(range(20))
    s.set_epoch(0, skip=8)
    assert list(s) == full[8:]
    s.set_epoch(1)
    assert list(s) != full  # 每个 epoch 顺序不同


def _final_params(save_dir):
    from safetensors.torch import load_file
    return load_file(os.path.join(save_dir, "pretrain.safetensors"))


def test_resume_matches_uninterrupted():
    ds = RandomTokens()
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        torch.manual_seed(0)
        Trainer(make_args(d1), BaiZeForCausalLM(tiny_config()), tiny_config(), "pretrain") \
            .fit(ds, lm_loss_fn())
        ref = _final_params(d1)

        # 第二次：训练到第 9 步时"崩溃"（第 8 步已存断点），然后续训
        torch.manual_seed(0)
        crash_args = make_args(d2)
        trainer = Trainer(crash_args, BaiZeForCausalLM(tiny_config()), tiny_config(), "pretrain")
        base_fn = lm_loss_fn()

        def crashing_fn(model, batch):
            if trainer.step == 9:
                raise RuntimeError("simulated crash")
            return base_fn(model, batch)
        try:
            trainer.fit(ds, crashing_fn)
            raise AssertionError("应当崩溃")
        except RuntimeError:
            pass
        ckpt = torch.load(os.path.join(d2, "ckpt_pretrain.pt"), weights_only=False)
        assert ckpt["step"] == 8

        torch.manual_seed(123)  # 新模型的初始化无关紧要：会被断点覆盖
        Trainer(make_args(d2, "--from_resume", "1"), BaiZeForCausalLM(tiny_config()), tiny_config(),
                "pretrain").fit(ds, lm_loss_fn())
        out = _final_params(d2)
        assert ref.keys() == out.keys()
        for k in ref:
            assert torch.allclose(ref[k].float(), out[k].float(), atol=1e-3), k


class _FakeWandb:
    """替代 wandb 模块：记录 init 参数与每次 log，不联网。"""

    def __init__(self):
        self.inits, self.logs, self.finished = [], [], 0

    def init(self, **kw):
        self.inits.append(kw)
        fake = self

        class Run:
            id = kw.get("id") or "run123"
            summary = {}

            def log(self, metrics, step):
                fake.logs.append((step, dict(metrics)))

            def finish(self):
                fake.finished += 1
        return Run()


def test_wandb_logging_and_resume_same_run():
    fake = _FakeWandb()
    sys.modules["wandb"] = fake
    try:
        ds = RandomTokens()
        with tempfile.TemporaryDirectory() as d:
            extra = ("--use_wandb", "1", "--log_interval", "2", "--max_steps", "4")
            Trainer(make_args(d, *extra), BaiZeForCausalLM(tiny_config()), tiny_config(), "pretrain") \
                .fit(ds, lm_loss_fn())
            assert fake.inits[0]["project"] == "baize" and fake.inits[0]["id"] is None
            assert fake.inits[0]["config"]["model"]["hidden_size"] == 32
            assert [s for s, _ in fake.logs] == [2, 4]
            keys = set(fake.logs[0][1])
            assert {"train/loss", "train/lr", "train/grad_norm", "train/rho_A", "act/avg_loops"} <= keys, keys
            assert fake.finished == 1
            assert torch.load(os.path.join(d, "ckpt_pretrain.pt"), weights_only=False)["wandb_id"] == "run123"

            # 续训：接着写同一个 run，step 从断点继续
            Trainer(make_args(d, "--use_wandb", "1", "--log_interval", "2", "--max_steps", "6",
                              "--from_resume", "1"),
                    BaiZeForCausalLM(tiny_config()), tiny_config(), "pretrain").fit(ds, lm_loss_fn())
            assert fake.inits[1]["id"] == "run123" and fake.inits[1]["resume"] == "allow"
            assert [s for s, _ in fake.logs[2:]] == [6]
    finally:
        del sys.modules["wandb"]


def test_wandb_missing_does_not_break_training():
    sys.modules["wandb"] = None  # import wandb → ImportError
    try:
        with tempfile.TemporaryDirectory() as d:
            Trainer(make_args(d, "--use_wandb", "1", "--max_steps", "2"), BaiZeForCausalLM(tiny_config()),
                    tiny_config(), "pretrain").fit(RandomTokens(), lm_loss_fn())
            assert os.path.exists(os.path.join(d, "pretrain.safetensors"))
    finally:
        del sys.modules["wandb"]


def test_token_bin_dataset_caps_and_alignment():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.bin")
        np.arange(100, dtype=np.uint16).tofile(path)
        with open(os.path.join(d, "x.meta.json"), "w") as f:
            json.dump({"dtype": "uint16", "n_tokens": 100, "vocab_size": 703}, f)
        ds = TokenBinDataset([path], seq_len=10, max_tokens=55, vocab_size=703)
        assert len(ds) == 5 and ds.n_tokens == 55
        x, y = ds[2]
        assert x.tolist() == list(range(20, 30)) and torch.equal(x, y)
        try:
            TokenBinDataset([path], seq_len=10, vocab_size=6400)
            raise AssertionError("词表不一致应报错")
        except ValueError:
            pass


def test_dpo_loss_starts_at_log2():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import copy
    import math
    from dpo import dpo_collate, dpo_loss_fn
    torch.manual_seed(0)
    model = BaiZeForCausalLM(tiny_config())
    ref = copy.deepcopy(model).eval()
    batch = dpo_collate([([1, 5, 6, 7], [-100, -100, 6, 7], [1, 5, 8], [-100, -100, 8])])
    assert batch[0].shape == (2, 4) and batch[1][1].tolist() == [-100, -100, 8, -100]
    model.eval()  # 策略 = 参考 → margin 为 0 → loss = log 2
    loss, logs = dpo_loss_fn(ref, beta=0.1)(model, batch)
    assert abs(logs["dpo"].item() - math.log(2)) < 1e-5 and abs(logs["margin"].item()) < 1e-6


DIST_SCRIPT = r'''
import os, sys, torch
sys.path.insert(0, {root!r})
sys.path.insert(0, os.path.join({root!r}, "tests"))
from test_trainer import RandomTokens, tiny_config, make_args
from baize import BaiZeForCausalLM
from baize.trainer import Trainer, lm_loss_fn
torch.manual_seed(0)
extra = sys.argv[2:]
Trainer(make_args(sys.argv[1], *extra), BaiZeForCausalLM(tiny_config()), tiny_config(), "pretrain") \
    .fit(RandomTokens(), lm_loss_fn())
'''


def _torchrun(save_dir, *extra):
    script = os.path.join(save_dir, "run.py")
    with open(script, "w") as f:
        f.write(DIST_SCRIPT.format(root=ROOT))
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
           "--master_addr=127.0.0.1", f"--master_port={port}", script, save_dir, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def test_ddp_two_processes():
    with tempfile.TemporaryDirectory() as d:
        out = _torchrun(d)
        assert "DDP × 2" in out and os.path.exists(os.path.join(d, "pretrain.safetensors"))


def test_fsdp_two_processes_and_resume():
    with tempfile.TemporaryDirectory() as d:
        out = _torchrun(d, "--fsdp", "1", "--grad_checkpoint", "1", "--max_steps", "4")
        assert "FSDP × 2" in out
        ckpt = torch.load(os.path.join(d, "ckpt_pretrain.pt"), weights_only=False)
        assert ckpt["step"] == 4 and ckpt["mode"] == "fsdp"
        out = _torchrun(d, "--fsdp", "1", "--from_resume", "1")
        assert "断点续训" in out and "step=4" in out
        # FSDP 存下的最终权重可被单进程模型严格加载
        from baize.trainer_utils import load_weights
        load_weights(BaiZeForCausalLM(tiny_config()), os.path.join(d, "pretrain.safetensors"))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
