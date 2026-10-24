"""Megatron-core 版冒烟测试（CPU + gloo）。

需要 megatron-core（pip install megatron-core）；没装，或者本机有 GPU（CPU 冒烟模式只用于无 GPU 环境）时自动跳过。
每项检查都在子进程里跑：CPU 冒烟模式会改写 torch.cuda 的若干函数，不能污染其它测试。

运行：python tests/test_mcore.py   （也兼容 pytest）
"""

import argparse
import importlib.util
import os
import re
import socket
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

SKIP_REASON = None
if importlib.util.find_spec("megatron") is None or importlib.util.find_spec("megatron.core") is None:
    SKIP_REASON = "未安装 megatron-core"
elif torch.cuda.is_available():
    SKIP_REASON = "本机有 GPU：CPU 冒烟测试只在无 GPU 环境运行"

ARCH = ["--hidden_size", "64", "--num_attention_heads", "4", "--num_key_value_heads", "2", "--head_dim", "16",
        "--intermediate_size", "128", "--moe_intermediate_size", "32", "--n_experts", "4", "--max_loop_iters", "4",
        "--n_loops_train", "3", "--max_seq_len", "64", "--use_moe", "1"]
MLA_ARCH = ARCH + ["--attn_type", "mla"]


def _common(arch=ARCH):
    return ["--data", os.path.join(ROOT, "data", "corpus.txt"), "--tokenizer", os.path.join(ROOT, "demo_weights"),
            *arch, "--max_steps", "6", "--log_interval", "1", "--save_interval", "100", "--learning_rate", "3e-3",
            "--warmup_steps", "2", "--epochs", "2", "--from_weight", "init"]


COMMON = _common()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


def _run(cmd, env_extra=None):
    env = dict(os.environ, **(env_extra or {}))
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"命令失败：{' '.join(cmd)}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r.stdout


def _losses(text):
    return {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"^step:(\d+)/\d+ loss:([\d.]+)", text, re.M)}


def _init_weights(*dirs, arch=ARCH):
    """同一份随机初始权重（fp32）写进各个 save_dir，原版与 mcore 都用 --from_weight init 加载。"""
    from pretrain import add_data_model_args, config_from_args
    from baize import BaiZeForCausalLM, BaiZeTokenizer
    from baize.trainer_utils import save_weights

    p = argparse.ArgumentParser()
    add_data_model_args(p)
    cfg = config_from_args(p.parse_args(arch), BaiZeTokenizer.from_pretrained(os.path.join(ROOT, "demo_weights")).vocab_size)
    torch.manual_seed(1)
    model = BaiZeForCausalLM(cfg)
    for d in dirs:
        save_weights(model, os.path.join(d, "init.safetensors"), half=False)


def _original_losses(d, arch=ARCH):
    out = _run([sys.executable, "scripts/pretrain.py", *_common(arch), "--batch_size", "4", "--accumulation_steps", "2",
                "--save_dir", d, "--dtype", "float32", "--device", "cpu"])
    return _losses(out)


def _mcore(d, *extra, nproc=1, arch=ARCH):
    if nproc == 1:
        cmd = [sys.executable, "scripts/pretrain.py", "--backend", "megatron"]
        env = {"MASTER_PORT": _free_port()}
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}",
               f"--master_port={_free_port()}", "scripts/pretrain.py", "--backend", "megatron"]
        env = {}
    return _run([*cmd, *_common(arch), "--save_dir", d, "--cpu_smoke", "1", *extra], env)


def _skip():
    if SKIP_REASON:
        print(f"SKIP: {SKIP_REASON}")
        try:
            import pytest
            pytest.skip(SKIP_REASON)
        except ImportError:
            pass
        return True
    return False


def test_model_equivalence_and_conversion():
    if _skip():
        return
    for check in ("equivalence", "tp_convert"):
        out = _run([sys.executable, "tests/mcore_checks.py", check], {"MASTER_PORT": _free_port()})
        assert f"OK {check}" in out


def test_training_matches_original_tp1_tp2_dp2_and_resume():
    """同样的初始权重与数据顺序：mcore 版（TP=1 / TP=2+序列并行 / DP=2 / 中断续训）与原版 Trainer 的 loss 轨迹一致。"""
    if _skip():
        return
    with tempfile.TemporaryDirectory() as tmp:
        dirs = {k: os.path.join(tmp, k) for k in ("orig", "tp1", "tp2", "dp2", "resume")}
        for d in dirs.values():
            os.makedirs(d)
        _init_weights(*dirs.values())
        ref = _original_losses(dirs["orig"])
        assert sorted(ref) == list(range(1, 7))

        runs = {
            "tp1": _losses(_mcore(dirs["tp1"], "--batch_size", "4", "--accumulation_steps", "2")),
            "tp2": _losses(_mcore(dirs["tp2"], "--batch_size", "4", "--accumulation_steps", "2",
                                  "--tensor_parallel_size", "2", nproc=2)),
            "dp2": _losses(_mcore(dirs["dp2"], "--batch_size", "2", "--accumulation_steps", "2", nproc=2)),
        }
        # 第 3 步存断点退出，再续训到第 6 步
        first = _mcore(dirs["resume"], "--batch_size", "4", "--accumulation_steps", "2", "--exit_interval", "3")
        assert "exit_interval" in first
        second = _mcore(dirs["resume"], "--batch_size", "4", "--accumulation_steps", "2", "--from_resume", "1")
        assert "断点续训" in second
        runs["resume"] = {**_losses(first), **_losses(second)}

        for name, losses in runs.items():
            assert sorted(losses) == list(range(1, 7)), (name, losses)
            for step, v in losses.items():
                assert abs(v - ref[step]) < 2e-4, f"{name} 第 {step} 步 loss {v} 与原版 {ref[step]} 不一致"

        # TP=2 训练后导出的原版格式权重：可严格加载进原版模型，且与 TP=1 导出的一致
        from safetensors.torch import load_file
        from baize import BaiZeConfig, BaiZeForCausalLM
        from baize.trainer_utils import load_weights

        a = load_file(os.path.join(dirs["tp1"], "pretrain.safetensors"))
        b = load_file(os.path.join(dirs["tp2"], "pretrain.safetensors"))
        assert set(a) == set(b)
        assert max((a[k].float() - b[k].float()).abs().max().item() for k in a) < 1e-3
        cfg = BaiZeConfig.from_pretrained(dirs["tp2"])
        assert cfg.n_loops_train == 3
        load_weights(BaiZeForCausalLM(cfg), os.path.join(dirs["tp2"], "pretrain.safetensors"))


def test_parallel_layouts_match_original_gqa_and_mla():
    """TP=2、EP=2、PP=2 各自的 loss 轨迹与原版一致；导出权重与 TP=1 一致，且 EP / PP 导出能严格加载进原版模型。
    GQA 与 MLA 两种注意力都跑（MLA 的 kv 下投影在 TP 下需要单独的梯度处理）。"""
    if _skip():
        return
    from safetensors.torch import load_file
    from baize import BaiZeConfig, BaiZeForCausalLM
    from baize.trainer_utils import load_weights

    for label, arch in (("gqa", ARCH), ("mla", MLA_ARCH)):
        with tempfile.TemporaryDirectory() as tmp:
            names = ("orig", "tp1", "tp2", "dp2", "ep2", "pp2")
            dirs = {k: os.path.join(tmp, k) for k in names}
            for d in dirs.values():
                os.makedirs(d)
            _init_weights(*dirs.values(), arch=arch)
            ref = _original_losses(dirs["orig"], arch=arch)
            assert sorted(ref) == list(range(1, 7))

            # 每个布局的 batch 按 DP 切分后与原版一致：TP / PP 的 DP=1 用 batch 4，DP=2 / EP=2（DP=2）用 batch 2
            runs = {
                "tp1": (dict(), "--batch_size", "4"),
                "tp2": (dict(nproc=2), "--tensor_parallel_size", "2", "--batch_size", "4"),
                "dp2": (dict(nproc=2), "--batch_size", "2"),
                "ep2": (dict(nproc=2), "--expert_parallel_size", "2", "--batch_size", "2"),
                "pp2": (dict(nproc=2), "--pipeline_parallel_size", "2", "--batch_size", "4"),
            }
            for name, (kw, *flags) in runs.items():
                losses = _losses(_mcore(dirs[name], "--accumulation_steps", "2", *flags, arch=arch, **kw))
                assert sorted(losses) == list(range(1, 7)), (name, losses)
                for step, v in losses.items():
                    assert abs(v - ref[step]) < 2e-4, f"{label} {name} 第 {step} 步 loss {v} 与原版 {ref[step]} 不一致"

            # 导出权重：与同样 batch / 数据切分的参照一致（TP、PP 对照 TP=1；EP 对照 DP=2）
            for name, ref_name in (("tp2", "tp1"), ("pp2", "tp1"), ("ep2", "dp2")):
                base = load_file(os.path.join(dirs[ref_name], "pretrain.safetensors"))
                got = load_file(os.path.join(dirs[name], "pretrain.safetensors"))
                assert set(got) == set(base)
                diff = max((base[k].float() - got[k].float()).abs().max().item() for k in base)
                assert diff < 1e-3, f"{label} {name} 导出权重与 {ref_name} 相差 {diff}"
            cfg = BaiZeConfig.from_pretrained(dirs["ep2"])
            load_weights(BaiZeForCausalLM(cfg), os.path.join(dirs["ep2"], "pretrain.safetensors"))


def test_pipeline_and_expert_parallel_resume():
    """EP=2 / PP=2 中断续训：每个 (TP, EP, PP) 坐标各自的断点都能读回，续训后的 loss 与不中断的一致。"""
    if _skip():
        return
    with tempfile.TemporaryDirectory() as tmp:
        for flags, nproc, batch in ((("--expert_parallel_size", "2"), 2, "2"),
                                    (("--pipeline_parallel_size", "2"), 2, "4")):
            base = os.path.join(tmp, "_".join(flags).replace("--", ""))
            straight, resume = (os.path.join(base, k) for k in ("straight", "resume"))
            for d in (straight, resume):
                os.makedirs(d)
                _init_weights(d)
            common = ("--accumulation_steps", "2", "--batch_size", batch, *flags)
            ref = _losses(_mcore(straight, *common, nproc=nproc))
            assert sorted(ref) == list(range(1, 7))
            # 同一个 save_dir：第一段存断点退出，第二段从它续训
            a = _losses(_mcore(resume, *common, "--exit_interval", "3", nproc=nproc))
            assert sorted(a) == [1, 2, 3], a
            b = _losses(_mcore(resume, *common, "--from_resume", "1", nproc=nproc))
            assert sorted(b) == [4, 5, 6], b
            resumed = {**a, **b}
            assert sorted(resumed) == list(range(1, 7))
            for step, v in resumed.items():
                assert abs(v - ref[step]) < 1e-5, f"{flags[0]} 续训第 {step} 步 loss {v} 与不中断的 {ref[step]} 不一致"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
