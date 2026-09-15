#!/usr/bin/env python3
"""
BaiZe 推理 demo —— 命令行与网页两种形态。

命令行：
    python scripts/demo.py --prompt "你好"                    # 单次生成（流式打印）
    python scripts/demo.py --prompt "你好" --loops 16          # 深度外推：推理圈数 > 训练圈数
    python scripts/demo.py --prompt "你好" --compare 2,4,8,16  # 同一 prompt 对比不同圈数（贪心）
    python scripts/demo.py                                     # 交互对话
    python scripts/demo.py --bench                             # 吞吐测试

网页（需 pip install gradio）：
    python scripts/demo.py --web
"""

import argparse
import os
import sys
import threading
import time

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baize import BaiZeConfig, BaiZeForCausalLM, BaiZeTokenizer
from baize.trainer_utils import load_weights


# ─────────────────────────── 流式输出 ───────────────────────────
class TextStreamer:
    """把逐 token 的输出解码成可打印文本。

    ByteLevel BPE 下单个 token 可能是多字节字符的一半，直接 decode 会得到
    替换符（U+FFFD）。这里维护"已打印文本"，只在新增部分完整时才输出，
    半截字符留到下一次拼齐再打。
    """

    def __init__(self, tokenizer, enabled=True):
        self.tokenizer = tokenizer
        self.enabled = enabled
        self.ids = []
        self.printed = ""
        self._skip_next = True  # generate 会先把 prompt 灌进来（HF Streamer 约定），跳过它只留新生成部分

    def put(self, value):
        if self._skip_next:
            self._skip_next = False
            return
        self.ids.extend(value[0].tolist())
        if not self.enabled:
            return
        text = self.tokenizer.decode(self.ids)
        common = 0
        for a, b in zip(text, self.printed):
            if a != b:
                break
            common += 1
        delta = text[common:]
        if delta and not delta.endswith("�"):
            sys.stdout.write(delta)
            sys.stdout.flush()
            self.printed = text

    def end(self):
        pass

    @property
    def text(self):
        return self.tokenizer.decode(self.ids)


# ─────────────────────────── 模型加载 ───────────────────────────
def resolve_dirs(args):
    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        for cand in ("demo_weights", "out"):
            if os.path.isdir(cand) and any(f.endswith(".safetensors") for f in os.listdir(cand)):
                save_dir = cand
                break
        else:
            raise SystemExit(f"没找到权重目录（{args.save_dir}/demo_weights/out），请先训练或用 --save_dir 指定")
    tokenizer_dir = args.tokenizer
    if not os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")):
        tokenizer_dir = save_dir  # 权重目录自带 tokenizer 时直接用它
    return save_dir, tokenizer_dir


def load_model(args):
    save_dir, tokenizer_dir = resolve_dirs(args)
    tokenizer = BaiZeTokenizer.from_pretrained(tokenizer_dir)
    config = BaiZeConfig.from_pretrained(save_dir)
    config.vocab_size = tokenizer.vocab_size

    model = BaiZeForCausalLM(config)
    name = args.weight if args.weight.endswith(".safetensors") else f"{args.weight}.safetensors"
    wp = os.path.join(save_dir, name)
    n_tensors = load_weights(model, wp)
    model = model.to(args.device).eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"权重   : {wp}（{n_tensors} 个张量，{n_params:.1f}M 参数）")
    print(f"词表   : {tokenizer.vocab_size}（{tokenizer_dir}）")
    print(f"循环   : 训练圈数 {config.max_loop_iters}，谱半径 ρ(A) = "
          f"{model.model.recurrent.injection.get_A().max().item():.4f}")
    print(f"设备   : {args.device}")
    return model, tokenizer, config


# ─────────────────────────── 生成 ───────────────────────────
def generate_once(model, tokenizer, prompt, args, loops=None, temperature=None, stream=False):
    """单次生成，返回 (新生成文本, 新 token 数, 耗时秒)。"""
    text = tokenizer.build_chat([{"role": "user", "content": prompt}], add_generation_prompt=True)
    ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=args.device)
    streamer = TextStreamer(tokenizer, enabled=stream)
    t0 = time.time()
    out = model.generate(
        ids,
        max_new_tokens=args.max_new_tokens,
        n_loops=args.loops if loops is None else loops,
        temperature=args.temperature if temperature is None else temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.im_end_id,
        streamer=streamer,
    )
    dt = time.time() - t0
    n_new = out.shape[1] - ids.shape[1]
    new_ids = out[0, ids.shape[1]:]
    return tokenizer.decode(new_ids.tolist()), n_new, dt


def compare_loops(model, tokenizer, prompt, args, loop_values):
    """同一 prompt 在不同循环圈数下的输出对比（贪心解码，排除采样随机性）。"""
    print(f"\n{'=' * 68}\nprompt: {prompt}\n（贪心解码，n_loops 从 {loop_values[0]} 到 {loop_values[-1]}）\n{'=' * 68}")
    for n in loop_values:
        reply, n_new, dt = generate_once(model, tokenizer, prompt, args, loops=n, temperature=0.0)
        print(f"\n[n_loops={n}] {dt:.2f}s / {n_new} tok  ({n_new / max(dt, 1e-6):.1f} tok/s)")
        print(reply.strip())


def benchmark(model, tokenizer, args):
    prompts = [p.strip() for p in args.bench_prompts.split("|") if p.strip()]
    print(f"\n吞吐测试：{args.bench_n} 次生成（每次最多 {args.max_new_tokens} token，"
          f"n_loops={args.loops or 'config'}，贪心）\n")
    total_tok, total_t = 0, 0.0
    for i in range(args.bench_n):
        _, n_new, dt = generate_once(model, tokenizer, prompts[i % len(prompts)], args, temperature=0.0)
        total_tok += n_new
        total_t += dt
        print(f"  #{i + 1:>2}  {dt:5.2f}s  {n_new:>4} tok  {n_new / max(dt, 1e-6):6.1f} tok/s")
    print(f"\n平均: {total_tok / max(total_t, 1e-6):.1f} tok/s"
          f"（{total_tok} tok / {total_t:.2f}s，{args.bench_n} 次）")


def interactive(model, tokenizer, args):
    print("\n进入交互对话：直接输入内容回车；命令 /loops N、/temp X、/reset、exit\n")
    history = []
    while True:
        try:
            line = input("用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("exit", "quit"):
            break
        if line.startswith("/"):
            cmd, _, val = line.partition(" ")
            if cmd == "/loops" and val.strip().isdigit():
                args.loops = int(val)
                print(f"  → 循环圈数 = {args.loops}")
            elif cmd == "/temp":
                try:
                    args.temperature = float(val)
                    print(f"  → temperature = {args.temperature}")
                except ValueError:
                    print("  用法: /temp 0.8")
            elif cmd == "/reset":
                history = []
                print("  → 已清空对话历史")
            else:
                print("  可用命令: /loops N、/temp X、/reset、exit")
            continue

        history.append({"role": "user", "content": line})
        text = tokenizer.build_chat(history, add_generation_prompt=True)
        ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=args.device)
        streamer = TextStreamer(tokenizer)
        t0 = time.time()
        out = model.generate(
            ids, max_new_tokens=args.max_new_tokens, n_loops=args.loops,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.im_end_id, streamer=streamer,
        )
        dt = time.time() - t0
        reply = streamer.text
        n_new = out.shape[1] - ids.shape[1]
        print(f"\n  [{n_new} tok / {dt:.2f}s / {n_new / max(dt, 1e-6):.1f} tok/s]\n")
        history.append({"role": "assistant", "content": reply})


# ─────────────────────────── 网页界面 ───────────────────────────
def run_web(model, tokenizer, args, config):
    try:
        import gradio as gr
    except ImportError:
        raise SystemExit("网页模式需要 gradio：pip install gradio（或直接用命令行模式）")

    def make_chatbot(**kwargs):
        """gradio 6 起 Chatbot 只有 messages 格式，不再接受 type 参数。"""
        try:
            return gr.Chatbot(type="messages", **kwargs)
        except TypeError:
            return gr.Chatbot(**kwargs)

    def respond(message, history, n_loops, temperature, top_k, top_p, rep, max_new):
        history = list(history or [])
        if not message.strip():
            yield history
            return
        history.append({"role": "user", "content": message})
        msgs = [{"role": h["role"], "content": h["content"]} for h in history]
        prompt = tokenizer.build_chat(msgs, add_generation_prompt=True)
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=args.device)
        streamer = TextStreamer(tokenizer, enabled=False)
        error, state = [], {"done": False}

        def run():
            try:
                model.generate(
                    ids, max_new_tokens=int(max_new), n_loops=int(n_loops),
                    temperature=float(temperature), top_k=int(top_k), top_p=float(top_p),
                    repetition_penalty=float(rep), eos_token_id=tokenizer.im_end_id,
                    streamer=streamer,
                )
            except Exception as e:  # noqa: BLE001 —— 线程内异常需带回主线程展示
                error.append(f"[生成失败] {e}")
            finally:
                state["done"] = True

        threading.Thread(target=run, daemon=True).start()
        shown = ""
        while not state["done"]:
            if streamer.text != shown:
                shown = streamer.text
                yield history + [{"role": "assistant", "content": shown}]
            time.sleep(0.05)
        yield history + [{"role": "assistant", "content": streamer.text or (error[0] if error else "")}]

    with gr.Blocks(title="BaiZe 推理 demo") as app:
        gr.Markdown(
            "# BaiZe（白泽）· RDT 推理 demo\n"
            f"循环深度 Transformer：同一套权重循环 **{config.max_loop_iters}** 圈，"
            "推理时可调节圈数（大于训练值即深度外推）。"
        )
        chatbot = make_chatbot(height=420, label="对话")
        with gr.Row():
            msg = gr.Textbox(placeholder="说点什么，回车发送…", show_label=False, scale=5, autofocus=True)
            send = gr.Button("发送", variant="primary", scale=1)
        with gr.Accordion("生成参数", open=False):
            with gr.Row():
                loops = gr.Slider(1, 32, value=config.max_loop_iters, step=1,
                                  label="循环圈数 n_loops（> 训练值为深度外推）")
                temp = gr.Slider(0, 1.5, value=args.temperature, step=0.05, label="temperature（0 = 贪心）")
                topk = gr.Slider(0, 200, value=args.top_k, step=1, label="top_k")
            with gr.Row():
                topp = gr.Slider(0.1, 1.0, value=args.top_p, step=0.01, label="top_p")
                rep = gr.Slider(1.0, 1.5, value=args.repetition_penalty, step=0.01, label="repetition_penalty")
                maxnew = gr.Slider(16, 1024, value=args.max_new_tokens, step=16, label="max_new_tokens")
        clear = gr.Button("清空对话")

        inputs = [msg, chatbot, loops, temp, topk, topp, rep, maxnew]
        msg.submit(respond, inputs, chatbot).then(lambda: "", None, msg)
        send.click(respond, inputs, chatbot).then(lambda: "", None, msg)
        clear.click(lambda: [], None, chatbot)

    # show_api 在 gradio 6 中已移除，按签名探测以兼容 4/5/6
    import inspect

    launch_kwargs = dict(server_name=args.host, server_port=args.port, share=args.share)
    if "show_api" in inspect.signature(app.launch).parameters:
        launch_kwargs["show_api"] = False
    app.launch(**launch_kwargs)


# ─────────────────────────── 入口 ───────────────────────────
def main():
    p = argparse.ArgumentParser(description="BaiZe 推理 demo")
    p.add_argument("--save_dir", type=str, default="demo_weights", help="权重目录")
    p.add_argument("--weight", type=str, default="model", help="权重名（不含 .safetensors）")
    p.add_argument("--tokenizer", type=str, default="tokenizer", help="分词器目录（缺省用权重目录内的）")
    p.add_argument("--prompt", type=str, default=None, help="给定时单次生成，否则进入交互对话")
    p.add_argument("--compare", type=str, default=None, help="圈数对比，如 2,4,8,16")
    p.add_argument("--loops", type=int, default=None, help="循环圈数（默认取 config.max_loop_iters）")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7, help="<=0 为贪心解码（toy 权重建议 0，输出最稳）")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.85)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--no_stream", action="store_true", help="关闭流式打印")
    p.add_argument("--bench", action="store_true", help="吞吐测试")
    p.add_argument("--bench_n", type=int, default=10)
    p.add_argument("--bench_prompts", type=str, default="你好|介绍一下你自己|今天天气怎么样", help="用 | 分隔")
    p.add_argument("--web", action="store_true", help="启动 Gradio 网页界面")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="生成公网分享链接（Gradio）")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model, tokenizer, config = load_model(args)
    print("-" * 68)

    if args.web:
        run_web(model, tokenizer, args, config)
    elif args.compare:
        compare_loops(model, tokenizer, args.prompt or "你好", args,
                      [int(x) for x in args.compare.split(",") if x.strip()])
    elif args.bench:
        benchmark(model, tokenizer, args)
    elif args.prompt:
        print(f"用户: {args.prompt}")
        print("白泽: ", end="", flush=True)
        reply, n_new, dt = generate_once(model, tokenizer, args.prompt, args, stream=not args.no_stream)
        if args.no_stream:
            print(reply)
        print(f"\n  [{n_new} tok / {dt:.2f}s / {n_new / max(dt, 1e-6):.1f} tok/s"
              f"{f' / n_loops={args.loops}' if args.loops else ''}]")
    else:
        interactive(model, tokenizer, args)


if __name__ == "__main__":
    main()
