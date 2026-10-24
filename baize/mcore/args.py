"""Megatron-core 后端专用的命令行参数（只依赖 argparse，不导入 megatron，原版后端无需安装 megatron-core）。

与原版共用的训练参数（--save_dir、--batch_size、--save_interval、--seed 等）由 baize.trainer.add_train_args 定义，
这里不重复声明。
"""

import os


def add_megatron_args(parser):
    g = parser.add_argument_group("Megatron-core 后端（--backend megatron）")
    g.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="张量并行度；>1 时自动开启序列并行；要求能整除 num_key_value_heads")
    g.add_argument("--expert_parallel_size", type=int, default=1,
                   help="专家并行度（MoE 专家切分到 EP 个 rank；要求整除 n_experts）")
    g.add_argument("--pipeline_parallel_size", type=int, default=1,
                   help="流水线并行段数：1 或 2（第一段 = 词嵌入+Prelude，第二段 = 循环块+Coda+输出层）")
    g.add_argument("--moe_dispatcher", type=str, default="allgather", choices=["allgather", "alltoall"],
                   help="MoE token 分发方式；alltoall 仅 GPU 可用（CPU 冒烟未通过，见 docs/megatron_core.md）")
    g.add_argument("--use_distributed_optimizer", type=int, default=0, choices=[0, 1],
                   help="优化器状态在 DP 组内切分（ZeRO-1），省显存；仅 GPU")
    g.add_argument("--bf16", type=int, default=1, choices=[0, 1], help="bf16 训练（fp32 主权重）；CPU 冒烟时强制关闭")
    g.add_argument("--exit_interval", type=int, default=None,
                   help="训练到该步数的整数倍时存断点并退出（同 Megatron-LM --exit-interval；用于分段调度 / 测试续训）")
    g.add_argument("--cpu_smoke", type=int, default=0, choices=[0, 1], help="无 GPU 冒烟测试：CPU + gloo")


def check_parallel_args(args, world_size=None):
    """并行方式的合法性检查（在初始化进程组之前给出明确报错）。"""
    tp, pp, ep = args.tensor_parallel_size, args.pipeline_parallel_size, args.expert_parallel_size
    world = int(os.environ.get("WORLD_SIZE", 1)) if world_size is None else world_size
    if pp not in (1, 2):
        raise ValueError("--pipeline_parallel_size 只能是 1 或 2：循环块与 Coda 必须在同一段内完整跑完所有圈")
    if args.num_key_value_heads % tp:
        raise ValueError(f"--tensor_parallel_size {tp} 必须能整除 num_key_value_heads={args.num_key_value_heads}")
    if ep > 1 and args.n_experts % ep:
        raise ValueError(f"--expert_parallel_size {ep} 必须能整除 n_experts={args.n_experts}")
    if world % (tp * pp * ep):
        raise ValueError(f"WORLD_SIZE={world} 必须能被 tp×pp×ep={tp * pp * ep} 整除（余下的为数据并行）")
