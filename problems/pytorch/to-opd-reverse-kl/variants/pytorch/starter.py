"""OPD（datawhalechina/agentic-rl-lab experiments）反向 KL 蒸馏账。

请实现 solve(mode=...) 四个入口（PyTRIO/datasets/swanlab 剥除、rollout 与
teacher 打分全入参）：

    mode="teacher_logprobs"
        all_logprobs + prompt_ids + completion_ids -> completion 区间切片
        （compute_logprobs 返回整段逐 token logprob，从 len(prompt_ids)
        切到结尾；长度 != len(completion_ids) 或含 None -> ValueError）
    mode="datum"
        prompt_ids + completion_ids + old_logprobs + advantages ->
        右移四字段 {input_ids, target_tokens, logprobs, advantages}
        （prompt_loss_len = len(prompt_ids)-1；input = prompt +
        completion[:-1]；target = [0]*prompt_loss_len + completion；
        logprobs/advantages 前段 0.0 占位；四序列等长否则 ValueError；
        target int64、其余 float32）
    mode="step"
        batches（[(prompt_ids, [seq…])]，seq.tokens/seq.logprobs，空序列
        跳过）+ teacher_fn(prompt_ids, completion_ids) + kl_penalty_coef
        -> (datums, reverse_kls, token_counts, metrics)
        （reverse_kl = student_lps − teacher_lps 逐 token 向量差；
        advantages = −coef × reverse_kl；reverse_kls 逐 token extend；
        datums 全空 -> RuntimeError；metrics = mean/std ddof=0）
    mode="schedule"（variant 分发）
        "build"：("medical",)*m + ("ceval",)*c 周期元组
        "resolve"：steps>0 直接用；steps=0 时两类数据 ceil(size/batch)
            步数取 max 周期数乘回周期长度
        "count"：schedule[step % len(schedule)] 逐 step 计数
        "batch"：rows[(task_step*batch_size + offset) % len(rows)] 回绕取批

参考 solution.py 实现你的版本。
"""
import math

import numpy as np


def solve(mode="teacher_logprobs", all_logprobs=None, prompt_ids=None,
          completion_ids=None, old_logprobs=None, advantages=None,
          batches=None, teacher_fn=None, kl_penalty_coef=1.0,
          variant="build", medical_steps_per_cycle=1, ceval_steps_per_cycle=1,
          steps=0, batch_size=2, medical_size=0, ceval_size=0,
          schedule=None, total_steps=0, rows=None, task_step=0):
    """统一入口（判题协议）：按 mode 分发到四个官方账段。"""
    # TODO: 实现四个入口
    raise NotImplementedError
