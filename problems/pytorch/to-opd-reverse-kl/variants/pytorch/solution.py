"""OPD（On-Policy Distillation，datawhalechina/agentic-rl-lab experiments）反向 KL 蒸馏账。

OPD 的核心主张：蒸馏不必让学生去拟合教师的前向分布（forward KL / JSD），
而是**学生先自主采样**、教师对学生实际采出的轨迹逐 token 打 logprob，
reverse KL = student_logprob − teacher_logprob——学生比教师更偏好的 token
惩罚、比教师更不偏好的 token 奖励，再用 −reverse_kl 当 advantage 走
importance_sampling 更新。官方 general-opd/01-demo-sync.py +
medical-opd/05-interleaved-multi-teacher-opd.py 的确定性账（PyTRIO/
datasets/swanlab 剥除，rollout 与 teacher 打分全入参），solve(mode=...)
四个入口：

    mode="teacher_logprobs"（completion_teacher_logprobs）：
        all_logprobs（teacher 对 [prompt+completion] 整段算出的逐 token
        logprob）+ prompt_ids + completion_ids -> completion 区间切片。
        compute_logprobs 返回的是**整段 all_ids 每个 token 的 logprob**，
        因此 completion 区间从 **len(prompt_ids) 开始切到结尾**；随后
        双重校验——长度必须恰好等于 len(completion_ids)、区间内不得有
        None——任一违反 raise ValueError（对齐错位比崩溃更危险的官方
        立场）。

    mode="datum"（build_opd_datum 的右移账）：
        prompt_ids + completion_ids + old_logprobs（student rollout 旧
        策略逐 token logprob）+ advantages -> 右移四字段。
        prompt_loss_len = len(prompt_ids) − 1；**input = prompt +
        completion[:-1]**（最后一个 input 位预测 completion 末 token）；
        **target = [0]×prompt_loss_len + completion**（prompt 区间 0
        占位、从最后一个 prompt token 位开始预测 completion）；logprobs
        与 advantages 同样前段 0.0 占位（prompt 内部不是 OPD 训练目标）；
        四个序列长度严格一致否则 ValueError；数值 dtype 官方是
        target int64 / logprobs+advantages **float32**。

    mode="step"（主循环一个 step 的批账）：
        batches（[(prompt_ids, [rollout 序列…]), …]，序列含 tokens/
        logprobs，空序列跳过不产 datum）+ teacher_fn（对 (prompt_ids,
        completion_ids) 返回逐 token logprob）+ kl_penalty_coef ->
        (datums, reverse_kls, token_counts, metrics)。
        核心 5 行账：student_lps 取自 rollout、teacher_lps 由 teacher_fn
        对同一轨迹给出、**reverse_kl = np.asarray(student_lps) −
        np.asarray(teacher_lps)**（逐 token 向量差，长度不等抛
        ValueError）、**advantages = −kl_penalty_coef × reverse_kl**、
        reverse_kls **逐 token extend**（统计是 token 粒度不是序列
        粒度）；completion_token_counts 记序列长度；全部序列为空 ->
        datums 空 -> **RuntimeError**；metrics = reverse_kls 的
        mean/std（np 默认 ddof=0）。

    mode="schedule"（05-interleaved-multi-teacher 的 IDT 调度账，variant
    分发）：
        variant="build"：medical_steps_per_cycle × "medical" +
        ceval_steps_per_cycle × "ceval" 拼成固定周期元组；
        variant="resolve"：steps > 0 时直接用；steps == 0 时按两类数据
        各自 ceil(size / batch_size) 的步数、**取 max 需要的周期数**、
        乘回周期长度（保证两类数据都至少遍历一次）；
        variant="count"：schedule[step % len(schedule)] 逐 step 计数，
        返回 {"medical": m, "ceval": c}；
        variant="batch"：batch_for_task_step 用**任务自己的 step 下标**
        回绕取批——start = task_step × batch_size，rows[(start + offset)
        % len(rows)]（两类数据互不跳步）。
"""
import math

import numpy as np


def _completion_teacher_logprobs(all_logprobs, prompt_ids, completion_ids):
    """官方 completion_teacher_logprobs：整段切片 + 双重校验。"""
    completion_logprobs = list(all_logprobs)[len(prompt_ids):]
    if len(completion_logprobs) != len(completion_ids) or any(
            v is None for v in completion_logprobs):
        raise ValueError("Invalid teacher logprobs for completion tokens")
    return [float(v) for v in completion_logprobs]


def _build_opd_datum(prompt_ids, completion_ids, old_logprobs, advantages):
    """官方 build_opd_datum：右移四字段 + 等长校验。"""
    prompt_loss_len = len(prompt_ids) - 1
    input_ids = list(prompt_ids) + list(completion_ids)[:-1]
    target_ids = [0] * prompt_loss_len + list(completion_ids)
    padded_logprobs = [0.0] * prompt_loss_len + list(old_logprobs)
    padded_advantages = [0.0] * prompt_loss_len + list(advantages)
    if not (len(input_ids) == len(target_ids) == len(padded_logprobs)
            == len(padded_advantages)):
        raise ValueError("OPD datum fields must have the same length")
    return {
        "input_ids": list(input_ids),
        "target_tokens": np.asarray(target_ids, dtype=np.int64).tolist(),
        "logprobs": np.asarray(padded_logprobs, dtype=np.float32).tolist(),
        "advantages": np.asarray(padded_advantages,
                                 dtype=np.float32).tolist(),
    }


def _opd_step(batches, teacher_fn, kl_penalty_coef):
    """官方主循环一个 step 的确定性批账（rollout/teacher 全入参）。"""
    datums = []
    reverse_kls = []
    completion_token_counts = []
    for prompt_ids, sequences in batches:
        for seq in sequences:
            ids = seq.tokens
            if not ids:
                continue
            student_lps = [float(x) for x in seq.logprobs]
            teacher_lps = teacher_fn(prompt_ids, ids)
            reverse_kl = np.asarray(student_lps) - np.asarray(teacher_lps)
            advantages = -kl_penalty_coef * reverse_kl
            datums.append(_build_opd_datum(prompt_ids, ids, student_lps,
                                           advantages))
            reverse_kls.extend(reverse_kl.tolist())
            completion_token_counts.append(len(ids))
    if not datums:
        raise RuntimeError("No OPD datums were built")
    metrics = {
        "opd/reverse_kl_mean": float(np.mean(reverse_kls)),
        "opd/reverse_kl_std": float(np.std(reverse_kls)),
    }
    return datums, reverse_kls, completion_token_counts, metrics


def _build_task_schedule(medical_steps_per_cycle, ceval_steps_per_cycle):
    """官方 build_task_schedule：固定周期元组。"""
    return ("medical",) * medical_steps_per_cycle \
        + ("ceval",) * ceval_steps_per_cycle


def _resolve_total_steps(steps, batch_size, medical_steps_per_cycle,
                         ceval_steps_per_cycle, medical_size, ceval_size):
    """官方 resolve_total_steps：steps=0 时保证两类数据都至少遍历一次。"""
    if steps > 0:
        return steps
    medical_steps = math.ceil(medical_size / batch_size)
    ceval_steps = math.ceil(ceval_size / batch_size)
    cycles = max(
        math.ceil(medical_steps / medical_steps_per_cycle),
        math.ceil(ceval_steps / ceval_steps_per_cycle),
    )
    return cycles * (medical_steps_per_cycle + ceval_steps_per_cycle)


def _count_scheduled_steps(schedule, total_steps):
    """官方 count_scheduled_steps：schedule[step % len] 逐 step 计数。"""
    counts = {"medical": 0, "ceval": 0}
    for step in range(total_steps):
        counts[schedule[step % len(schedule)]] += 1
    return counts


def _batch_for_task_step(rows, task_step, batch_size):
    """官方 batch_for_task_step：任务自己的 step 下标回绕取批。"""
    start = task_step * batch_size
    return [rows[(start + offset) % len(rows)] for offset in range(batch_size)]


def solve(mode="teacher_logprobs", all_logprobs=None, prompt_ids=None,
          completion_ids=None, old_logprobs=None, advantages=None,
          batches=None, teacher_fn=None, kl_penalty_coef=1.0,
          variant="build", medical_steps_per_cycle=1, ceval_steps_per_cycle=1,
          steps=0, batch_size=2, medical_size=0, ceval_size=0,
          schedule=None, total_steps=0, rows=None, task_step=0):
    """统一入口（判题协议）：按 mode 分发到四个官方账段。"""
    if mode == "teacher_logprobs":
        return _completion_teacher_logprobs(all_logprobs, prompt_ids,
                                            completion_ids)
    if mode == "datum":
        return _build_opd_datum(prompt_ids, completion_ids, old_logprobs,
                                advantages)
    if mode == "step":
        return _opd_step(batches, teacher_fn, kl_penalty_coef)
    if mode == "schedule":
        if variant == "build":
            return list(_build_task_schedule(medical_steps_per_cycle,
                                             ceval_steps_per_cycle))
        if variant == "resolve":
            return _resolve_total_steps(steps, batch_size,
                                        medical_steps_per_cycle,
                                        ceval_steps_per_cycle, medical_size,
                                        ceval_size)
        if variant == "count":
            return _count_scheduled_steps(schedule, total_steps)
        if variant == "batch":
            return _batch_for_task_step(rows, task_step, batch_size)
        raise ValueError("Unknown schedule variant: %s" % variant)
    raise ValueError("Unknown mode: %s" % mode)
