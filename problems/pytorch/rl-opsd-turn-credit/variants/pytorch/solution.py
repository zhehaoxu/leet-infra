"""AgentOPSD 的 turn-level credit 账。

回源: datawhalechina/agentic-rl-lab experiments/agentopsd
（advantages.py 全文件 + teacher.py::_score_turns + loss.py::build_agentopsd_ppo_datum），
论文 arXiv:2608.05987（清华/浙大/美团）。pytrio/Trajectory/Step 依赖全部入参化：
步级 teacher/student logprob、action token 区间、终局奖励均为显式入参，
只保留纯 numpy 确定性账。全部返回值与官方字段一一对应。

核心机制（论文口径）:
- token 级 teacher-student logprob gap 沿 turn（两次环境观测之间的一段动作）
  边界求和得 turn 级 evidence e_k；
- 受贝叶斯启发的信念递推: c_k = gamma*c_{k-1} + e_k（c_0=0），
  B_k = sigmoid(logit(B_0) + c_k)，B_0 = clip(组均值奖励, eps, 1-eps)；
- turn credit = sign(A_seq) * (B_k - B_{k-1})——真正的 credit 是
  **局部 gap 在历史累积信念上的边际变化**，不是孤立 gap 本身；
- credit 组内标准化后作 bounded multiplier（clip 到 [1-b, 1+b]），
  再按 (1-lambda) + lambda*w 与序列级 GRPO advantage 相乘重塑。
"""

from dataclasses import dataclass
from math import exp, log
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AgentOPSDConfig:
    """论文对齐的 turn advantage 重塑超参（官方默认值）。"""

    gamma: float = 0.95
    reshape_lambda: float = 0.5
    weight_bound: float = 0.2
    epsilon: float = 1e-4


def group_sequence_advantages(
    rewards: Sequence[float],
    *,
    epsilon: float,
) -> tuple[list[float], float]:
    """计算组内 GRPO 序列优势与初始 belief。

    advantage = (r - mean) / (std(ddof=0) + epsilon)；返回值第二项是
    **未 clip 的组均值奖励**，作 reshape 阶段的 initial_belief（clip 在
    reshape 内部做）。全同分组 std=0、advantage 全 0。
    """

    mean_reward = sum(rewards) / len(rewards)
    std_reward = float(np.std(rewards, ddof=0))
    advantages = [
        (reward - mean_reward) / (std_reward + epsilon)
        for reward in rewards
    ]
    return advantages, mean_reward


def _sigmoid(value: float) -> float:
    """数值稳定 sigmoid：正数走 1/(1+e^-x)，负数走 e^x/(1+e^x)。"""

    if value >= 0.0:
        inverse = exp(-value)
        return 1.0 / (1.0 + inverse)
    forward = exp(value)
    return forward / (1.0 + forward)


def score_turns(
    steps: Sequence[tuple[int, int, Sequence[float]]],
    teacher_logprobs: Sequence[float],
    student_initial_length: int,
    teacher_initial_length: int,
) -> tuple[tuple[float, ...], int, tuple[tuple[float, ...], ...]]:
    """按 turn 聚合 Teacher-Student 的 token 级 logprob gap。

    steps 每项为 (action_start, action_end, student_logprobs)——动作 token
    在 student 全序列里的下标区间与 student 侧 logprob。Teacher 轨迹 =
    换掉技能前缀后拼接同一条 student 后缀，前缀长度差
    prefix_offset = teacher_initial_length - student_initial_length 平移
    全部区间。每个 turn 的 evidence = sum(teacher_lp - student_lp)，
    **zip 截断语义**：teacher 切片比 student logprobs 短时按短侧求和。
    返回 (每 turn evidence, action token 总数, 每 turn teacher 切片)。
    """

    evidences: list[float] = []
    slices: list[tuple[float, ...]] = []
    action_tokens = 0
    prefix_offset = teacher_initial_length - student_initial_length

    for action_start, action_end, student_logprobs in steps:
        teacher_start = action_start + prefix_offset
        teacher_end = action_end + prefix_offset
        values = teacher_logprobs[teacher_start:teacher_end]
        numeric = [float(value) for value in values]
        evidence = sum(
            teacher - student
            for teacher, student in zip(numeric, student_logprobs)
        )
        slices.append(tuple(numeric))
        evidences.append(float(evidence))
        action_tokens += len(numeric)
    return tuple(evidences), action_tokens, tuple(slices)


def reshape_turn_advantages(
    sequence_advantage: float,
    initial_belief: float,
    evidences: Sequence[float],
    config: AgentOPSDConfig = AgentOPSDConfig(),
) -> tuple[float, list[dict]]:
    """根据每轮证据重塑 advantage，返回 (clip 后 initial_belief, 每 turn 账)。

    递推账（官方顺序，逐字段对应 TurnCredit dataclass）:
      b0 = clip(initial_belief, eps, 1-eps)，logit_b0 = log(b0/(1-b0))
      c_k = gamma*c_{k-1} + e_k            （c_0 = 0）
      B_k = sigmoid(logit_b0 + c_k)        （belief_before 链式推进）
      outcome_credit_k = sign(A_seq) * (B_k - B_{k-1})
      z_k = (credit_k - mean(credits)) / (std(credits, ddof=0) + eps)
      w_k = clip(1 + b*z_k, 1-b, 1+b)
      A_turn_k = A_seq * ((1-lambda) + lambda*w_k)
    sign(0) = 0 → 零优势轨迹 outcome_credit 全 0、z 全 0、w 全 1，
    A_turn 全 0（乘子 = (1-lambda) + lambda*1 = 1 但 A_seq=0）。
    turn_index 从 1 起。
    """

    b0 = min(max(initial_belief, config.epsilon), 1.0 - config.epsilon)
    logit_b0 = log(b0 / (1.0 - b0))
    cumulative = 0.0
    belief_before = b0
    raw: list[tuple[float, float, float, float]] = []

    for evidence in evidences:
        cumulative = config.gamma * cumulative + evidence
        belief_after = _sigmoid(logit_b0 + cumulative)
        raw.append((evidence, cumulative, belief_before, belief_after))
        belief_before = belief_after

    sign = 1.0 if sequence_advantage > 0.0 else (
        -1.0 if sequence_advantage < 0.0 else 0.0
    )
    outcome_credits = [
        sign * (belief_after - belief_before_value)
        for _, _, belief_before_value, belief_after in raw
    ]
    credit_mean = sum(outcome_credits) / len(outcome_credits)
    credit_std = float(np.std(outcome_credits, ddof=0))
    normalized = [
        (credit - credit_mean) / (credit_std + config.epsilon)
        for credit in outcome_credits
    ]

    lower_weight = 1.0 - config.weight_bound
    upper_weight = 1.0 + config.weight_bound
    turns: list[dict] = []
    for index, (
        (evidence, cumulative_value, before, after),
        outcome_credit,
        normalized_credit,
    ) in enumerate(zip(raw, outcome_credits, normalized), start=1):
        weight = min(
            max(1.0 + config.weight_bound * normalized_credit, lower_weight),
            upper_weight,
        )
        multiplier = (
            (1.0 - config.reshape_lambda)
            + config.reshape_lambda * weight
        )
        turn_advantage = sequence_advantage * multiplier
        turns.append(
            {
                "turn_index": index,
                "evidence": evidence,
                "cumulative_evidence": cumulative_value,
                "belief_before": before,
                "belief_after": after,
                "delta_belief": after - before,
                "outcome_credit": outcome_credit,
                "normalized_credit": normalized_credit,
                "weight": weight,
                "advantage": turn_advantage,
            }
        )

    return b0, turns


def assign_agentopsd_advantages(
    trajectories: Sequence[tuple[str, float, Sequence[float], int]],
    config: AgentOPSDConfig = AgentOPSDConfig(),
) -> tuple[list[dict], int, int]:
    """为每组轨迹计算序列优势与 turn 优势并展开到 step。

    trajectories 每项为 (group_id, reward, turn_evidences, n_steps)。
    返回 (逐轨迹结果, 组数, 退化组数)；逐轨迹结果含
    sequence_advantage / initial_belief（clip 后）/ turn_advantages /
    step_evidences / step_turn_advantages——后两者按
    zip(steps, turns) 截断写回（steps 少于 evidences 时只填前缀）。
    退化组 = 组内全部 |advantage| <= epsilon（全同奖励 → std=0）。
    """

    groups: dict[str, list[tuple]] = {}
    for trajectory in trajectories:
        groups.setdefault(str(trajectory[0]), []).append(trajectory)

    degenerate_groups = 0
    results: list[dict | None] = [None] * len(trajectories)
    index_of = {id(t): i for i, t in enumerate(trajectories)}
    for group in groups.values():
        sequence_advantages, initial_belief = group_sequence_advantages(
            [trajectory[1] for trajectory in group],
            epsilon=config.epsilon,
        )
        if all(abs(value) <= config.epsilon for value in sequence_advantages):
            degenerate_groups += 1

        for trajectory, sequence_advantage in zip(group, sequence_advantages):
            group_id, _, turn_evidences, n_steps = trajectory
            clipped_belief, turns = reshape_turn_advantages(
                sequence_advantage,
                initial_belief,
                turn_evidences,
                config,
            )
            step_evidences = [t["evidence"] for t in turns[:n_steps]]
            step_turn_advantages = [t["advantage"] for t in turns[:n_steps]]
            results[index_of[id(trajectory)]] = {
                "group_id": group_id,
                "sequence_advantage": sequence_advantage,
                "initial_belief": clipped_belief,
                "turn_advantages": [t["advantage"] for t in turns],
                "step_evidences": step_evidences,
                "step_turn_advantages": step_turn_advantages,
            }

    return results, len(groups), degenerate_groups  # type: ignore[return-value]


def build_ppo_arrays(
    full_tokens: Sequence[int],
    steps: Sequence[tuple[int, Sequence[float]]],
    turn_advantages: Sequence[float],
) -> dict:
    """构造只训练 action token 的 PPO 数组（官方 loss.py 账）。

    全 token 先填 0，再按 zip(steps, turn_advantages) 把每个动作 token 的
    old_logprob 与其 turn 的 advantage 广播到 position = action_start +
    offset。随后右移对齐：input = tokens[:-1]、target = tokens[1:]，
    logprobs/advantages 取 [1:]（position p 的标签是 token p+1）。
    steps 每项为 (action_start, logprobs)；zip 截断语义与官方一致。
    """

    old_logprobs_by_token = [0.0] * len(full_tokens)
    advantages_by_token = [0.0] * len(full_tokens)
    action_tokens = 0

    for (action_start, logprobs), credit in zip(steps, turn_advantages):
        for offset, old_logprob in enumerate(logprobs):
            position = action_start + offset
            old_logprobs_by_token[position] = old_logprob
            advantages_by_token[position] = credit
        action_tokens += len(logprobs)

    input_tokens = list(full_tokens[:-1])
    target_tokens = list(full_tokens[1:])
    old_logprobs = old_logprobs_by_token[1:]
    advantages = advantages_by_token[1:]

    return {
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
        "action_tokens": action_tokens,
    }


def solve(mode=None, **kw):
    if mode == "group":
        return group_sequence_advantages(
            kw["rewards"], epsilon=kw["epsilon"])
    if mode == "reshape":
        config = AgentOPSDConfig(
            gamma=kw.get("gamma", 0.95),
            reshape_lambda=kw.get("reshape_lambda", 0.5),
            weight_bound=kw.get("weight_bound", 0.2),
            epsilon=kw.get("epsilon", 1e-4),
        )
        return reshape_turn_advantages(
            kw["sequence_advantage"], kw["initial_belief"],
            kw["evidences"], config)
    if mode == "assign":
        config = AgentOPSDConfig(
            gamma=kw.get("gamma", 0.95),
            reshape_lambda=kw.get("reshape_lambda", 0.5),
            weight_bound=kw.get("weight_bound", 0.2),
            epsilon=kw.get("epsilon", 1e-4),
        )
        return assign_agentopsd_advantages(kw["trajectories"], config)
    if mode == "score":
        return score_turns(
            kw["steps"], kw["teacher_logprobs"],
            kw["student_initial_length"], kw["teacher_initial_length"])
    if mode == "ppo":
        return build_ppo_arrays(
            kw["full_tokens"], kw["steps"], kw["turn_advantages"])
    raise ValueError("unknown mode: %s" % mode)
