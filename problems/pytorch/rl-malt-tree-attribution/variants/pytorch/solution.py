"""MALT 的搜索树成败归因账。

回源: 论文 MALT (arXiv:2412.01928, COLM 2025 v3) §4.2/A.6.1 的公式——
官方无开源实现，本题口径 = 论文公式唯一确定的算术账（无折扣因子，
期望回传 + 0.5 阈值 + 同父偏好配对）。叶奖励与树结构全部显式入参，
模型采样本身不出题。

三层 Generate→Verify→Refine 树（分支因子 n，每题 n³ 条轨迹）:
- V(r_{j,k,l}) = 叶二元奖励 R ∈ {0,1}（与 ground truth 比较，入参给定）
- V(v_{j,k}) = (1/n) Σ_l V(r_{j,k,l})     （同一 verifier 输出的兄弟期望）
- V(g_j)   = (1/n) Σ_k V(v_{j,k})        （同一 generator 输出的兄弟期望）
- 题根不计价值。

二值化（对齐多数投票）: label = (V > θ)，θ=0.5 时**严格大于**——
V 恰等于 0.5 记 ×（n 为偶数时兄弟均值可取到 0.5）。

偏好对（同父上下文配对，非跨分支）:
- D_R: 同一 verifier 输出 v_{j,k} 下的 (r⁺, r⁻)，V(r⁺)=1 ∧ V(r⁻)=0
- D_V: 同一 generator 输出 g_j 下的 (v⁺, v⁻)，v̂⁺=✓ ∧ v̂⁻=×
- D_G: 同一题目 q 下的 (g⁺, g⁻)——论文定义了 D_G 但 Generator 只做
  SFT（基模型已是偏好对齐的生成器），不用 DPO。

SFT 正样本: R 层取 V(r)=1；V/G 层取二值 label 为 ✓ 的输出。
"""

from typing import Sequence


def propagate_values(
    rewards: Sequence[Sequence[Sequence[int]]],
    n: int,
) -> tuple[list[list[float]], list[float]]:
    """自下而上期望回传：叶奖励 → verifier 值 → generator 值。

    rewards[j][k][l] 是 r_{j,k,l} 的二元奖励（0/1）。V(v_{j,k}) =
    sum_l / n、V(g_j) = sum_k(V(v_{j,k})) / n——两层都是对**直接子节点**
    的样本均值（论文的 Monte Carlo 无偏估计），无折扣因子。
    校验树形：len(rewards) == n 且每层 len == n，否则 ValueError。
    """

    if len(rewards) != n or any(
        len(branch) != n or any(len(leaf) != n for leaf in branch)
        for branch in rewards
    ):
        raise ValueError("tree must be n x n x n")

    verifier_values = [
        [sum(leaf_rewards) / n for leaf_rewards in branch]
        for branch in rewards
    ]
    generator_values = [
        sum(branch_values) / n for branch_values in verifier_values
    ]
    return verifier_values, generator_values


def binarize(
    values: Sequence[float],
    theta: float = 0.5,
) -> list[bool]:
    """阈值二值化：V > θ 记 ✓，其余（含恰等于 θ）记 ×。

    论文对齐多数投票：θ=0.5 时 V=0.5 判不正确（严格大于）。"""

    return [value > theta for value in values]


def build_preference_pairs(
    unit_values: Sequence[float],
    parent_ids: Sequence[int],
    theta: float = 0.5,
) -> list[tuple[int, int]]:
    """同父上下文内的 (正, 负) 偏好对（下标对）。

    unit_values 是候选输出的连续值，parent_ids 是每个输出所属父节点
    的下标（R 层 = (j*n+k) 编码的 verifier 节点、V 层 = generator 节点、
    G 层 = 题目）。同父内正子集 × 负子集的**全部组合**各出一对
    （正下标在前）。unit_values 的二值化用与标签一致的严格 > θ。
    """

    positives: dict[int, list[int]] = {}
    negatives: dict[int, list[int]] = {}
    for index, (value, parent) in enumerate(zip(unit_values, parent_ids)):
        if value > theta:
            positives.setdefault(parent, []).append(index)
        else:
            negatives.setdefault(parent, []).append(index)
    pairs: list[tuple[int, int]] = []
    for parent in sorted(set(positives) | set(negatives)):
        for pos in positives.get(parent, []):
            for neg in negatives.get(parent, []):
                pairs.append((pos, neg))
    return pairs


def sft_positives(
    unit_values: Sequence[float],
    parent_ids: Sequence[int],
    theta: float = 0.5,
) -> dict:
    """SFT 正样本收集与逐父诊断。

    返回 {positives, counts}——positives 是 ✓ 输出的下标（保持输入
    顺序），counts 是逐父节点的 (正数, 负数) 对账（父下标升序）。
    """

    positives = [
        index
        for index, (value, _) in enumerate(zip(unit_values, parent_ids))
        if value > theta
    ]
    tally: dict[int, list[int]] = {}
    for index, (value, parent) in enumerate(zip(unit_values, parent_ids)):
        entry = tally.setdefault(parent, [0, 0])
        entry[0 if value > theta else 1] += 1
    counts = [(parent, tally[parent][0], tally[parent][1])
              for parent in sorted(tally)]
    return {"positives": positives, "counts": counts}


def attribute_question(
    rewards: Sequence[Sequence[Sequence[int]]],
    n: int,
    theta: float = 0.5,
) -> dict:
    """单题全流程：期望回传 → 二值化 → 三层配对与正样本。

    树编码：r 叶下标 (j, k, l) 线性化 idx = j*n*n + k*n + l；
    v 下标 (j, k) → j*n + k；g 下标 j。返回
    {verifier_values, generator_values, verifier_labels, generator_labels,
    refiner_indices/labels（按线性序）, refiner_pairs, verifier_pairs,
    generator_pairs, sft}——pairs 是 (正下标, 负下标) 列表，与论文
    D_R/D_V/D_G 定义一致（D_G 仅构造不用于 DPO）。
    """

    verifier_values, generator_values = propagate_values(rewards, n)

    flat_refiner_values: list[float] = []
    refiner_parents: list[int] = []
    flat_verifier_values: list[float] = []
    verifier_parents: list[int] = []
    for j, branch in enumerate(verifier_values):
        for k, value in enumerate(branch):
            flat_verifier_values.append(value)
            verifier_parents.append(j)
            for l, leaf_reward in enumerate(rewards[j][k]):
                flat_refiner_values.append(float(leaf_reward))
                refiner_parents.append(j * n + k)

    verifier_labels = binarize(flat_verifier_values, theta)
    generator_labels = binarize(generator_values, theta)
    refiner_labels = binarize(flat_refiner_values, theta)

    refiner_pairs = build_preference_pairs(
        flat_refiner_values, refiner_parents, theta)
    verifier_pairs = build_preference_pairs(
        flat_verifier_values, verifier_parents, theta)
    generator_pairs = build_preference_pairs(
        generator_values, [0] * len(generator_values), theta)

    sft = {
        "refiner": sft_positives(flat_refiner_values, refiner_parents, theta),
        "verifier": sft_positives(
            flat_verifier_values, verifier_parents, theta),
        "generator": sft_positives(
            generator_values, [0] * len(generator_values), theta),
    }

    return {
        "verifier_values": verifier_values,
        "generator_values": generator_values,
        "verifier_labels": verifier_labels,
        "generator_labels": generator_labels,
        "refiner_values": flat_refiner_values,
        "refiner_labels": refiner_labels,
        "refiner_pairs": refiner_pairs,
        "verifier_pairs": verifier_pairs,
        "generator_pairs": generator_pairs,
        "sft": sft,
    }


def solve(mode=None, **kw):
    if mode == "propagate":
        return propagate_values(kw["rewards"], kw["n"])
    if mode == "binarize":
        return binarize(kw["values"], kw.get("theta", 0.5))
    if mode == "pairs":
        return build_preference_pairs(
            kw["unit_values"], kw["parent_ids"], kw.get("theta", 0.5))
    if mode == "sft":
        return sft_positives(
            kw["unit_values"], kw["parent_ids"], kw.get("theta", 0.5))
    if mode == "attribute":
        return attribute_question(
            kw["rewards"], kw["n"], kw.get("theta", 0.5))
    raise ValueError("unknown mode: %s" % mode)
