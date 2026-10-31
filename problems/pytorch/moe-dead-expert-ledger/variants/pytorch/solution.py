"""参考答案：外层循环 step、内层全 E 个专家向量化，把双阈值滞回写成一台显式状态机。

四个容易写错的点：

1. **一步只走一条分支**。同一 step 里只能按「当前状态」更新那一个计数器：alive 态只看
   `ema < death_threshold` 累计 miss，dead 态只看 `ema >= revive_threshold` 累计 revive。
   这条口径本身其实**不影响结果**：题面保证 `death_threshold <= revive_threshold`，于是两侧
   判据互斥，「每步两个计数器都推进」与「只推进当前状态那个」给出同一张表（变异测试证实判题区分不了）。
   写成分支式是因为它才是状态机的本来的样子，也是唯一能推广到 `revive < death` 的写法。
2. **翻转时把两个计数器一起清零**，同理是防御性写法：正因为判据互斥，进入新状态时另一个
   计数器必然已经是 0，清不清都一样。由此得到一个真的、可判的性质：**翻转发生的那一步
   `streak` 必为 0**，这是判题里最好用的一条不变量。
3. **比较的严格性不能糊**。低于用严格 `<`，复活用 `>=`；于是落在闭区间
   [death_threshold, revive_threshold) 里的 EMA 既不算 miss 也不算 revive，这就是滞回带。
   两个阈值相等时退化成单阈值，此时 ema 恰好等于阈值只可能让它复活、不会让它判死。
   这两处严格性**是**能被判题抓到的，别顺手改成 `<=`。
4. **步内不能提前覆盖 `dead[s]`**。`dead[s]` 记的是「这一 step 结束后的状态」，
   `streak[s]` 记的是那个状态下的活计数器，两者必须在状态机推完之后一起写。

EMA 递推按字面顺序 `beta * ema + (1 - beta) * load` 计算，与纯 Python 的浮点序列一致；
判题的容差是 `2e-11`，所以数学等价的写法（比如 `torch.lerp(load, ema, beta)`）在本机也能过——
本题约束的是状态机语义，不是归约顺序。真正必须逐位对齐的是**冷启动口径**：EMA 初值 0、
第一步照常做递推，因此 `ema[0] == (1 - beta) * load[0]`，把它改成 `ema[0] = load[0]` 会整体错位。
"""

import torch


def _as_real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a python int or float")
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _as_count(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a python int")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _check(load, beta, death_threshold, revive_threshold, patience, revive_patience):
    if not isinstance(load, torch.Tensor):
        raise ValueError("load must be a torch.Tensor")
    if load.device.type != "cpu":
        raise ValueError("load must be on CPU")
    if load.dtype != torch.float64:
        raise ValueError("load must be float64")
    if load.dim() != 2 or min(load.shape) < 1:
        raise ValueError("load must be a non-empty 2-D [S, E] tensor")
    if not bool(torch.isfinite(load).all()):
        raise ValueError("load must be finite")
    if bool((load < 0.0).any()):
        raise ValueError("load must be non-negative")
    beta = _as_real(beta, "beta")
    death_threshold = _as_real(death_threshold, "death_threshold")
    revive_threshold = _as_real(revive_threshold, "revive_threshold")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    if death_threshold < 0.0 or revive_threshold < 0.0:
        raise ValueError("thresholds must be non-negative")
    if revive_threshold < death_threshold:
        raise ValueError("revive_threshold must not be below death_threshold")
    _as_count(patience, "patience")
    _as_count(revive_patience, "revive_patience")


def solve(load, beta, death_threshold, revive_threshold, patience, revive_patience):
    _check(load, beta, death_threshold, revive_threshold, patience, revive_patience)
    steps, experts = load.shape
    keep = float(beta)
    rise = 1.0 - keep
    death = float(death_threshold)
    revive = float(revive_threshold)

    ema_all = torch.empty(steps, experts, dtype=torch.float64)
    dead_all = torch.zeros(steps, experts, dtype=torch.bool)
    streak_all = torch.zeros(steps, experts, dtype=torch.int64)
    first_dead = torch.full((experts,), -1, dtype=torch.int64)
    changes = torch.zeros(experts, dtype=torch.int64)
    dead_steps = torch.zeros(experts, dtype=torch.int64)

    ema = torch.zeros(experts, dtype=torch.float64)
    is_dead = torch.zeros(experts, dtype=torch.bool)
    miss = torch.zeros(experts, dtype=torch.int64)
    back = torch.zeros(experts, dtype=torch.int64)

    for step in range(steps):
        ema = keep * ema + rise * load[step]
        ema_all[step] = ema
        alive = ~is_dead
        # 只推进当前状态那一个计数器
        miss = torch.where(alive & (ema < death), miss + 1, torch.where(alive, torch.zeros_like(miss), miss))
        back = torch.where(is_dead & (ema >= revive), back + 1, torch.where(is_dead, torch.zeros_like(back), back))

        dies = alive & (miss >= patience)
        wakes = is_dead & (back >= revive_patience)
        # 翻转一律两个计数器一起清零（防御性写法：判据互斥时另一个本来就是 0，清不清同结果）
        miss = torch.where(dies | wakes, torch.zeros_like(miss), miss)
        back = torch.where(dies | wakes, torch.zeros_like(back), back)
        is_dead = torch.where(dies, torch.ones_like(is_dead), torch.where(wakes, torch.zeros_like(is_dead), is_dead))

        dead_all[step] = is_dead
        streak_all[step] = torch.where(is_dead, back, miss)
        first_dead = torch.where(dies & (first_dead < 0), torch.full_like(first_dead, step), first_dead)
        changes = changes + dies.to(torch.int64) + wakes.to(torch.int64)
        dead_steps = dead_steps + is_dead.to(torch.int64)

    return ema_all, dead_all, streak_all, first_dead, changes, dead_steps
