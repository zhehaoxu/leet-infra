import torch


def solve(load, beta, death_threshold, revive_threshold, patience, revive_patience):
    """中文契约 / English contract
    load [S,E] 是 CPU float64、非空 2-D、有限且非负：每个 step 每个专家收到的 token 数。
    beta、death_threshold、revive_threshold 是 python 实数（bool 不算）：0 <= beta < 1，
    0 <= death_threshold <= revive_threshold；patience、revive_patience 是 >= 1 的 python int。
    EMA：初值 0，每个 step 都做 ema = beta*ema + (1-beta)*load[s]，冷启动不豁免 /
        start at 0 and update every step, no warm-up exemption.
    状态机（每个专家独立，alive 起步）：alive 态数「连续 ema < death_threshold」的步数，达到
    patience 就转成 dead；dead 态数「连续 ema >= revive_threshold」的步数，达到 revive_patience
    才转回 alive。比较的严格性照字面：低于用 <，不低于用 >= /
        below uses a strict comparison, revival uses a non-strict one.
    每步只按**当前状态**更新对应计数器；一旦翻转，两个计数器都清零，本步不再用另一侧判据 /
        one state-dependent counter per step; a transition resets both counters.
    非法输入（dtype/形状/设备/非有限/负负载/beta 越界/阈值倒置/计数器非 int 或 < 1）抛 ValueError；
    不修改输入，接受非连续张量 / raise ValueError otherwise; mutate nothing; accept noncontiguous tensors.
    返回 / Returns (ema float64 [S,E], dead bool [S,E], streak int64 [S,E],
        first_dead int64 [E]（首次判死的 step，从未死为 -1）,
        state_changes int64 [E]（状态翻转次数）, dead_steps int64 [E]（处于 dead 的步数）)；
        streak 在 alive 态是该专家的 miss 连击数，在 dead 态是 revive 连击数 /
        streak is the live counter of whichever state the expert is in at that step.
    """
    raise NotImplementedError
