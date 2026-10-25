"""参考答案：llama.cpp 采样管线全序装配（capstone 顺序题）。

装配顺序严格对齐源码：
- `common/sampling.cpp::common_sampler_init`：**logit-bias 恒在链首**（并把 vocab 的 suppress
  tokens 以 `-INFINITY` 追加进同一条 bias 串），链尾恒为 `dist`（`temp==0` 或 `top_k==1`
  时换成 `greedy`）。
- `common/common.h::common_params_sampling::samplers` 默认序
  = penalties; dry; top-n-sigma; top-k; typical; top-p; min-p; xtc; temp
  ⇒ **截断类在 temp 之前**（temp 是链上最后一个），这正是 llama.cpp 口径与 HF
  「先温度缩放再截断」的本质差别：`top_p/min_p/xtc` 看到的那份 softmax 概率
  （`llama_sampler_softmax_impl(cur_p)`：对**当前存活候选**的 logits 归一化）是**未过温度**的。

行下标固定表示 stage，每行 = 该 stage 执行完后的存活集合（temp 行跟随自己的执行时机）：
0=logit-bias(+banned) 1=penalties 2=dry 3=top-k 4=top-p 5=min-p 6=xtc 7=temp

返回 (masks, probs, chosen)：masks bool (8,V)、probs (V,) 为 dist 步真正使用的最终分布
（存活集合上归一化，其余为 0）、chosen 标量 int64。
"""

import math

import torch

NEG_INF = float("-inf")
_FLOAT_MAX_LOG = 88.7228391


def _probs(lg, alive):
    """对存活候选的 logits 做 softmax（等价 llama.cpp 的 softmax_impl(cur_p)）。"""
    if not bool(alive.any()):
        return torch.zeros_like(lg)
    mx = lg[alive].max()
    ex = torch.where(alive, torch.exp(lg - mx), torch.zeros_like(lg))
    return ex / ex.sum()


def _desc_order(lg, alive):
    """按 logit 降序（并列按索引升序）返回存活 token 的索引列表。"""
    order = torch.argsort(lg, dim=-1, descending=True, stable=True)
    return [int(i) for i in order.tolist() if bool(alive[int(i)])]


def _apply_temp(lg, alive, temp):
    if float(temp) > 0.0:
        lg[alive] = lg[alive] / float(temp)
        return alive, lg
    # temp <= 0：只留 logit 最大的一个（llama.cpp 把其余置 -INF，效果等价于出链）
    cand = _desc_order(lg, alive)
    new_alive = torch.zeros_like(alive)
    if cand:
        new_alive[cand[0]] = True
        lg[~new_alive & alive] = NEG_INF
    return new_alive, lg


def _apply_penalties(lg, alive, history, repeat, freq, present, last_n):
    last_n = int(last_n)
    win = list(history) if last_n < 0 else list(history)[-last_n:] if last_n > 0 else []
    counts = {}
    for t in win:
        counts[int(t)] = counts.get(int(t), 0) + 1
    if float(repeat) == 1.0 and float(freq) == 0.0 and float(present) == 0.0:
        return lg
    for tid, c in counts.items():
        if 0 <= tid < lg.shape[0] and bool(alive[tid]) and c > 0:
            l = float(lg[tid])
            l = l * float(repeat) if l <= 0.0 else l / float(repeat)
            l = l - c * float(freq) - float(present)
            lg[tid] = l
    return lg


def _apply_dry(lg, alive, history, mult, base, allowed, last_n):
    """反向 Z-algorithm（对齐 llama.cpp 无 sequence-breaker 版 dry）。"""
    out = lg.clone()
    allowed = int(allowed)
    last_n = int(last_n)
    last = list(history) if last_n < 0 else list(history)[-last_n:] if last_n > 0 else []
    n = len(last)
    if float(mult) == 0.0 or float(base) < 1.0 or last_n == 0 or n <= allowed:
        return out
    rev = last[::-1]
    z = [0] * n
    lt = rt = 0
    for k in range(1, n):
        if k > rt:
            c = 0
            while k + c < n and rev[c] == rev[k + c]:
                c += 1
            z[k] = min(c, n)
            if c > 0:
                lt, rt = k, k + c - 1
        else:
            p = k - lt
            right = rt - k + 1
            if z[p] < right:
                z[k] = min(z[p], n)
            else:
                i = rt + 1
                while i < n and rev[i] == rev[i - k]:
                    i += 1
                z[k] = min(i - k, n)
                lt, rt = k, i - 1
    best = {}
    for k in range(1, n):
        if z[k] >= allowed:
            tok = last[n - k]
            if z[k] > best.get(tok, 0):
                best[tok] = z[k]
    max_exp = int(_FLOAT_MAX_LOG / math.log(float(base))) if float(base) > 1.000001 else 0
    for tok, rl in best.items():
        if 0 <= tok < out.shape[0] and bool(alive[tok]):
            e = rl - allowed
            if max_exp and e > max_exp:
                e = max_exp
            out[tok] = out[tok] - float(mult) * (float(base) ** e)
    return out


def solve(logits, history, bias_tokens, bias_values, banned,
          dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
          penalty_repeat, penalty_freq, penalty_present, penalty_last_n,
          xtc_probability, xtc_threshold, min_keep, top_k, top_p, min_p,
          temp, temp_first, u_xtc, u):
    lg = logits.clone().float()
    v = lg.shape[0]
    mk = int(min_keep)
    alive = torch.ones(v, dtype=torch.bool)
    masks = torch.zeros(8, v, dtype=torch.bool)

    # ---- 0. logit-bias + banned（链首，恒定）----
    for t, b in zip(list(bias_tokens), list(bias_values)):
        tid = int(t)
        if 0 <= tid < v:
            lg[tid] = lg[tid] + float(b)
    for t in list(banned):
        tid = int(t)
        if 0 <= tid < v:
            alive[tid] = False
            lg[tid] = NEG_INF
    masks[0] = alive.clone()

    # ---- 1. penalties ----
    lg = _apply_penalties(lg, alive, history, penalty_repeat, penalty_freq,
                          penalty_present, penalty_last_n)
    masks[1] = alive.clone()

    # ---- 2. dry ----
    lg = _apply_dry(lg, alive, history, dry_multiplier, dry_base,
                    dry_allowed_length, dry_penalty_last_n)
    masks[2] = alive.clone()

    # ---- temp 的时机：llama.cpp 口径在截断之后；temp_first=True 切到 HF 口径 ----
    if bool(temp_first):
        alive, lg = _apply_temp(lg, alive, temp)
        masks[7] = alive.clone()

    # ---- 3. top-k（作用在 logit 域，softmax 单调 ⇒ 与概率降序同序）----
    k = int(top_k)
    if k > 0:
        cand = _desc_order(lg, alive)
        if len(cand) > k:
            new_alive = torch.zeros_like(alive)
            for t in cand[:k]:
                new_alive[t] = True
            alive = new_alive
    masks[3] = alive.clone()

    # ---- 4. top-p：在「当前存活候选、未过温度」的归一化概率上累积 ----
    if float(top_p) < 1.0:
        cand = _desc_order(lg, alive)
        if len(cand) > 1:
            p = _probs(lg, alive)
            cum = 0.0
            last_idx = len(cand)
            for i, t in enumerate(cand):
                cum += float(p[t])
                if cum >= float(top_p) and i + 1 >= mk:
                    last_idx = i + 1
                    break
            new_alive = torch.zeros_like(alive)
            for t in cand[:last_idx]:
                new_alive[t] = True
            alive = new_alive
    masks[4] = alive.clone()

    # ---- 5. min-p：logit 域阈值 max_logit + ln(p)（等价 p_i >= p * p_max，归一化口径无关）----
    if float(min_p) > 0.0 and bool(alive.any()):
        cand = _desc_order(lg, alive)
        min_logit = float(lg[cand[0]]) + math.log(float(min_p))
        i = 1
        while i < len(cand):
            if float(lg[cand[i]]) < min_logit and i >= mk:
                break
            i += 1
        new_alive = torch.zeros_like(alive)
        for t in cand[:i]:
            new_alive[t] = True
        alive = new_alive
    masks[5] = alive.clone()

    # ---- 6. xtc：砍头部前缀（概率门槛作用在当前存活集合的归一化概率上）----
    if float(xtc_probability) > 0.0 and float(xtc_threshold) <= 0.5:
        cand = _desc_order(lg, alive)
        if len(cand) >= 2 and float(u_xtc) <= float(xtc_probability):
            p = _probs(lg, alive)
            pos_last = 0
            for i, t in enumerate(cand):
                if float(p[t]) >= float(xtc_threshold):
                    pos_last = i
                else:
                    break
            if pos_last > 0 and len(cand) - pos_last >= mk:
                new_alive = torch.zeros_like(alive)
                for t in cand[pos_last:]:
                    new_alive[t] = True
                alive = new_alive
    masks[6] = alive.clone()

    if not bool(temp_first):
        alive, lg = _apply_temp(lg, alive, temp)
        masks[7] = alive.clone()

    # ---- 7. dist（链尾）：存活集合上 softmax，原顺序 CDF 逆变换（见 notes 的口径说明）----
    probs = _probs(lg, alive)
    cdf = torch.cumsum(probs, dim=-1)
    hit = cdf >= float(u)
    chosen = int(torch.nonzero(hit)[0, 0]) if bool(hit.any()) else int(
        torch.nonzero(alive)[-1, 0]) if bool(alive.any()) else 0
    return masks, probs, torch.tensor(chosen, dtype=torch.long)
