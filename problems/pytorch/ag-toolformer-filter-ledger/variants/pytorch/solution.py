"""
Toolformer 候选调用过滤账：把 Toolformer（arXiv 2302.04761，以
v5 论文口径为准，官方未开源代码，去重/实现口径补 lucidrains/
toolformer-pytorch）的三段流水线无状态化：位置采样（τ_s 门 +
top-k + 每位置 m 上限）→ 加权 CE 过滤（w̃_t 权重、L⁻=min 双对照、
τ_f 门）→ 文本级去重（replace_all_but_first）。

solve(text_with_calls, positions, cands, tau_s=0.05, tau_f=1.0,
      k=5, m=5)

- text_with_calls：含 API 调用标记的增强文本（调用格式
  ' [tool(args)]'，正则要求**前导空格**）。
- positions：[(index, p), ...]，index 为 <API> token 的候选位置、
  p 为该位置 <API> token 的采样概率。
- cands：候选调用 [{"pos", "call", "lp_plus", "lp_none",
  "lp_call_only"}, ...]；pos 对应 positions 的 index；三个 lp 列表
  是**同一后缀 token 序列**（x_pos..x_n）在三前缀（e(c,r)/ε/
  e(c,ε)）下的逐 token log 概率，**后缀对齐**（下标 t=0 为调用
  位置 token、t 即权重步号），长度须一致否则 ValueError；
  t≥5 的 token 权重为 0 但仍须在列表内。

口径（论文 Sec 2.2 / 4.1 / App A + lucidrains 实现）：
1. **位置采样**：I = {i | p_i > tau_s}（**严格 >**，恰等丢弃）；
   超过 k 个时取 top-k（按 p 降序稳定排序，平分保原顺序）；
   再对每个保留位置取其**前 m 个**候选（输入序）。
2. **加权 CE**：w̃_t = max(0, 1 − 0.2·t)（t≥5 权重 0），
   **全局归一** w_t = w̃_t / Σ_{s∈ℕ} w̃_s = w̃_t / 3.0
   （Σ = 1+0.8+0.6+0.4+0.2 = 3，常数；**不按候选后缀长度重归一**）。
   L_i(z) = −Σ_j w_{j−pos} · lp_z[j]；L⁺ = L(e(c,r))；
   L⁻ = min(L(ε), L(e(c, ε)))（双对照取小：完全无调用 vs 只有
   调用没结果）；score = L⁻ − L⁺；保留条件 **≥ tau_f（含等）**。
3. **文本级去重**：正则 r'(\\ \\[(\\w+)\\(([^)]*)\\))(\\])'
   （lucidrains create_function_regex：前导空格、\\w+ 工具名、
   args [^)]* 遇 ')' 截断——args 含 ')' 的调用**不匹配**）；
   保留第一个完整匹配，其余替换为 ''（replace_all_but_first）；
   removed 按出现序记录被移除的完整匹配串。
   has_calls = 去重前文本是否含调用（lucidrains 数据管线把无调用
   样本整对剔除的依据）。

返回账：
{"kept_positions": [index...],        # 采样后（τ_s + top-k，原序）
 "tau_dropped": [index...],           # p <= tau_s（原序）
 "topk_dropped": [index...],          # 过 τ_s 但被 top-k 截掉
 "evaluated": [cand 下标...],         # 实际算分的候选（输入序）
 "skipped_cands": [cand 下标...],     # 位置未过采样或超 m 上限
 "scores": [{"cand", "pos", "L_plus", "L_none", "L_call_only",
             "L_minus", "score", "kept"}, ...],   # evaluated 序
 "kept_calls": [过 tau_f 的 call 串（evaluated 序）],
 "has_calls": bool, "kept_text": str, "removed": [str...]}
域约束（ValueError）：cand 缺键；lp 三表长度不一致；pos 不在
positions 里。
"""
import re

_CALL_RE = re.compile(r'(\ \[(\w+)\(([^)]*)\))(\])')

_W_SUM = 1.0 + 0.8 + 0.6 + 0.4 + 0.2   # Σ_{s∈ℕ} max(0, 1-0.2s) = 3.0


def _wtilde(t):
    return max(0, 1 - 0.2 * t)


def _loss(lps):
    """lp 列表按后缀对齐：下标 t 即权重步号（t=0 为调用位置
    token），t>=5 权重为 0 仍占位。"""
    total = 0.0
    for t, lp in enumerate(lps):
        w = _wtilde(t) / _W_SUM
        if w:
            total += w * lp
    return -total


def solve(text_with_calls, positions, cands, tau_s=0.05, tau_f=1.0,
          k=5, m=5):
    # ---- 阶段一：位置采样
    tau_dropped = [i for i, p in positions if not (p > tau_s)]
    passed = [(i, p) for i, p in positions if p > tau_s]
    order = sorted(range(len(passed)),
                   key=lambda t: (-passed[t][1], t))
    topk_dropped = [passed[t][0] for t in order[k:]]
    kept = [passed[t][0] for t in sorted(order[:k])]
    kept_set = set(kept)

    # ---- 阶段二：候选校验（全体，先于评估）+ m 上限 + 加权 CE
    pos_set = {i for i, _ in positions}
    for ci, cd in enumerate(cands):
        for key in ("pos", "call", "lp_plus", "lp_none", "lp_call_only"):
            if key not in cd:
                raise ValueError(f"candidate #{ci} missing {key}")
        if not (len(cd["lp_plus"]) == len(cd["lp_none"])
                == len(cd["lp_call_only"])):
            raise ValueError(f"candidate #{ci} lp length mismatch")
        if cd["pos"] not in pos_set:
            raise ValueError(f"candidate #{ci} pos not in positions")
    per_pos = {}
    for ci, cd in enumerate(cands):
        per_pos.setdefault(cd["pos"], []).append(ci)
    evaluated = []
    for pos in kept:
        for ci in per_pos.get(pos, [])[:m]:
            evaluated.append(ci)
    evaluated_set = set(evaluated)
    skipped = [ci for ci in range(len(cands))
               if ci not in evaluated_set]
    scores = []
    for ci in evaluated:                    # evaluated 序
        cd = cands[ci]
        pos = cd["pos"]
        l_plus = _loss(cd["lp_plus"])
        l_none = _loss(cd["lp_none"])
        l_call = _loss(cd["lp_call_only"])
        l_minus = min(l_none, l_call)
        score = l_minus - l_plus
        scores.append({"cand": ci, "pos": pos, "L_plus": l_plus,
                       "L_none": l_none, "L_call_only": l_call,
                       "L_minus": l_minus, "score": score,
                       "kept": score >= tau_f})
    kept_calls = [cands[s["cand"]]["call"] for s in scores if s["kept"]]

    # ---- 阶段三：文本级去重
    removed = []
    count = 0

    def _sub(mo):
        nonlocal count
        count += 1
        if count > 1:
            removed.append(mo.group(0))
            return ''
        return mo.group(0)

    kept_text = _CALL_RE.sub(_sub, text_with_calls)
    return {"kept_positions": kept,
            "tau_dropped": tau_dropped,
            "topk_dropped": topk_dropped,
            "evaluated": evaluated,
            "skipped_cands": skipped,
            "scores": scores,
            "kept_calls": kept_calls,
            "has_calls": count > 0,
            "kept_text": kept_text,
            "removed": removed}
