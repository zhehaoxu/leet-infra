# Toolformer 调用过滤账（ag-toolformer-filter-ledger）

**难度**：Medium ｜ **合集**：agentic-systems ｜ **order**：727

把 Toolformer（arXiv **2302.04761**，官方未开源，以 **v5 论文口径**为准；去重与实现细节补 [lucidrains/toolformer-pytorch](https://github.com/lucidrains/toolformer-pytorch)）的三段候选调用流水线无状态化：位置采样（τ_s 门 + top-k + 每位置 m 上限）→ 加权交叉熵过滤（w̃_t 权重、L⁻ 双对照取小、τ_f 门）→ 文本级去重（replace_all_but_first）。LLM 逐 token log 概率全部显式入参。

## 你要实现的

`solve(text_with_calls, positions, cands, tau_s=0.05, tau_f=1.0, k=5, m=5)`：返回采样三态账、逐候选 loss/score 账与去重文本账。

## 账点（论文 + lucidrains 实现逐行核对）

1. **位置采样**：`I = {i | p_i > tau_s}`——**严格 >**（恰等丢弃，App A 默认 τ_s=0.05）；超过 k 个时取 top-k（按 p 降序**稳定**排序，平分保原顺序；默认 k=5）。
2. **每位置 m 上限**：每个保留位置只评估其**前 m 个**候选（输入序；默认 m=5；计算器/翻译工具官方另设 τ_s=0、k=20、m=10、τ_f=0.5——均为显式入参）。
3. **权重函数（Sec 4.1）**：`w̃_t = max(0, 1 − 0.2·t)`（t 为后缀内步号，t≥5 权重 0 但仍占位），**全局归一** `w_t = w̃_t / Σ_{s∈ℕ} w̃_s = w̃_t / 3.0`（Σ = 1+0.8+0.6+0.4+0.2 = 3 常数，**不按候选后缀长度重归一**——2 token 后缀全 -1 的 L = 1.8/3 = 0.6 而非 1）。
4. **加权 CE**：`L(z) = −Σ_t w_t · lp_z[t]`（lp 列表按后缀对齐：下标 t=0 为调用位置 token）。
5. **双对照取小**：`L⁻ = min(L(ε), L(e(c, ε)))`——完全无调用 vs 只有调用没结果，取小者；`score = L⁻ − L⁺`。
6. **过滤门**：保留条件 **score ≥ tau_f（含等）**（App A 默认 τ_f=1.0）。**浮点 quirk**：全 -1 的 5 token 后缀 score = 0.9999999999999999 < 1.0——恰等边界被 `(w̃/3)·lp` 舍入吃掉，τ_f=1.0 时判丢；含等语义以「三表全同 → score 恰 0.0」锚点钉住。
7. **文本级去重（lucidrains）**：正则 `' [tool(args)]'` 三 quirk——**前导空格必须**（行首无空格的调用不匹配）、工具名 `\w+`（空名 ` [(2)]` 不匹配）、args `[^)]*` 遇 `)` 截断（args 含 `)` 的调用整体不匹配）；保留第一个完整匹配、其余**替换为 ''**；removed 记完整匹配串；`has_calls` = 去重前是否含调用（数据管线据此整对剔除无调用样本）。
8. **域约束**：候选缺键 / lp 三表长度不一致 / pos 不在 positions → ValueError（**全体候选前置校验**，未过采样的候选也查）。

## 样例

```python
positions = [(0, 0.05), (1, 0.6), (2, 0.7)]
cands = [{"pos": 1, "call": "qa1",
          "lp_plus": [0, 0, -5, 0, 0],        # t=2 权重 0.2 → L⁺=1.0 精确
          "lp_none": [0, 0, -10, 0, 0],       # L=2.0
          "lp_call_only": [0, 0, -10, 0, 0]}]
led = solve("a [calc(1)] b [calc(2)] c", positions, cands,
            tau_s=0.05, tau_f=0.9, k=2, m=1)
# tau_dropped == [0]（0.05 恰等 → 严格 > 丢弃）
# kept_positions == [1, 2]；m=1 每位置取 1 个
# score = 2.0 − 1.0 = 1.0 ≥ 0.9 → kept；kept_calls == ["qa1"]
# removed == [" [calc(2)]"]（去重保第一个）
```

## 计分

8 个检查点：τ_s 严格门与 top-k 稳定序、m 上限与 skipped、权重表与全局归一、t≥5 零权重、双对照取小、含等门与浮点恰等 quirk、去重三 quirk（前导空格/空名/括号 args），以及 120 组随机流水线对拍。
