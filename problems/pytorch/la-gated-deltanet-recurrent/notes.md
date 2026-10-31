# 详解：Gated DeltaNet 递推账

## 出处

`fla-org/flash-linear-attention` 的 `fla/ops/gated_delta_rule/naive.py::naive_recurrent_gated_delta_rule`（2026-10-31 经 GitHub contents API 逐行取得）。论文出处 Gated DeltaNet（2412.06464）：`S_t = α_t S_{t-1}(I − β_t k_t k_tᵀ) + β_t k_t v_tᵀ`，`α_t = exp(g_t)`。fla 参考实现把它写成等价的「衰减 → delta 修正 → 外积写入」三步，数值上与论文式一致。

## 核心与易错点

1. **衰减与修正的顺序**：`h *= exp(g)` 必须在计算 `v' = v − h·k` **之前**——v' 用的是衰减后的状态。把衰减挪到写入之后的变异被 `gating_order`（`g=−50` 时状态≈0、修正只剩 `β·v`）与 `delta_rule_corrects` 标量账抓住。
2. **`β` 是逐头标量**：`v' = (v − h·k)·β` 与「β 乘在 k 上」（`h += (βk)⊗(v−hk)`）数学等价（外积两侧同缩放）——变异测试中 `beta-on-k` 是**观测等价变异**，notes 如实记录。真正不等价的是丢 β 或把 β 用错轴。
3. **`scale` 乘 q 且只乘一次**：默认 `1/sqrt(K)`；变异「q 不乘 scale」「默认当 1.0」「o 端补除」都被 default_scale / 随机对拍区分——`scale-after`（删除乘法）与 `o-no-scale`（读时补除）在浮点上不可交换（`q*scale` 后 h 内积 ≠ 内积后除，舍入不同），数值差异在 1e-4 容差内可判。
4. **初态与门控的复合**：`g=0、β=0` 时 `o_t = q_t·scale·h_0` 全程不变、终态等于初态——这是最干净的机制锚点；`initial_state` 不 float32 化会被 dtype 账抓。
5. **输出布局**：内部按 `[B,H,T,·]` 递推，`o` 需 transpose 回 `[B,T,H,V]`——oracle 初版转置序写错（V/H 颠倒）在 V>1 的随机用例上才暴露，说明固定小维度测试必须覆盖 V≠H。

## 变异测试（8 有效全抓 + 1 等价）

丢衰减、衰减后移、丢 β、β 移到 k（等价）、丢 delta 修正、q 不乘 scale、默认 scale 当 1.0、丢初态、o 端补除 scale。

## 复杂度

O(B·H·T·K·V) 递推，规模 ≤ 2×3×6×4×4，毫秒级。
