# 动态温度采样 (Dynatemp)

## 题目描述

固定温度要么一直「冷静」要么一直「放飞」，可模型的自信程度是**逐步变化**的。
**Dynatemp**（llama.cpp 的 `--dynatemp-range/-exp`）让温度随分布的**熵**（不确定度）在
$[\,\text{temp}-\Delta,\ \text{temp}+\Delta\,]$ 窗口里滑动：分布越**平**（熵越高、模型越犹豫）
→ 温度越**高**（更果断地重新洗牌）；分布越**尖**（熵低、模型已有把握）→ 温度越**接近下限**。
指数 `exponent` 控制这条插值曲线的弯曲方向。

## 规则

- 实现 `solve(logits, temp, delta, exponent, u)`：`logits` 形状 `(..., V)`，`temp > 0`、
  `delta ≥ 0`、`exponent > 0` 为标量，`u` 取值 $[0,1)$（形状 `(...,)` 或标量）。
- 步骤（逐行，记 $n=V$）：
  1. `probs = softmax(logits)`。
  2. 若 `delta <= 0`：**禁用动态**，直接令 $\text{dyn}=\text{temp}$、`filtered = softmax(logits/temp)`。
  3. 否则：
     - $\text{max\_entropy} = -\ln(1/n) = \ln n$；
       $\text{entropy} = \sum_i -p_i \ln p_i$；
       $C = \text{entropy}/\text{max\_entropy} \in [0,1]$。
     - $T_{\min} = \max(0,\ \text{temp}-\Delta)$，$T_{\max} = \text{temp}+\Delta$；
       $$\text{dyn} = T_{\min} + (T_{\max}-T_{\min})\cdot C^{\text{exponent}}$$
     - `filtered = softmax(logits / dyn)`。
  4. 用 `u` 在 `filtered`（原顺序）CDF 上逆变换采样得 `chosen`。
- 返回 `(filtered, chosen, dyn)`：`filtered` 形状同 `probs`，`dyn` 形状 `(...,)`（每行的动态温度）。

## 示例

```
输入：logits = [2, 1, 0.5, -1, -2, -3]，temp = 1.0，delta = 0.3，exponent = 1.0，u = 0.5
  probs = [0.600294, 0.220836, 0.133944, 0.029887, 0.010995, 0.004045]
  entropy = 1.085951, max_entropy = ln 6 = 1.791759, C = 0.606081
  T_min = 0.7, T_max = 1.3 → dyn = 0.7 + 0.6 * 0.606081^1 = 1.063648
  filtered = softmax(logits / 1.063648) = [0.579178, 0.226207, 0.141368, 0.034506, 0.013477, 0.005264]
  CDF = [0.579178, ...]，u = 0.5 → 首项即 ≥0.5 → chosen = 0
输出：filtered = [0.579178, 0.226207, 0.141368, 0.034506, 0.013477, 0.005264], chosen = 0, dyn = 1.063648
```

## 约束

- float32 输入；`temp > 0`、`delta ≥ 0`、`exponent > 0`；`u ∈ [0,1)`
- `filtered` 每行和为 1（softmax 不删候选）；`chosen` 为逆变换命中位置
- `entropy` 用**自然对数**；`delta <= 0` 时 `dyn` 恒等于 `temp`
- 均匀分布 → $C=1$ → `dyn=T_max`；极尖分布 → $C\to0$ → `dyn\to T_min`
