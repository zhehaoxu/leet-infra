# order 843 · Captum LRP 的 z-rule relevance 传播账

> 官方源：`pytorch/captum`（main 分支）—— `captum/attr/_core/lrp.py`、`captum/attr/_utils/lrp_rules.py`

## 背景

Layer-wise Relevance Propagation（LRP）把「模型输出对某个目标的贡献」沿网络逐层**分配**到输入神经元。
它与 Integrated Gradients（order 842）的路径积分思路完全不同：IG 沿直线路径**积分**梯度，
LRP 则是**自上而下逐层重分配**。

Captum 的 LRP 实现有两个容易被忽略的关键点，本题把它们做成必答：

1. **两次前向**：官方先用一个 forward 记录每层激活并改写权重，再用**改写后的权重 × 第一次的激活**重算一遍。
2. **`sign(0) = +1`**：z-rule 的分母是 `outputs + sign(outputs) * eps`，官方把 `sign(0)` 强制成 `+1`，
   所以输出恰好为 0 时分母是 `+eps` 而不是 0。

本题要求把这套传播账**用精确有理数**重建出来。

---

## 口径（本题的唯一标准）

- **所有输入量都是整数**（权重、偏置、激活、输入向量）。
- **`eps` 取精确有理数 `1/10^9`**（官方是 float64 的 `1e-9`；本题把被浮点掩盖的语义显式化）。
- 全链路用 `fractions.Fraction` **精确**计算，只有最终输出时转 `float`（每个输出量只转一次）。
- 浮点输出的判定用相对容差 `1e-12`（这个容差只用来吸收「精确有理 → float」的一次舍入，
  以及官方 float64 自身的 ~1e-16 尾差；任何语义错误都会被放大到远超 1e-12）。
- 除法前后**不要**引入浮点中间量：一旦某一步用了 float，尾差会顺着传播链放大。

---

## mode 1 · `rewrite_rule`

官方 `PropagationRule._manipulate_weights` 的权重改写账。

**入参**

| 参数 | 类型 | 说明 |
|---|---|---|
| `rule` | str | `"epsilon"` / `"gamma"` / `"alpha1_beta0"` |
| `weights` | `[[int]]` | 权重矩阵（行列表） |
| `bias` | `[int]` | 偏置向量（长度 = 权重行数） |
| `gamma` | str | 有理数字符串，如 `"1/4"`、`"1/2"`、`"0"`、`"1"`（仅 `gamma` 规则用） |
| `set_bias_to_zero` | bool | 是否把偏置清零（默认 `False`） |

**出参**：`{"weights": [[float]], "bias": [float]}`

**三条规则的语义**

- `epsilon`：**权重与偏置都原样返回**。
  官方 `EpsilonRule._manipulate_weights` 的函数体是 `pass`——它只改 `STABILITY_FACTOR`，不动权重。
- `gamma`：`W'[i][j] = W[i][j] + gamma * max(W[i][j], 0)`
  注意是「**加上** gamma 倍的正部」，等价于「正权重乘 `(1 + gamma)`，负权重不变」。
  **负权重不会被放大，也不会被缩小。**
- `alpha1_beta0`：`W'[i][j] = max(W[i][j], 0)`（负权重清零，即 Deep-Taylor 规则）

`set_bias_to_zero=True` 时，偏置全部置 0（权重仍按上表改写）。

**非法 `rule` 必须抛 `ValueError`。**

---

## mode 2 · `z_rule_layer`

单层线性层的 z-rule relevance 分解。

**入参**：`weights`（`[[int]]`，**已经改写后**的权重）、`bias`（`[int]`）、`inputs`（`[int]`，该层的输入激活）、`target`（int，目标输出下标）

**出参**：`{"outputs": [...], "relevance_out": [...], "relevance_in": [...], "attribution": [...]}`

**计算步骤**

1. 前向：`outputs[i] = sum_j W[i][j] * inputs[j] + bias[i]`
2. 上游梯度是 one-hot：`grad = e_target`
3. **output hook**（逐元素）：
   ```
   relevance_out[i] = grad[i] / (outputs[i] + sign(outputs[i]) * eps)
   ```
   其中 `sign(v)`：`v > 0` → `+1`，`v < 0` → `-1`，`v == 0` → **`+1`**（官方 quirk）。
4. **input hook**：
   ```
   relevance_in[j] = (relevance_out @ W)[j] * inputs[j]
   ```
   即先把 `relevance_out` 当行向量右乘权重矩阵，再逐元素乘 `inputs`。
5. 归一化输出：`attribution[j] = relevance_in[j] * outputs[target]`

**`target` 越界必须抛 `ValueError`。**

---

## mode 3 · `stack_relevance`

多层线性栈的完整传播账。这是本题的核心。

**入参**

- `layers`：`[{weights, bias, rule, gamma?, set_bias_to_zero?}, ...]`，至少一层；层与层首尾相接
  （第 `i` 层的输入维度 = 第 `i-1` 层的输出维度）。
- `inputs`：`[int]`，栈的输入向量
- `target`：int，最顶层（输出层）的目标下标

**出参**

```
{
  "activations":    [[...]],   # 每层【第一次前向】的输出
  "outputs":        [[...]],   # 每层【第二次前向】的输出
  "relevance_out":  [[...]],   # 每层的 relevance_out
  "relevance_in":   [[...]],   # 每层的 relevance_in
  "attribution":    [...],     # 最输入层的 relevance_in × 原权重下的 target 输出
}
```

**所有列表都按前向顺序**（下标 0 = 靠输入层的第一层）。

**两次前向的准确语义**

1. **第一次前向**：用**原始权重**（`W`、`b`，未改写）链式前向，
   得到 `activations[k]` = 第 `k` 层的输出。
   同时，第 0 层的输入就是 `inputs`，第 `k` 层的输入是 `activations[k-1]`。

2. **改写权重**：按 mode 1 的语义得到每层的 `W'`、`b'`。

3. **第二次前向**：第 `k` 层的输出是
   ```
   outputs[k][i] = sum_j W'[k][i][j] * activations[k-1][j] + b'[k][i]
   ```
   **乘的是「对应层的原激活」，不是上一层的 `outputs`。**
   这是官方 `forward_pre_hook_activations` 的直接后果：第二次前向时每层的输入都被还原成
   第一次前向存下的激活。

   ⚠️ 写成「改写权重链式前向」是最常见的错误。

4. **反向传播**（从最顶层往最底层）：
   ```
   upstream = e_target                       # 只在最顶层
   rel_out[k] = upstream / (outputs[k] + sign(outputs[k]) * eps)
   rel_in[k]  = (rel_out[k] @ W'[k]) * activations[k-1]
   upstream   = rel_in[k]                   # 传给下一层
   ```
   注意 `rel_in[k]` 乘的是**原激活** `activations[k-1]`。

5. **归一化输出**：
   ```
   output_target = activations[-1][target]   # 【原权重】下的 target 值
   attribution[j] = rel_in[0][j] * output_target
   ```

**非法 `rule`、空 `layers`、`target` 越界都必须抛 `ValueError`。**

---

## mode 4 · `completeness`

完备性 delta。

**入参**：`attributions`（`[int]`，单样本的归因向量）、`output`（int，**原始权重**下的 target 输出值）

**出参**：`{"attr_sum": float, "delta": float}`

```
attr_sum = sum(attributions)
delta    = output - attr_sum
```

⚠️ **符号顺序**：官方 LRP 的 `compute_convergence_delta` 是 `output - attr_sum`（**output 在前**）。
这与 Captum 的 Integrated Gradients（order 842）**相反**——IG 是 `attr_sum - (end_out - start_out)`。

---

## 常见错误清单

| 错误 | 后果 |
|---|---|
| `gamma` 规则写成「全体权重乘 `(1+gamma)`」 | 负权重被错误放大 |
| `alpha1_beta0` 忘记清零负权重 | 负权重残留 |
| `epsilon` 规则也去改权重 | 权重被错误截断 |
| `sign(0)` 返回 0 | 输出恰为 0 时除零 / 分母错 |
| `eps` 用 `1e-8` / `1e-10` | 分母偏大偏小 |
| 第二次前向写成「改写权重链式」 | 每层 `outputs` 全错 |
| `rel_in` 乘 `outputs` 而不是原激活 | 权重改写的影响被抹掉 |
| `relevance_in` 用 `W`（原权重）而不是 `W'` | 改写规则的影响丢失 |
| `upstream` 每层重置为 one-hot | 跨层传播断裂 |
| `attribution` 乘 `outputs[-1][target]`（改写后） | 归一化基准错 |
| `delta` 写成 `attr_sum - output` | 符号反了 |
| 中途混用 float 与 Fraction | 尾差放大到超出 1e-12 |

---

## 自检

```python
from fractions import Fraction

EPS = Fraction(1, 10 ** 9)

# mode 1：gamma 只放大正权重
# W = [[-3, 5]]  ->  W' = [[-3, 25/4]]

# mode 2：sign(0) = +1
# W=[[1,1]], b=[-2], inputs=[1,1] -> outputs=[0]
# relevance_out = [1/eps]，attribution = [0, 0]

# mode 3：两次前向
# layer0 = gamma(1/4), layer1 = epsilon
# activations[1] = [44, -66, -38]（原权重链式）
# outputs[1]     = [44, -66, -38]（改写权重 × 原激活 a1，
#                                不是 × outputs[0]）
```
