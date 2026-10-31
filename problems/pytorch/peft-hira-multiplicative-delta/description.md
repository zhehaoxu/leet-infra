# HiRA 乘性高秩适配：W⊙(BA) 与合并语义

## 题目描述

LoRA 系适配器的更新是**加性**的：W' = W + BA，两个低秩矩阵的乘积直接加在基权重上。HiRA（huggingface/peft `tuners/hira`）换了一种耦合方式——把 BA 当成**调制矩阵**逐元素乘上去：

$$W' = W \odot (1 + BA) = W + W \odot (BA)$$

前向时有效权重是 $W_{\text{eff}} = W \odot (BA)$， HiRA 残差 = $x W_{\text{eff}}^T$——**不是** base 输出加 $x(BA)^T$。合并时上游代码写的是 `merged_weight = weight + base_weight_snapshot * delta`，由于快照就是当前基权重，语义即 $W(1+BA)$。这一字之差让 HiRA 保留了基权重的完整秩（"high-rank adaptation" 的由来），也意味着：W 的某个元素为 0 时，无论 BA 多大，那个位置永远不动。

本题给定权重、两个低秩因子和输入，交出 Δ、有效权重、HiRA 残差与合并结果四份量，并处理 `fan_in_fan_out` 布局（HuggingFace Conv1D 把权重存成 `[in, out]`）。

## 任务

实现 `solve(weight, hira_A, hira_B, x, fan_in_fan_out=False)`：

- `weight`：二维浮点 tensor。`fan_in_fan_out=False` 时形状 `[out, in]`；`True` 时 `[in, out]`（Conv1D 存储，数学权重为其转置）；
- `hira_A`：`[r, in]`，`r >= 1`；`hira_B`：`[out, r]`；
- `x`：至少一维的 tensor，末维等于 `in`；可为批量 `[batch, in]` 或单 token `[in]`；
- `fan_in_fan_out`：bool。

返回 `dict`，键为 `delta_weight`（`B@A`）、`effective_weight`（`W⊙(B@A)`）、`hira_result`（`x @ effective^T`，末维 `out`）、`merged_weight`（`W + W⊙(B@A)`）。**四个权重键全部按输入的存储布局返回**：`fan_in_fan_out=True` 时为 `[in, out]`（即数学布局的转置），`hira_result` 不受影响。

## 精确规则

1. 还原数学权重：`W = weight`（fio=False）或 `weight.T`（fio=True），形状恒 `[out, in]`。
2. `delta = B @ A`，`[out, in]`。
3. `effective = W ⊙ delta`（逐元素，不是矩阵乘也不是加）。
4. `hira_result = x @ effective.T`，形状 `[..., out]`。
5. `merged = W + effective`（等价 `W⊙(1+delta)`）。
6. fio=True 时 `delta/effective/merged` 转置回 `[in, out]` 后返回。

## 示例

```text
W = [[1, 2, 0.5], [0, -1, 2]]（out=2, in=3）
A = [[1, 0, -1], [0, 1, 1]]（r=2）
B = [[1, 2], [0, -1]]
x = [[1, 1, 1]]

B@A = [[1, 2, 1], [0, -1, -1]]
W⊙(B@A) = [[1, 4, 0.5], [0, 1, -2]]
hira_result = x @ eff.T = [[5.5, -1.0]]
merged = W + W⊙(B@A) = [[2, 6, 1], [0, 0, 0]]
```

两个语义锚点：

- 把 `W` 换成全零矩阵：Δ = I、x = I 时 `effective/merged/hira_result` 全为零——**乘性**语义下 W=0 冻结一切；加性 LoRA 会给出 BA=I。
- `W = [[2,0],[0,2]]`、Δ = [[1,1],[1,1]]：merged = `[[4,0],[0,4]]`——非对角位置因 W 为 0 而保持 0；加性合并会得到 `[[3,1],[1,3]]`。

## 约束

- 所有输入为有限值；不修改输入；非法输入（形状不匹配、fio 非 bool、fio 与 weight 形状矛盾、x 末维不匹配、零维 x、非有限值）抛 `ValueError`。
