# 解题说明 · CriticalKV

## 动机：attention 高 ≠ 信息量大

ScorerPress 家族（knorm、observed/expected attention、SnapKV）全部只用 **key 侧或 attention 侧**信号。CriticalKV（arXiv:2502.03805）指出：一个 token 对最终 logits 的实际影响是 `attn_t · (v_t W_o)`——attention 大但 value 投影后接近 0 的 token（各 head 输出相互抵消）其实可以被安全驱逐。因此它把 \(\|v_t W_o\|_1\)（按 GQA 组内 query head 平均）乘到基础分上重排。题目里该范数直接作为输入给出，避开真实模型权重；其数学定义即 `CriticalKVPress.vwl1norm`：对每个 query head 计算 `‖v @ Wo_head‖₁`，再在组内取均值。

## 两阶段为何存在

若直接全量换成加权分，基础分极高（attention 极大）的 token 可能被一个小的 \(\|vW_o\|_1\) 误杀。上游实现（kvpress `criticalkv_press.py`，`CriticalKVPress.score`）用两阶段折中：

```python
selection_budget = int((1 - ratio) * k_len * first_stage_ratio)   # 阶段1只吃预算的一个分数
if first_stage_ratio > 0:
    selection_budget = max(1, selection_budget)                   # 短上下文防呆：至少保 1 个
top_k_index = torch.topk(scores, selection_budget).indices
scores = (scores + epsilon) * projected_norm                      # 阶段2重新计分
scores.scatter_(-1, top_k_index, torch.finfo(scores.dtype).max)   # 阶段1位置 → 无穷大保底
```

注意 scatter 发生在**乘完之后**：阶段 1 的 token 完全绕过 value 加权，这就是「保底」的语义。最终由外层 ScorerPress 流程对返回的 scores 做全局 top-k（等价于本题的第 5 步）。

## 与 ObservedAttention / ExpectedAttention 的区别（重点）

- `kv-observed-attention`（order 149）：score = 历史 attention 列和 ÷ (L−j)，纯 key 侧累计信号，**不看 value**。
- `kv-expected-attention`（order 150）：对 query 的高斯分布取期望，softmax(μ·k/√d + kᵀΣk/2d)×‖v‖，只用 value 的 **L2 范数**做标量。
- 本题 CriticalKV：不是一个新的 attention 估计器，而是**包装器**——先拿任意基础分（可以是上面两者），再乘 \(\|vW_o\|_1\)。区别在“通道相关性”：‖v‖₂ 只看 value 大小；\(\|vW_o\|_1\) 看 value 在**输出基**下的逐通道贡献绝对和，一个 ‖v‖₂ 很大但与 W_o 零空间对齐的 value 会被前者高估、被后者正确判低。

## 平局与确定性

上游 `torch.topk` 不承诺平局语义；本题规范化为 (−score, position) 字典序，输出升序。实现时用稳定排序（如 `torch.sort(-x, stable=True)`）或显式 key 元组即可与 oracle 一致。另外 epsilon 不是装饰：`(scores + eps)` 保证基础分为 0 的 token 在加权后仍是 0（不被负分翻号），且给极端分布留数值余量。

## 退化情形清单

- `first_stage_ratio=0`：无阶段 1，等价纯 \((s+\varepsilon)\cdot n\) top-k。
- `wov_norm ≡ 0`：非保护 token 最终分全为 0，保留集 = 阶段 1 ∪（剩余名额按最小位置填充）。
- `first_stage_ratio=1`：\(b_1=n\_kept\)，保留集恰为原始分的 top-\(n\_kept\)，value 加权完全不生效（可作为对拍不变量）。
- `compression_ratio=0`：\(n\_kept=L\)，返回全部位置。
