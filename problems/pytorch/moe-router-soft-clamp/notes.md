# 参考做法：限幅的位置、偏置的隔离与饱和造成的并列

## 实现

1. `clamped = torch.tanh(logits / tau) * tau`（`tau > 0`），`tau == 0` 时直接透传 `logits`，不要写成除以 0。
2. `scores = torch.sigmoid(clamped)`。
3. `selection = scores if bias is None else scores + bias`。
4. `indices = torch.argsort(selection, dim=-1, descending=True, stable=True)[:, :k]`。并列口径「编号小者优先」是由 `stable=True` 承担的，纯 Python 侧对应 `sorted(range(E), key=lambda i: (-sel[i], i))`。
5. `picked = scores.gather(-1, indices)`——注意从 **scores** 取，不是从 selection 取。
6. `total = picked.sum(-1, keepdim=True)`；`weights = torch.where(total > 0, picked / total.clamp_min(1e-300), full_like(picked, 1.0 / k))`。这里的 `clamp_min` 只是为了让 `where` 的假分支不真的去算 0/0，真分支的数值不受影响。

常见错误：

- **限幅写在 sigmoid 之后**。`tau*tanh(z/tau)` 限的是 logits，落到概率上就变成了把 $s\in(0,1)$ 再压一次，`tau` 的语义完全不同；判题的 `-800/-900` 下溢用例正是用来区分这两种写法。
- **忘了先除以 `tau`**，写成 `tanh(z) * tau`：限幅上界变成 $\pm\tau$ 但归一化系数没跟着走，饱和点也错了。
- **归一化用 `s + bias`**。偏置是负载均衡的补偿项，把它带进权重就等于改了专家输出的实际系数，正是 aux-free 路由反复强调要避免的事。
- **忘了 `tau = 0` 分支**，让 `logits / 0` 变成 inf 再乘 0 得到 NaN。
- **没有下溢退路**：`tau = 0` 且 logits 里有 `-800` 这种值时，sigmoid 会精确给 0，选中项之和为 0，`picked / total` 得到 NaN 而不是契约要求的 `1/k`。

## 正确性检验

7 个 `check_*` 覆盖：手算示例（含「加 bias 会换人但权重仍由 `s` 归一化」）、64 组随机对拍（seed × tau × 有无 bias 全网格，`k` 和 logit 尺度随 seed 变）、饱和与并列（`tanh(20)==tanh(21)`、`tau=1e6` 与关闭限幅同选择、`tau=1e-3` 时由 bias 决定选择）、bias 隔离（把专家 1 抬进选择、加常数 bias 不改结果）、边界与下溢（`k=E`、全零并列、`[-800,-900,5]` 分数为 0 且权重 `[1,0]`、`[-800,-900]` 给均匀 `1/k`、非连续输入）、非法输入、最大形状 `8×8, k=8`。

oracle 完全在纯 Python 标量上跑：`math.tanh`、自己写的稳定 sigmoid（负数走 `exp(z)/(1+exp(z))`，否则 `math.exp(800)` 会溢出）、`sorted(key=lambda i: (-sel[i], i))` 选择、`math.fsum` 归一化，不借用 torch 的 `sigmoid`/`argsort`/`gather`。

变异测试：把限幅挪到 sigmoid 之后、漏掉除以 `tau`、漏掉 `tau=0` 分支、把 bias 混进权重、从 selection 里 gather、去掉下溢退路——六个 mutant 全部被抓到，其中「去掉下溢退路」只被 `edges_and_underflow` 抓到，说明那个用例是必要而非冗余的。另外两个 mutant（不带 `stable=True` 的 `argsort`、以及 `torch.topk`）在本机的判题环境里与契约给出的顺序恰好一致，因此测不出差别；这不是漏测，题目约束的是「并列取小编号」这个结果，不锁定具体用哪个 API。

## 范围与来源

机制直接对应判题环境里 transformers 5.x 的 `models/laguna/modeling_laguna.py::LagunaTopKRouter.forward`：

```text
if self.router_logit_softcapping > 0.0:
    router_logits = torch.tanh(router_logits / self.router_logit_softcapping) * self.router_logit_softcapping
routing_scores = torch.sigmoid(router_logits)
scores_for_selection = routing_scores + self.e_score_correction_bias...
_, selected_experts = torch.topk(scores_for_selection, self.top_k, dim=-1)
routing_weights = routing_scores.gather(-1, selected_experts)
routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
```

本题把这条链路抽成显式规则：软上限用参数 `tau` 表示，`tau = 0` 等价于配置里的「关闭软限幅」；偏置只进 `scores_for_selection`，`gather` 从 `routing_scores` 取。同样的 `tanh(z/cap)*cap` 形态也出现在 gemma2 的 `final_logit_softcapping` 与注意力 `score_mod` 里，说明它是常见的稳定性手段，只是作用位置不同。

真实实现用 `torch.topk`，本题把并列顺序写死为「编号小者优先」，因为判题需要可复现的期望值；下溢时给均匀 `1/k`、`tau=0` 关闭限幅、不实现偏置更新，都是题目口径而不是上游代码的默认行为。本题不涉及分组 top-k（见 `moe-aux-free-routing`）、不涉及偏置的符号更新、不涉及 z-loss 或负载辅助损失，也不做前向以外的任何训练动作。
