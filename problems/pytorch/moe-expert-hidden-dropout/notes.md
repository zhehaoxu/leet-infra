# 参考做法：dropout 的位置、inverted scaling 与门控重归一化

## 实现

1. 稳定 softmax：`shifted = logits - logits.max(dim=1, keepdim=True).values`，`scores = exp(shifted) / exp(shifted).sum(1, keepdim=True)`。
2. `expert_idx = torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :k]`；并列口径「编号小者优先」由 `stable=True` 承担，纯 Python 侧对应 `sorted(range(E), key=lambda e: (-scores[e], e))[:k]`。
3. `raw = scores.gather(1, expert_idx)`，`gates = raw / raw.sum(1, keepdim=True)`。注意 gather 的是 **softmax 分数**，重归一化只发生在选中的 k 项之间。
4. 用 `w1[expert_idx]` 得到 `[T,k,D,H]`，`pre = einsum("td,tkdh->tkh", x, ...) + b1[expert_idx]`，`hidden = relu(pre)`。
5. dropout：

   ```python
   if training and float(expert_dropout) > 0.0:
       mask = drop_mask.gather(1, expert_idx.unsqueeze(-1).expand(-1, -1, pre.shape[-1]))
       hidden = hidden * mask / (1.0 - float(expert_dropout))
   ```

   用 `gather` 而不是 `drop_mask[:, expert_idx, :]`，后者在 `[T,E,H]` 上做花式索引会先广播成 `[T,k,E,H]` 再挑，形状对但内存开销大；两者数值等价。
6. `expert_out = einsum("tkh,tkhd->tkd", hidden, w2[expert_idx]) + b2[expert_idx]`，`output = (gates.unsqueeze(-1) * expert_out).sum(dim=1)`。

常见错误：

- **dropout 打错了张量**：作用在输入 `x` 或专家输出 `y` 上，而不是专家隐藏激活上。这是本题真正的语义边界，判题在 `hidden` 与 `output` 上都会抓。
- **用非 inverted scaling**（训练不缩放、推理时乘 `1-p`）。本题规定训练态乘 `1/(1-p)`、推理态恒等；这种写法下 `check_dropout_scaling` 的全 1 mask 会精确差一个 `1/(1-p)`。
- **推理态仍套用 mask**。`training=False` 时 `drop_mask` 必须被完全忽略——`check_inference_asymmetry` 用全 0 mask + `p=0` 比较恒等性，且对 `training=False` 扫 `p∈{0,0.3,0.9}` 要求逐位相同。
- **漏掉 `p=0` 短路**。`p=0` 且 `training=True` 时若仍然乘 `1/(1-p)=1`，数值上没问题，但会把全 0 的 `drop_mask` 也套上去、把激活清零；`check_inference_asymmetry` 里「全 0 mask + p=0 + training=True」这一条正是为它准备的。
- **先重归一化再选 top-k**，或**把未选中专家的 softmax 质量也算进分母**。门控是「选 k 个 → 在这 k 个上归一化」，顺序和分母都不能换。
- **并列用 `torch.topk` 但不稳定**。契约要求编号小者优先；`torch.topk` 在并列上不保证顺序，判题用稳定排序的期望值。
- **忘记 `p=0` 时关闭 dropout**。`p=0` 且 `training=True` 时乘 `1/(1-0)=1`，数学上恰好恒等，但 `mask` 全 0 时若不短路也会把激活清零；本题规定 `p=0` 一律忽略 mask。

## 正确性检验

7 个 `check_*` 覆盖：手算示例（训练/推理两态同一 mask、输出不同）、10 组随机对拍（`T/E/D/H/k/p/training` 全网格，含非连续输入）、训练/推理不对称（`training=False` 扫 `p` 逐位不变、`training=True` 且 `p=0` 时全 0 mask 也不得改变结果）、dropout 缩放（全 1 mask 让 hidden 精确变成 `2×`、全 0 mask 让 hidden 归零且输出退化为 `sum_j gates_j * b2[e_j]`）、门控（`k=E` 时 gate 恰等于完整 softmax、`k=1` 时 gate 恒为 1、三路并列按编号取 0/1/2）、单 token / 单专家 / 宽度 1 的退化形状、非法输入。

oracle 完全在纯 Python 标量上跑：`math.exp` 手写稳定 softmax、`sorted(key=lambda e: (-scores[e], e))` 选择、`math.fsum` 做点积与加权和，不借用 torch 的 `softmax`/`sort`/`einsum`。

变异测试：把 dropout 改到输入 `x` 上、改到专家输出上、去掉门控重归一化、推理态仍套 mask、用 `(1-p)` 代替 `1/(1-p)` —— 都被抓到；「漏掉 `p=0` 短路」只被 `check_inference_asymmetry` 抓到，说明那条用例是必要而非冗余的。有两个 mutant 存活，且都不是漏测：`torch.topk` 顶替稳定排序时，随机输入上没有精确并列，恰好一致，而题目约束的是并列结果、不锁定 API；dropout 挪到 relu **之前**在本题契约下数值等价——正数缩放与 relu 可交换（`relu(c·a) = c·relu(a)`，`c>0`），无论 mask 取 0 还是 1 结果都一样，所以这个「位置」并不构成可区分的语义差异，真正的边界是 dropout 必须落在专家隐藏向量上。

## 范围与来源

题目对应 Google 的 ST-MoE（Zoph et al., *ST-MoE: Designing Stable and Transferable Sparse Expert Models*, arXiv:2202.08906）。论文在「expert dropout」的定义上写得很明确：它是在**专家隐藏激活**上、与全局 dropout 相互独立的一个 dropout rate。论文还报告尝试过「整个专家被 mask 掉」的变体，结果没有提升泛化，因此没有采用——所以「丢弃整专家再 renorm」不是这个机制的语义，本题不去复刻它。

上游 Mesh TensorFlow 的实现（`mesh_tensorflow/transformer/moe.py`）里，专家前向的隐藏层之后是：

```text
h = mtf.dropout(h, is_training=train, keep_prob=1.0 - hparams.moe_dropout_rate)
```

即 dropout 作用在第二个投影之前的激活上，用 `keep_prob` 控制保留率，训练态生效、推理态关闭。Mesh 的 `mtf.dropout` 默认走 inverted scaling，本题把「训练态乘 `1/(1-p)`、推理态恒等」写进契约，避免依赖具体框架的默认值。

本题的口径与上游的差异：mask 是显式输入而不是 RNG；并列顺序写死为「编号小者优先」以便判题可复现；门控重归一化在这里是 top-k 稀疏门控的常规做法（与 dropout 无因果绑定），题目把它显式化。本题不涉及容量分派（见 `moe-capacity-jitter-dispatch`）、不做负载统计（见 `moe-dead-expert-ledger`）、不实现门控偏置更新或负载辅助损失，也不做反向传播。
