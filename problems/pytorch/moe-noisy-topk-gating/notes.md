# 参考做法：加噪选择、选中项内 softmax

## 实现

1. 计算门控 logits `x @ gate_w.T`。
2. 训练态计算噪声尺度 `softplus(x @ noise_w.T)`，乘入已给定的 `noise` 样本后加到 logits；推理态完全跳过这两步。
3. 对加噪 logits 做降序、稳定的 top-k，完全相等时保留较小专家编号。
4. 只对选中的 k 个加噪 logits 做 softmax。
5. 把权重散射回 `[T,E]` 全零矩阵，未选位置严格为 0。

`k` 必须是真正的 `int`，因为 `bool` 是 `int` 的子类，不能直接接受。`softplus` 用 `logaddexp(0,z)` 计算，负输入同样得到正尺度，不能用绝对值代替。

## 正确性检验

15 个 `check_*` 覆盖：手算示例、零噪声退化、噪声改变选择、softplus 正负输入、仅选中项 softmax、`k=E` 退化为全专家 softmax、并列取小索引、推理忽略噪声、训练使用噪声、负尺度、非连续输入、零行、30 组随机标量 oracle、非法 k、最大形状。

oracle 用纯 Python 对每个 token、每个专家计算 logit、softplus 和排序，再做独立的选中项 softmax；不复用 `argsort` 或 `scatter` 路径。`k=E` 与推理态用于区分「噪声只影响 top-k 选择」和「噪声污染最终权重」两类错误。

## 范围与来源

机制来自 Shazeer 等人 2017 年《Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer》中提出的 noisy top-k gating：$H(x)_i=(xW_g)_i+\mathrm{Softplus}((xW_{noise})_i)\cdot\epsilon_i$，随后取 top-k 并 softmax。

本题把噪声样本作为显式入参，不依赖全局随机数生成器，因此可完全复现。题面固定了并列顺序和推理行为，不声称与任何框架的随机流、初始化或完整 MoE 层一致；也不包含负载均衡损失、容量约束或专家前向。
