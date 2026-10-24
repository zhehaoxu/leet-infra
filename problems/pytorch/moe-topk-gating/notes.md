# 解题说明 · MoE Top-k Gating

## 为什么先 softmax 再 topk

有些实现是「先 topk 再对 logits 做 softmax」，两种顺序给出的最终 scores **不同**：
- 先 softmax 再 renorm（本实现）：renorm 的分母只是 $k$ 个选中概率的和，$\sum_j s_j = 1$。
- 先 topk 再 softmax：把 $k$ 个原始 logits 直接 softmax，等价于在选中集合内做带偏置的分配。

HF `MixtralTopKRouter.forward` 采用**先 softmax 再 topk 再 renorm**，判题按这条路径。

## tie-break

`torch.topk` 对相等值的顺序**不保证稳定**（不同 backend / 版本可能不同）。测试样例避免 logits 严格并列；若必须处理并列，按「较小专家索引优先」：
```
order = sorted(range(E), key=lambda e: (-float(probs[t,e]), e))
```

## renorm 的意义

top-k 之后 $k$ 个概率之和 $<1$（长尾被丢弃）。若不 renorm 直接作为 combine 权重，MoE 输出会**系统性地小于** dense 基线；renorm 保证每 token 的最终加权和仍等价于「按选中的 $k$ 个专家做一次加权平均」。

## 与 softmax-with-temp 的区别

dense FFN 用 softmax 是因为「所有神经元都算」；MoE 只用被激活的 $k$ 个神经元，softmax 只是提供**一个初始打分**，真正的权重是 renorm 之后的 $k$ 维分布。因此 top-k gating 的输出**总是每行和为 1**。
