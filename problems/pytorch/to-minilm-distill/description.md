# MiniLM：自注意力分布与关系表示蒸馏

## 题目描述

**MiniLM**（Wang et al. 2020, *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*, arXiv:2002.10957, NeurIPS 2020）是一篇「小模型怎么学大模型」的经典工作。它的核心主张是：只对最后输出的 logits 做蒸馏不够，**自注意力分布本身就是知识**——教师每一个头「在看哪些 token」的模式，值得逐头迁移给学生。为此它提出两项损失：

1. **自注意力分布 KL**：把教师、学生同一个头的注意力分布（softmax 之后）用 KL 拉近，每个头、每个 query 位置各算一项；
2. **关系表示 MSE**：把若干对 token 之间的关系向量（如 CLS·CLS、CLS·token、token·token 的缩放点积表示）逐元素做均方误差。

两项直接相加得到总损失。本题给定**softmax 之前**的注意力打分与三组关系表示，要求还原这两项。

## 规则

给定 batch 大小 $B$、头数 $H$、序列长度 $L$：

1. **注意力分布**：打分除以温度 $\tau$ 后，在 **key 维（最后一维）** 做 masked softmax。`mask=False` 的 key 概率**严格为 0**（用 $-\infty$ 填充再 `log_softmax`，或等价的掩码方式）。
2. **逐 (b, head, query) 的 KL**：
   $$\mathrm{KL}\big(p_t\,\|\,p_s\big)=\sum_j p_t(j)\big(\log p_t(j)-\log p_s(j)\big),$$
   两侧各按自己的 softmax 归一，但用**同一个 mask**。
3. **注意力损失** = 所有**有效 query** 与所有头的 KL 之和，除以（有效 query 数 × $H$）。`mask=False` 的 query 整行不进分子也不进分母。
4. **关系损失** = 三组关系表示差值的平方，对全部 $B\times3\times D$ 元素取均值。
5. **总损失** = 注意力损失 + 关系损失（权重 1:1）。

返回三个 0 维 float64 标量：`attn_kl`、`cls_mse`、`loss`。

## 示例

```
teacher_attn[0,0] = [[1.0, 0.0], [0.0, 1.0]]   student_attn[0,0] = [[0.0, 1.0], [1.0, 0.0]]
teacher_attn[0,1] = [[0.5, 0.5], [1.0, 0.0]]   student_attn[0,1] = [[0.0, 0.5], [0.5, 0.0]]
mask = [[True, True]]      τ = 1.0

teacher_cls = [[[1.0], [0.0], [-1.0]]]
student_cls = [[[0.0], [0.0], [ 1.0]]]

attn_kl = 0.245377      # 4 组 (query, head) KL 的平均
cls_mse = 1.666667      # (1² + 0² + 2²) / 3
loss    = 1.912044
```

头 0 的两个 query 上，教师与学生的注意力分布正好互为镜像，各贡献 $\ln 2$；头 1 的分布差异更小，于是四项平均后落在 0.245 附近。

## 约束

- 返回的三个量都是 0 维 float64；输入可以是 float32，内部升到 float64。
- `mask` 同时作用于 query 与 key：被 mask 的 key 概率严格为 0；被 mask 的 query 整行不计入平均。
- KL 的方向是教师在前（$p_t\|p_s$），方差项是 `(teacher - student)²`；写成相反方向会得到不同数值。
- 非法输入抛 `ValueError`：非张量 / 注意力非四维或非方阵 / 非浮点 / 含 NaN·Inf / 形状不一致 / mask 非 bool 或形状不符 / mask 全 False / 关系表示形状不是 $(B,3,D)$ 或与 $B$ 不符 / $D<1$ / $\tau\le0$。
- CPU、确定性、不调用随机数。
