# 知识蒸馏温度损失

## 题目描述

知识蒸馏（Knowledge Distillation, Hinton et al., arXiv:1503.02531）让「学生」网络去模仿「教师」网络输出的软标签。 softened 分布用温度 \(T\) 把 logits 拉平：\(q_i=\mathrm{softmax}(z_i/T)\)。蒸馏损失是师生软分布之间的反向 KL，并乘 \(T^2\) 以在反向传播时补偿梯度被 \(1/T\) 缩小的量级，使软/硬两路损失的相对权重不随 \(T\) 漂移。

\[ \mathcal{L}_{\text{KD}} = T^2 \cdot \mathrm{KL}\!\big(\text{softmax}(z_t/T)\,\big\|\,\text{softmax}(z_s/T)\big) \]

## 规则

实现 `solve(student_logits, teacher_logits, T, reduction="mean")`，返回标量（或逐样本向量）。`student_logits`、`teacher_logits` 为 float32、形状 \((N, C)\) 的未归一化打分，`T>0` 为温度。逐样本计算：把两侧 logits 各除以 \(T\) 后取 log-softmax，得 \(\log p_s,\ \log p_t\)；令 \(p_t=\exp(\log p_t)\)，逐样本 KL 为 \(\sum_c p_{t,c}(\log p_{t,c}-\log p_{s,c})\)，再乘 \(T^2\)。按 `reduction`（`"mean"`/`"sum"`/`"none"`）归约。`reduction` 非法须抛 `ValueError`。

## 示例

\(N=1, C=3\)，`student_logits=[[2.0,1.0,0.1]]`，`teacher_logits=[[1.0,2.0,0.5]]`：

```
T=2.0 -> 0.39207178
T=1.0 -> 0.40606672   （此时等于 softmax 分布间的 KL(P_t‖P_s)）
student==teacher -> 0.0
```

## 约束

- CPU float32；`T>0`（标量浮点）。
- KL 方向为 \(\mathrm{KL}(\text{teacher}\|\text{student})\)（学生向教师看齐）。
- 乘子 \(T^2\) 作用在逐样本 KL 之后、归约之前。
- `T=1` 时结果应等于对原始 softmax 分布求 \(\mathrm{KL}(p_t\|p_s)\)。
- 支持 `reduction` 三种取值，其它字符串抛 `ValueError`。
