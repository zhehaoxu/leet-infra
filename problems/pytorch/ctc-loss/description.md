# CTC 损失（前向算法）

## 题目描述

连接时序分类（Connectionist Temporal Classification, Graves et al., ICML 2006）让 RNN/Transformer 无需对齐就能输出变长序列。它约定一个空白符 `blank`，把一条逐步发射路径经「合并连续重复、再删除 blank」后映射到目标标签。给定每条时间步各类别的对数概率，CTC 损失是所有能坍缩到目标标签的路径概率之和的负对数：

\[ \mathcal{L} = -\log \sum_{\pi:\,\mathcal{B}(\pi)=\ell} \prod_{t} y_{t,\pi_t} \]

直接用动态规划（前向变量 \(\alpha\)）在 log 空间累加，避免指数级枚举路径。

## 规则

实现 `solve(log_probs, labels)`，返回一个 0 维标量张量（即负对数似然 \(-\log p(\ell\mid x)\)）。`log_probs` 为 float32、形状 \((T, V)\) 的**对数**概率（每行是 softmax 后取 `log`），空白符固定为最后一类 `blank = V-1`；`labels` 为 int64、形状 \((L,)\) 的目标类别下标（不含 blank）。用标准 CTC 前向算法：把标签扩展成 \(\ell'=[\text{blank},\ell_1,\text{blank},\ell_2,\ldots,\text{blank}]\)（长 \(L'=2L+1\)），逐时间步递推 \(\alpha_t(k)\)，重复类相邻时可不经过 blank 直接跳转；终态取 \(\alpha_T(L'-1)\) 与 \(\alpha_T(L'-2)\) 之和，返回其相反数。

## 示例

`probs=[[0.3,0.6,0.1],[0.5,0.2,0.3],[0.1,0.4,0.5]]`（\(T=3, V=3\)，`blank=2`），`log_probs=log(probs)`：

```
labels=[0]    -> 1.78379130
labels=[0,1]  -> 1.77195680
```

## 约束

- CPU float32；`log_probs` 每行是某 softmax 分布的对数（含负值）。
- `blank = V - 1`；`labels` 取值范围 \([0, V-2]\)（不含 blank）。
- 递推须在 log 空间用稳定的 log-add-exp 合并，避免下溢。
- 返回值是标量负对数似然；标签为空（\(L=0\)）时损失等于「全 blank 路径」的负对数。
