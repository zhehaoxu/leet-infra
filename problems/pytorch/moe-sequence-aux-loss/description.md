# 序列级负载均衡损失

## 任务

实现 `solve`：根据逐 token 专家亲和度和**已给定的** top-k 选择，计算每条序列的专家负载与平均概率，再求序列级辅助损失。同时返回先合并整个 batch 后计算的损失，以展示两种均值次序的差异。

本题只计算损失数值，不选择专家、不求梯度、不执行优化器更新。它不是专家总重要性的 CV 损失，也不是把整个 batch 当成一条长序列的 Switch 统计。

## 输入

- `scores`：CPU float64 张量 `[B,T,E]`，为**非负亲和度，不是 logits**。每个值在 `[0,1]`，每行专家和至少为 `1e-6`。softmax 概率或 sigmoid 亲和度都可作为输入。
- `selected_ids`：CPU int64 张量 `[B,T,K]`，每个 token 的 K 个编号互不重复且在 `[0,E)`。
- `alpha`：Python 浮点数，范围 `[0,2]`，默认 `1.0`。
- `B=1..4`，`T,E=1..8`，`1 <= K <= E`。每条序列具有相同有效长度 T，无 padding。
- 输入均合法，允许非连续张量，不修改输入。

选择可以来自带偏置或其他受限路由器，**不保证是 scores 的 top-k**，不得重新计算选择。

## 精确公式

先在每个 token 内对全部 E 个专家归一化，而不是只对选中专家归一化：

$$
p_{b,t,e}=\frac{s_{b,t,e}}{\sum_{j=0}^{E-1}s_{b,t,j}}.
$$

定义序列 b 中专家 e 被选中的次数 $c_{b,e}$，并计算：

$$
f_{b,e}=\frac{E}{TK}c_{b,e},\qquad
P_{b,e}=\frac{1}{T}\sum_{t=0}^{T-1}p_{b,t,e}.
$$

因此 $\sum_e f_{b,e}=E$，$\sum_e P_{b,e}=1$。f 是相对均衡负载，不是和为 1 的分布。

$$
\ell_b=\alpha\sum_e f_{b,e}P_{b,e},\qquad
L_{\mathrm{seq}}=\frac1B\sum_b\ell_b.
$$

对照项先跨序列平均，再相乘：

$$
L_{\mathrm{batch}}=\alpha\sum_e
\left(\frac1B\sum_b f_{b,e}\right)
\left(\frac1B\sum_b P_{b,e}\right).
$$

两项一般不相等。不得把 `alpha` 乘入 f 或 P；`alpha=0` 时仍需返回正确统计。

## 返回

返回五项 `tuple`：`(f, p, per_sequence, sequence_loss, batch_loss)`。

- `f`、`p`：形状 `[B,E]`，分别对应 $f,P$。
- `per_sequence`：形状 `[B]`，对应 $\ell$。
- `sequence_loss`、`batch_loss`：零维张量，分别对应 $L_{\mathrm{seq}},L_{\mathrm{batch}}$。
- 五项均为 CPU float64 张量；按 `rtol=2e-11, atol=2e-11` 比较。

## 示例

```python
scores = [[[0.9, 0.1], [0.9, 0.1]],
          [[0.1, 0.9], [0.1, 0.9]]]
selected_ids = [[[0], [0]], [[1], [1]]]
alpha = 0.1
```

输出按张量的内容表示：

```python
f = [[2.0, 0.0], [0.0, 2.0]]
p = [[0.9, 0.1], [0.1, 0.9]]
per_sequence = [0.18, 0.18]
sequence_loss = 0.18
batch_loss = 0.1
```

整个 batch 看似均衡，但每条序列都集中在单一专家。如果 `K=E`，则所有 f 都为 1，两种损失都等于 `alpha`。
