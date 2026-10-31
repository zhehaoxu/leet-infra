# MoE：专家隐藏激活 dropout 与门控重归一化

## 任务

实现一次 top-k 稀疏 MoE 前向：先按 softmax 分数选专家并重归一化门控，再让每个被选中的专家跑一遍两层前馈，**dropout 只施加在专家的隐藏激活上**（激活函数之后、第二个投影之前），训练态用 inverted scaling、推理态完全关闭。

这是「专家隐藏激活 dropout」的确定性前向模型，对应 ST-MoE 里与全局 dropout 相互独立的那个 dropout。题面显式规定 mask 的语义、缩放系数、门控的重归一化位置和并列顺序；它不是任何训练框架的逐步复刻，不涉及随机数、反向传播或参数更新。

## 输入

- `logits`：CPU float64 `[T,E]`，路由 logits。
- `x`：CPU float64 `[T,D]`，token 输入。
- `w1`：`[E,D,H]`，`b1`：`[E,H]`；`w2`：`[E,H,D]`，`b2`：`[E,D]`。每个专家是 `D→H→D` 的两层前馈。
- `drop_mask`：CPU float64 `[T,E,H]`，元素只能是 `0.0`（丢弃）或 `1.0`（保留）。它是**显式输入**，题目内部不产生任何随机数。
- `k`：真正的 Python `int`，`1 <= k <= E`。
- `expert_dropout`：Python `int`/`float`，`0 <= p < 1`。
- `training`：Python `bool`。

`T,E,D,H >= 1`。所有张量必须 CPU float64 且有限；`drop_mask` 出现 0/1 以外的值、`bool` 冒充 `int`、`k` 越界、`p` 越界或为 NaN/±inf、`training` 不是 `bool`、形状不匹配——都抛出 `ValueError`。不修改输入，接受非连续张量。

## 精确规则

**门控。** 逐行 softmax：

$$
s_{t,e}=\frac{e^{z_{t,e}-\max_j z_{t,j}}}{\sum_{j}e^{z_{t,j}-\max_j z_{t,j}}}.
$$

按 $s_{t,e}$ 降序取 $k$ 个专家 $\pi_{t,1},\dots,\pi_{t,k}$，**完全并列时专家编号小者优先**。权重只在选中的 $k$ 项上重新归一化：

$$
g_{t,j}=\frac{s_{t,\pi_{t,j}}}{\sum_{\ell=1}^{k}s_{t,\pi_{t,\ell}}}.
$$

**专家前向。** 对每个 token 的每个被选专家 $e=\pi_{t,j}$：

$$
a_{t,j}=x_tW_1[e]+b_1[e],\qquad h_{t,j}=\operatorname{relu}(a_{t,j}),
$$

```text
if training and p > 0:
    h = h * drop_mask[t, e] / (1 - p)     # inverted dropout
else:
    h = h                                  # drop_mask 被完全忽略
```

$$
y_{t,j}=h_{t,j}W_2[e]+b_2[e],\qquad \text{output}_t=\sum_{j=1}^{k}g_{t,j}\,y_{t,j}.
$$

三个要点：

1. dropout 作用在 **relu 之后的隐藏激活**上，不是作用在 token 输入 $x$、门控分数或专家输出上。
2. 训练态是 inverted scaling：保留元素乘 $1/(1-p)$，推理态不做任何补偿。`training=False` 或 `p=0` 时 dropout 整体关闭，此时 `drop_mask` 即使全 0 也不许改变结果。
3. 门控的重归一化与 dropout 无关，它只是 top-k 稀疏门控的常规做法（原始 softmax 的质量会漏给未选中的专家）。

## 返回

返回四元 `tuple`：

```text
(expert_idx, gates, hidden, output)
```

| 项 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `expert_idx` | int64 `[T,k]` | 选中的专家编号，按分数降序 |
| `gates` | float64 `[T,k]` | 重归一化后的门控权重，每行和为 1 |
| `hidden` | float64 `[T,k,H]` | **dropout 之后**的隐藏激活 |
| `output` | float64 `[T,D]` | 门控加权的专家输出之和 |

浮点结果按 `rtol=1e-9, atol=1e-12` 比较。

## 示例

```python
logits = [[0.0, 0.0, -1.0]]
x      = [[2.0, -1.0]]
w1     = [[[1, 0], [0, 1]],      # 专家 0
          [[0, 1], [1, 0]],      # 专家 1
          [[1, 1], [1, 1]]]      # 专家 2
b1     = zeros(3, 2)
w2     = 三个 2x2 单位阵
b2     = [[0, 0], [1, 1], [2, 2]]
mask   = 专家 0 -> [1, 0]，专家 1 -> [0, 1]，专家 2 -> [0, 0]

k = 2, p = 0.5, training = True
# expert_idx = [[0, 1]]
# gates      = [[0.5, 0.5]]
# hidden     = [[[4, 0], [0, 4]]]
# output     = [[2.5, 2.5]]

training = False
# expert_idx = [[0, 1]]
# gates      = [[0.5, 0.5]]
# hidden     = [[[2, 0], [0, 2]]]
# output     = [[1.5, 1.5]]
```

`softmax([0,0,-1]) = [0.42231880, 0.42231880, 0.15536240]`，两个 `0` 严格并列，按编号取专家 0、1，重归一化后各 `0.5`。

专家 0 的 pre-activation 是 `[2*1+(-1)*0, 2*0+(-1)*1] = [2, -1]`，relu 后是 `[2, 0]`；mask `[1,0]` 与 `1/(1-0.5)=2` 给出 `[4, 0]`，过 `W2=I` 与 `b2=[0,0]` 得到 `[4, 0]`。专家 1 对称地得到 `[0, 4]`，加 `b2=[1,1]` 是 `[1, 5]`。加权和 `0.5*[4,0] + 0.5*[1,5] = [2.5, 2.5]`。

推理态不做缩放：隐藏激活回到 `[2,0]` 与 `[0,2]`，专家输出 `[2,0]` 与 `[1,3]`，加权和 `[1.5, 1.5]`。**同一个输入、同一个 mask，训练与推理的输出不同**——这正是 dropout 的训练/推理不对称性。

## 与相邻题目的边界

- `moe-router-soft-clamp` 处理 logits 的 tanh 软限幅与选择偏置，只到门控为止；本题的门控是普通 softmax + top-k。
- `moe-capacity-jitter-dispatch` 关注容量抖动下的 token 丢弃与分派；本题不做容量限制，也不丢 token。
- `moe-dead-expert-ledger` 统计专家组件的负载与死专家；本题不统计负载。
- `moe-shared-expert-forward` 是共享专家与路由专家的合成；本题所有专家都是路由专家。
- 本题只做前向，不更新门控偏置、不计算负载辅助损失、不回传梯度。
