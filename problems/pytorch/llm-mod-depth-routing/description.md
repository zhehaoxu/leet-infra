# Mixture-of-Depths：逐层 top-k 深度路由与残差传递

## 题目描述

Mixture-of-Depths（Raposo et al., 2024, arXiv 2404.02258）把"算力分配"从**专家维度**搬到**深度维度**：每个 block 前面挂一个标量 router，只有 router 分数最高的若干个 token 才真正进入该 block，其余 token 直接沿残差流过。因为每层只算一部分 token，同一张图里不同 token 走的深度不同——这就是"Mixture of Depths"。

本题不给真实 Transformer block，而是把它替换成一个确定性的替身：被选中的 token 组成子序列，做一次**共享 q/k/v 投影的因果自注意力** \(F=P H\)。你要实现完整的逐层前向：打分、按容量取 top-k、子序列 attention、按分数缩放写回残差，并把每层的路由决策一起返回。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- 每层分数 \(s_i=x_i\cdot w_l\) 必须用**进入该层时的当前状态**计算；上一层更新过的 token 会改变下一层的路由结果。
- 选容量 \(k_l=\min(C_l,T)\)：按 \((s_i\text{ 降序},\ \text{token 下标升序})\) 取前 \(k_l\) 个。分数并列时下标小的优先，这是本题为确定性补的约定（原文只说"超出容量的 token 丢掉"，并列时依赖实现顺序）。
- 选中的下标要按**原始下标升序**排成子序列，再整体过 block；因果掩码因此作用在子序列的位置上，而不是原始序列的位置上。
- block：\(H=\tilde X W_l\)，\(A=HH^{\mathsf T}/\sqrt d\)，\(A_{ij}=-\infty\) 当 \(j>i\)，\(P=\mathrm{softmax}(A)\)，\(F=PH\)。
- 写回：\(x_i\leftarrow x_i+s_i\cdot F_{\text{row}(i)}\)。router 分数是**乘在 block 输出上**的实数系数（原文式 (1)），不是二值门控。
- 未被选中的 token 必须严格保持原值（不是乘 0 的近似）；容量为 0 的层什么都不算，但它的 `router_scores` 仍要如实给出。
- 返回 `(output, route_mask, router_scores)`：`output` 是 `(T, d)` float32，`route_mask` 是 `(L, T)` bool，`router_scores` 是 `(L, T)` float32（各层决策时用到的分数）；内部用 float64 计算。

## 示例

```
hidden     = [[1, 0], [0, 1], [1, 1]]
routers    = [[1, 0], [0, 1]]
blocks     = [I, I]
capacities = [2, 1]

router_scores = [[1.000000, 0.000000, 1.000000],
                 [0.000000, 1.000000, 1.669762]]
route_mask    = [[1, 0, 1],
                 [0, 0, 1]]
output        = [[2.000000, 0.000000],
                 [0.000000, 1.000000],
                 [5.339523, 4.457865]]
```

第 0 层分数 \([1,0,1]\)，容量 2、并列取下标小的，选中 \(\{0,2\}\)；子序列是 \([[1,0],[1,1]]\)，因果 attention 让第 2 个 token 混到 \([1.0,0.669762]\)，乘上分数 1 写回得到 \([2.0,1.669762]\)。第 1 层用**更新后**的状态打分得到 \([0,1,1.669762]\)，容量 1 选中下标 2，单行子序列的 attention 退化为 \(F=x\)，于是 \(x_2\leftarrow x_2+1.669762\cdot x_2=[5.339523,4.457865]\)。下标 1 从未被选中，保持 \([0,1]\)。

## 约束

- `hidden_states` 必须是 float32/float64 张量且为二维 `(T, d)`，`T >= 1`、`d >= 1`；`router_weights` 二维 `(L, d)`、`block_weights` 三维 `(L, d, d)`，`L >= 1`；三者都必须有限。
- `capacities` 必须是 list/tuple，长度等于 `L`，每个元素是 `int`（`bool`、`float`、字符串都不算）且 \(\ge 0\)；超过 `T` 按 `T` 处理。
- 所有维数/形状不匹配、非浮点张量、含 NaN/±inf、以及上面列的非法 `capacities` 都抛 ValueError。
- 每层时间 \(O(Td+c^2 d)\)（\(c=k_l\)），总时间 \(O(L(Td+c^2d))\)，空间 \(O(Td+cd)\)。
