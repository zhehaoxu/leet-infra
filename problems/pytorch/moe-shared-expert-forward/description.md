# 共享专家与路由专家合成

## 任务

实现 `solve`：对每个 token 计算已选路由专家的加权输出，再加上一个**常开、无门控**的共享 SwiGLU 专家，并统计专家参数量。路由索引和权重已经给出，不需要实现路由器。

这是 DeepSeek 风格共享路径的受限数值练习，不是 Qwen2-MoE 的 sigmoid-gated shared expert，也不是整模型前向。

## 输入

所有浮点张量为 CPU `torch.float64`；`expert_ids` 为 CPU `torch.int64`。允许非连续张量，不修改输入。

| 参数 | 形状 | 含义 |
| --- | --- | --- |
| `x` | `[T,D]` | token 隐态 |
| `expert_ids` | `[T,K]` | 每行互不重复的专家编号，范围 `[0,E)` |
| `routing_weights` | `[T,K]` | 与索引对应的非负系数；**不保证和为 1，不得重新归一化** |
| `gate_w`, `up_w` | `[E,H,D]` | 路由专家的门和上投影 |
| `down_w` | `[E,D,H]` | 路由专家的下投影 |
| `shared_gate_w`, `shared_up_w` | `[Hs,D]` | 共享专家的门和上投影 |
| `shared_down_w` | `[D,Hs]` | 共享专家的下投影 |

`T,E,D,H,Hs` 均为 1..8，`1 <= K <= E`。`|x| <= 5`，所有投影权重绝对值不超过 3，路由系数在 `[0,4]`。输入均合法；没有偏置、dropout、残差或归一化层。

## 精确规则

投影使用 `x @ W.T`。对行向量 $x$ 定义：

$$
F_e(x)=\bigl(\operatorname{SiLU}(xW_{g,e}^{\mathsf T})\odot xW_{u,e}^{\mathsf T}\bigr)W_{d,e}^{\mathsf T},\qquad
\operatorname{SiLU}(z)=\frac{z}{1+e^{-z}}.
$$

共享函数 $F_s$ 使用三张共享权重，隐藏宽度为 $H_s$。逐 token 计算：

$$
r_t=\sum_{j=0}^{K-1}a_{t,j}F_{i_{t,j}}(x_t),\qquad
s_t=F_s(x_t),\qquad y_t=r_t+s_t.
$$

共享输出既不乘路由系数，也不因未被选中而关闭。即使全部路由系数为零，共享路径仍然生效。

参数账只包含这些专家的投影矩阵，**不包含未给出的路由器或整模型参数**：

$$
P_{\mathrm{total}}=3D(EH+H_s),\qquad
P_{\mathrm{active}}=3D(KH+H_s).
$$

`active_params` 是**每个 token 所选路径的结构参数量**，不是整个 batch 触达专家的并集，也不乘 token 数。系数为零仍计入该选中路径。

## 返回

返回五项 `tuple`：`(output, routed, shared, total_params, active_params)`。

- 前三项是 CPU float64 张量，形状均为 `[T,D]`，分别为 $y,r,s$。
- 后两项是 Python `int`。
- 浮点结果按 `rtol=2e-10, atol=2e-10` 比较。

## 示例

所有浮点量按 float64 构造：

```python
x = [[1.0]]
expert_ids = [[1, 0]]
routing_weights = [[2.0, 0.5]]
gate_w = [[[1.0]], [[1.0]]]
up_w = [[[2.0]], [[4.0]]]
down_w = [[[1.0]], [[1.0]]]
shared_gate_w = [[1.0]]
shared_up_w = [[3.0]]
shared_down_w = [[1.0]]
```

令 $q=\operatorname{SiLU}(1)=1/(1+e^{-1})$，完整输出为：

```python
(output, routed, shared, total_params, active_params)
= ([[12*q]], [[9*q]], [[3*q]], 9, 9)
```

前三项表示张量。路由部分为 $2\cdot4q+0.5\cdot2q=9q$；共享部分为 $3q$。只把 `routing_weights` 换成 `[[0.0,0.0]]`，输出即为 `([[3*q]], [[0.0]], [[3*q]], 9, 9)`，结构参数账不变。
