# MLA 解码：权重吸收后的潜在空间注意力

## 题目描述

DeepSeek 的 **Multi-head Latent Attention**（MLA）把 KV 压成一个低维潜在向量 \(c_{KV}\in\mathbb R^{d_c}\) 存进缓存，用两组每头投影把潜在向量升回键和值：

\[ k^{\text{nope}}_h=W_{UK,h}\,c_{KV},\qquad v_h=W_{UV,h}\,c_{KV} \]

注意 \(W_{UK}\) 只作用于**非 RoPE** 的键分量；位置信息由另一条 \(d_r\) 维的 RoPE 键 \(k^{PE}\) 提供，且它在所有头之间共享。

朴素做法是每步解码都先把缓存里每个 token 升维成 \(k^{\text{nope}}\) 和 \(v\)（每头 \(d_h\) 维），再做注意力。**权重吸收**（weight absorption）则把这个升维整个"推到"注意力两端：

\[ \underbrace{q_h^\top W_{UK,h}}_{\text{吸收进查询}}\;c_{KV}\;+\;q^{PE}_h\cdot k^{PE} \]

因为 \(W_{UK}\) 与 token 无关，可以先把它乘进查询得到 \(\tilde q_h=q_h^\top W_{UK,h}\in\mathbb R^{d_c}\)，之后注意力打分直接在 \(d_c\) 维潜在空间里与 \(c_{KV}\) 内积；同理值的加权求和也可以先在潜在空间里做（\(\sum_t p_t c_{KV,t}\)），最后对每个头做**一次** \(W_{UV,h}\) 升维。这样解码时完全不需要物化 \(S\times H\times d_h\) 的键值张量。

## 规则

输入：

- `q_nope`：\((B,H,d_h)\)，每头的非 RoPE 查询分量（已经过 \(W_{UQ}\)）；
- `q_pe`：\((B,H,d_r)\)，每头已施加 RoPE 的查询分量；
- `c_kv`：\((B,S,d_c)\)，潜在 KV 缓存；
- `k_pe`：\((B,S,d_r)\)，所有头共享、已施加 RoPE 的键分量；
- `W_UK`、`W_UV`：都是 \((H,d_h,d_c)\)，即 \(W_{UK,h}\) 的**第 \(k\) 行**是 \(k^{\text{nope}}\) 的第 \(k\) 个坐标对 \(c_{KV}\) 的系数；
- `scale`：softmax 前的缩放因子，有限正实数。

计算：

1. \(\tilde q_h=q_h^\top W_{UK,h}\)，即 \(\tilde q_h[c]=\sum_k q_h[k]\,W_{UK,h}[k,c]\)，得到 \((B,H,d_c)\)；
2. 打分 \(s_{h,t}=\big(\tilde q_h\cdot c_{KV,t}+q^{PE}_h\cdot k^{PE}_t\big)\cdot\text{scale}\)，softmax 到 \(p_{h,t}\)；
3. 潜在空间加权求和 \(z_h=\sum_t p_{h,t}\,c_{KV,t}\)，得到 \((B,H,d_c)\)；
4. 每头一次升维 \(o_h=W_{UV,h}\,z_h\)。

返回 float32 张量，形状 \((B,H,d_h)\)。

## 示例

\[ q^{\text{nope}}=\begin{bmatrix}1.0&0.5\\0.0&-1.0\end{bmatrix},\quad q^{PE}=\begin{bmatrix}0.5&0.5\\-0.5&0.5\end{bmatrix},\quad c_{KV}=\begin{bmatrix}1.0&0.0\\0.5&1.0\\-1.0&0.5\end{bmatrix},\quad k^{PE}=\begin{bmatrix}0.0&1.0\\1.0&0.0\\0.5&0.5\end{bmatrix} \]

\[ W_{UK,0}=\begin{bmatrix}1.0&0.0\\0.0&1.0\end{bmatrix},\ W_{UK,1}=\begin{bmatrix}0.5&0.5\\0.0&1.0\end{bmatrix},\quad W_{UV,0}=\begin{bmatrix}1.0&0.5\\0.0&1.0\end{bmatrix},\ W_{UV,1}=\begin{bmatrix}0.5&0.0\\1.0&1.0\end{bmatrix} \]

（这里 \(H=2\)、\(d_h=d_r=d_c=2\)、\(S=3\)、\(B=1\)，每个矩阵的上方下标是头号。）取 `scale=0.5`：

```
返回 shape (1, 2, 2):
[[[0.698159, 0.500000],
  [0.146223, 0.632368]]]
```

## 约束

- 六个输入都必须是张量且维数固定：`q_nope`/`q_pe`/`c_kv`/`k_pe`/`W_UK`/`W_UV` 均为 3 维，否则抛 `ValueError`。
- 形状必须自洽：`q_pe` 的 batch 与头数同 `q_nope`；`k_pe` 为 \((B,S,d_r)\) 且与 `c_kv` 的 batch、序列长度一致；`W_UK`、`W_UV` 都是 \((H,d_h,d_c)\)。
- `scale` 必须是有限正实数，`bool` 与字符串不接受。
- **等价性质**：结果必须与「先逐 token 升维 \(k^{\text{nope}},v\)，再做常规注意力」逐值一致（误差在 float32 精度内）；\(S=1\) 时 softmax 恒为 1，输出恰为 \(W_{UV,h}\,c_{KV,0}\)，与查询无关；查询与 RoPE 键全零时权重均匀，输出为 \(W_{UV,h}\) 作用在 \(c_{KV}\) 的均值上。
- 各头互相独立：改动某个头的 \(W_{UK}\) / \(W_{UV}\) 不得影响其它头的输出。
- 计算内部可用 float64，返回 float32。
