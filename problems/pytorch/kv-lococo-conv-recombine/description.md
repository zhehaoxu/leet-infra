# KV Cache：LoCoCo 卷积重算合并

## 题目描述

H2O 一类方法按累计注意力给 token 打分，把分数低的一刀切掉。可分数低不等于信息无用：被丢的 token 往往能用邻居的线性组合近似出来，直接删掉才是永久损失。LoCoCo（2406.05317）因此做**硬保留 + 软重算**两级压缩：

1. 先按 H2O 口径保留重击 token，末尾 `local_len` 个 token 无条件留下；
2. 剩下的非重击 token 里再挑 `mem_compress` 个分数最高的做「残差代表」，其余全部送进一个沿 **token 轴**滑动的一维卷积，学出一组软权重，把它们压成同样多的摘要 token；
3. 卷积权重与「代表 token 的 one-hot 选择」按 `normalizer` 混合，得到最终权重，再乘回 K/V。

卷积核在 token 轴上滑动，每个摘要 token 都能看到一段邻域里的全部非重击 K/V；`[K;V]` 拼在同一批通道上，所以一次卷积就能跨 key/value 通道自由组合。上游的卷积核和 `normalizer` 都是训练出来的参数，本题把它们当**显式输入**，不训练任何东西。

## 任务

实现 `solve(keys, values, scores, conv_weight, normalizer, hh_keep_rate, local_len, mem_size)`：

- `keys`/`values`：`(S,H,D)`，同 dtype（float32/float64），NHD 布局；
- `scores`：`(H,S)` 累计注意力分数，同 dtype，越大越重要；
- `conv_weight`：`(m,2D,K)`，`K` 为正奇数；
- `normalizer`：标量（float 或 0 维 tensor）；
- `hh_keep_rate ∈ (0,1)`、`local_len ≥ 0`、`mem_size ≥ 2` 为整数。

记 \(m=\lfloor \texttt{mem\_size}\,(1-\texttt{hh\_keep\_rate})\rfloor\)（向零截断）、\(\textit{keep}=\texttt{mem\_size}-m\)、\(n=S-\textit{keep}\)。重击集合

\[
\mathcal H=\operatorname*{TopK}_{\textit{keep}-\texttt{local\_len}}\{s_i:0\le i<S-\texttt{local\_len}\}\ \cup\ \{S-\texttt{local\_len},\dots,S-1\}
\]
取分数降序、同分取小 token 号，最后**按 token 号升序**返回。残差集合 \(\mathcal R\) 是补集 \(\bar{\mathcal H}\) 里分数最高的 \(m\) 个，同样按 token 号升序。卷积

\[
c_{o,t}=\sum_{d=0}^{2D-1}\sum_{j=0}^{K-1}w_{o,d,j}\,x_{d,\,t+j-p},\qquad p=\frac{K-1}{2},\qquad x=[k_{\bar{\mathcal H}};v_{\bar{\mathcal H}}]
\]
输出长度仍是 \(n\)，两侧各补 \(p\) 个零。权重

\[
q_{o,t}=\frac{\exp(\max(c_{o,t},0))}{\sum_{t'}\exp(\max(c_{o,t'},0))},\qquad
b_{r,t}=(1-\nu)\,\mathbf 1[t=\pi(r)]+\nu\,q_{r,t}
\]
其中 \(\pi(r)\) 是第 \(r\) 个残差代表在 \(\bar{\mathcal H}\) 升序序列里的位置。最终

\[
K'=[k_{\mathcal H};\,b\,k_{\bar{\mathcal H}}],\qquad V'=[v_{\mathcal H};\,b\,v_{\bar{\mathcal H}}]
\]
返回 6 键字典：`hh_idx`、`res_idx`（int64）、`conv_prob`、`blend`、`merged_keys`、`merged_values`。内部 float64，浮点输出按输入 dtype；不得修改输入。

## 示例

```
keys = [[[1, 0]], [[0, 3]], [[1, 3]], [[-1, 3]], [[0, 3]], [[2, 2]]]   # S=6, H=1, D=2
values = [[[1, 2]], [[2, 5]], [[0, 5]], [[1, 5]], [[-1, 5]], [[.5, .5]]]
scores = [[.9, .1, .7, .2, .6, 5.]]      # 末位是 local token，只保不竞标
mem_size = 4, hh_keep_rate = .5          # m = 2, keep = 2, n = 4
local_len = 1; normalizer = .5; K = 3     # 前缀只选 keep-local = 1 个重击
conv_weight[0, 1, 1] = 1, conv_weight[1, 3, 1] = 1   # 只看 k1 / v1 的中心拍

hh_idx   = [[0, 5]]          # 前缀最高分 token 0，加上恒留的 token 5
res_idx  = [[2, 4]]          # 补集里分数最高的两个（.7 与 .6）
conv_prob = [[[.25, .25, .25, .25],      # 中心拍读到常数通道 3 / 5 → softmax 均匀
              [.25, .25, .25, .25]]]
blend    = [[[.125, .625, .125, .125],   # 0.5*one-hot(res) + 0.5*conv_prob
             [.125, .125, .125, .625]]]
merged_keys   = [[[1, 0], [2, 2], [.5, 3], [0, 3]]]
merged_values = [[[1, 2], [.5, .5], [.25, 5], [-.25, 5]]]
```

## 约束

- **padding 与输出长度**：`nn.Conv1d` 的 same padding 是两侧各 `(K-1)//2` 个零，核长 21 时每侧 10 个；只接受正奇数核，否则输出会短一格。它做的是互相关，`t` 处读 `x[t+j-p]`，不翻转核。
- **`int()` 是向零截断不是四舍五入**：`mem_size*(1-rate)` 的小数部分直接丢弃，`mem_size=10, rate=0.75` 得 `m=2`。`keep` 由 `mem_size-m` 反推，两者相加恒等于 `mem_size`。
- **残差不够就报错**：非重击 token 少于 `m` 时上游 `topk` 会直接崩，这里显式抛 `ValueError`。
- **同分取小号**：重击与残差两次 topk 都按「分数降序、同分取更小 token 号」定序，`torch.topk` 的并列顺序不可依赖。
- **重击顺序是 token 序**：上游用 `masked_select` 收集重击，天然按 token 号升序，而非分数序；残差 one-hot 的行也按代表 token 升序排列。
- **local 段不竞标**：前缀 topk 名额是 `keep-local_len`，末段直接置 True，分数再高也不挤占前缀名额。
- **softmax 在 token 轴且先过 ReLU**：每个输出通道对全部非重击 token 归一，`blend` 每行之和恒为 1；`normalizer=0` 退化成纯选择，`normalizer=1` 退化成纯卷积；上游默认 `hidden_act='relu'`，先 `max(c,0)` 再 softmax，否则负权重会让凸组合失效。
- **`conv_weight` 是全通道卷积**：`(m,2D,K)` 同时混合 K/V 的 2D 个通道，不是 depthwise；上游 `Conv1d` 的 bias 在本题按 0 处理。
