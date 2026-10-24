# KeyRerotation：驱逐后的 RoPE 重旋转

## 题目描述

RoPE 把"绝对位置"编码成对 key 向量的一个旋转角。KV 压缩**丢掉中间 token 后**，剩下的 key 仍带着它们**原始位置**的旋转角——可新缓存里它们被当成了连续的第 $0,1,2,\dots$ 个位置，相对关系就错了。**KeyRerotation**（见 FINCH，[arXiv:2407.09112](https://arxiv.org/abs/2407.09112)；亦即 HF SinkCache 的做法）在 gather 之后，把每个保留 key 再旋转 $(q-p)$，让它"看起来"就待在压缩后的新位置 $q$ 上。

## 规则

给定 `keys` 形状 $(H, L, D)$（**已在各自原始位置做过 RoPE**）、`indices` 形状 $(H, n\_kept)$（每个 head 保留哪些原始位置，long）、`inv_freq` 形状 $(D/2,)$：返回 $(H, n\_kept, D)$ 的重旋转 key。

对第 $h$ 头、新位置 $i\in[0,n\_kept)$、其原始位置 $p=\text{indices}_{h,i}$：

1. $delta = i - p$
2. 频率角 $freqs_i = delta \cdot \text{inv\_freq}\ \ (D/2\text{ 维}),\quad emb_i = [\,freqs_i,\ freqs_i\,]\ \ (\text{拼接成 } D)$
3. 取 gather 后的 $k = \text{keys}_{h,p}$，$\text{rotate\_half}(k)$ 定义为：前一半取 $-k[t+D/2]$、后一半取 $k[t-D/2]$
4. $\text{out}_{h,i} = k \odot \cos(emb_i) + \text{rotate\_half}(k) \odot \sin(emb_i)$

## 示例

```
H=1, L=5, D=4, inv_freq=[1.0, 0.1], indices=[[0, 2, 4]]
keys[0] = [[1.0,0.5,0.2,0.8], [0.3,0.7,0.1,0.4], [0.9,0.2,0.6,0.5],
           [0.4,0.4,0.4,0.4], [0.7,0.1,0.3,0.9]]

i=0, p=0 → delta=0 → cos=1, sin=0 → 原样 [1.0, 0.5, 0.2, 0.8]
i=1, p=2 → delta=-1, k=[0.9,0.2,0.6,0.5], freqs=[-1.0,-0.1]
          emb=[-1.0,-0.1,-1.0,-0.1]
          out=[0.9912, 0.2489, -0.4331, 0.4775]
i=2, p=4 → delta=-2, k=[0.7,0.1,0.3,0.9] → out=[-0.0185, 0.2768, -0.7614, 0.8622]

out[0] = [[1.0, 0.5, 0.2, 0.8],
          [0.9912, 0.2489, -0.4331, 0.4775],
          [-0.0185, 0.2768, -0.7614, 0.8622]]
```

## 约束

- CPU float64 容差 $10^{-5}$；$D$ 为偶数，`inv_freq` 长度恰为 $D/2$。
- `indices` 每行**严格升序**、值域 $[0, L)$；不同 head 的 `indices` 可以不同。
- $delta$ 可为负，$\cos$ 偶、$\sin$ 奇自动处理反向旋转。
