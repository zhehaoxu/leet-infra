# KV Cache：TPA 张量积注意力的因子收缩与掩码

## 题目描述

TPA（Tensor Product Attention，2501.06425）把每个 token 的 Q/K/V 拆成两个低秩因子的张量积：一个**逐头**系数，一个**逐位置、跨 head 共享**的基。以 key 为例，

\[
K_{m,h,d}=\sum_{s=1}^{S}\sigma(a^{k}_{m,h,s})\,b^{k}_{m,d,s},
\]

其中 \(a^k\) 的形状是 `(M, H, S)`，\(b^k\) 是 `(M, D, S)`——注意 \(b^k\) 不带 head 轴。于是 KV cache 每 token 只需存 \(H\cdot S + D\cdot S\) 个数（value 同理是 \(H\cdot S_V + E\cdot S_V\)），而不是稠密的 $H\cdot(D+E)$；取 `S=4, D=128` 时压缩非常可观。decode 时也**不物化**每头的 K：先算 `bq @ bk` 得到 \((R,S)\) 的小块，再用 \(a^q\) 收缩到 \(R\)、用 \(a^k\) 收缩到 head，最后才 softmax。

本题实现上游 `decode/tpa_flash_kernel_prefill_mask.py` 中 `tpa_decode_torch` 的 CPU 参考路径：sigmoid 门控、四个 scale 的折算、causal/加法/布尔三种掩码。

## 任务

实现 `solve(aq, ak, av, bq, bk, bv, is_causal=False, attn_mask=None, scale=None, scale_q=None, scale_k=None, scale_v=None)`：

- `aq` `(B, N, H, R)`、`ak` `(B, M, H, S)`、`av` `(B, M, H, S_V)`：逐头门控系数；
- `bq` `(B, N, R, D)`、`bk` `(B, M, D, S)`、`bv` `(B, M, E, S_V)`：逐位置、跨 head 共享的基；
- 先取 `sigmoid`，再按最后一轴收缩：

\[
q = \sigma(a^q)\cdot b^q\cdot s_q,\quad
k = \sigma(a^k)\cdot b^k\cdot s_k,\quad
v = \sigma(a^v)\cdot b^v\cdot s_v
\]

- 默认 \(s_q=1/R\)、\(s_k=1/S\)、\(s_v=1/S_V\)、\(s=D^{-1/2}\)；显式传入时覆盖默认值；
- 打分 \(S_{bhnm} = q\cdot k \cdot s + \text{bias}\)，softmax 沿 `M` 轴；
- `is_causal=True`：用**左上对齐**的 `tril(N, M)` 掩码（第 `i` 行只允许 `j <= i`），此时 `attn_mask` 必须为 `None`；
- `attn_mask`：`(N, M)` 的 bool（`True` 表示保留）或与输入同 dtype 的加法掩码，广播到所有 batch/head；
- 返回 `dict`：
  - `out` `(B, N, H, E)`：注意力输出；
  - `prob` `(B, H, N, M)`：实际使用的 softmax 概率（被遮处恰为 0）。

内部用 float64 计算，`out`/`prob` 转回输入 dtype；不得修改任何输入。张量维数不对、dtype 混杂、`attn_mask` 形状/类型不对、`is_causal` 非 bool、`is_causal` 与 `attn_mask` 同给、scale 非有限实数，均抛 `ValueError`。

## 示例

```
aq = [[[[0]], [[0]]]]                     # (B,N,H,R) = (1,2,1,1)
bq = [[[[1, 0]], [[0, 1]]]]               # (B,N,R,D) = (1,2,1,2)
ak = [[[[0]], [[0]], [[0]]]]              # (B,M,H,S) = (1,3,1,1)
bk = [[[[1], [0]], [[0], [1]], [[1], [1]]]]        # (B,M,D,S) = (1,3,2,1)
av = [[[[0]], [[0]], [[0]]]]              # (B,M,H,S_V) = (1,3,1,1)
bv = [[[[1], [0]], [[0], [1]], [[2], [2]]]]        # (B,M,E,S_V) = (1,3,2,1)
is_causal = True

# sigmoid(0)=0.5、s_q=s_k=s_v=1、s=1/sqrt(2)
prob[0,0] = [[1, 0, 0],
             [0.45592056, 0.54407944, 0]]
out[0]    = [[[0.5, 0]],
             [[0.22796028, 0.27203972]]]
```

第 0 行 query 只能看第 0 个 key，概率必为 `[1,0,0]`，输出就是 `v0 = 0.5*[1,0]`。第 1 行可看 key 0/1，两个打分是 0 与 \(0.25/\sqrt{2}\)，softmax 后为 `[0.4559, 0.5441]`；key 2 虽然打分也非零，但被 causal 掩码压成 `-inf`，概率恰为 0。

## 约束

- `B, N, M, H, R, S, S_V, D, E` 均 ≥ 1；所有输入同 dtype（float32/float64）。
- 张量积口径固定为「先 `sig(a)` 再乘基」：`aq` 与 `bq` 在 `R` 轴收缩、`ak`/`av` 与各自的基在 `S`/`S_V` 轴收缩。
- scale 的落点：\(1/R\)、\(1/S\)、\(1/S_V\) 分别乘在 q、k、v 上，$D^{-1/2}$ 乘在点积上；四者都是可显式覆盖的实数。
- `attn_mask` 的 bool 语义是 **True=保留**（上游是 `masked_fill_(mask.logical_not(), -inf)`）；加法掩码直接加到打分上。
- 每个 query 至少要有一个可看的 key（否则 softmax 全 `-inf`，结果未定义）；测试只构造这种输入。
- 本题不考 varlen（上游 torch 路径忽略 `cu_seqlens`）、不考 Triton 在线 softmax/LSE 输出，也不等价于无 sigmoid 的 `flashtpa_decode_torch.py` 单秩变体——口径以带掩码的 `tpa_decode_torch` 为准。
