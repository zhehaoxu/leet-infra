# 量化：KIVI KV Cache 非对称量化——算法详解

## 1. 为什么 Key 和 Value 要不同粒度

KIVI（Liu et al., ICML 2024, arXiv:2402.02750）的核心观察是 KV cache 里 Key 与 Value 的**分布方向性**不同：

- Key 的几个特征通道（`head_dim` 上的少数几维）承载了绝大部分方差，且这些「离群通道」在**所有 token 上位置固定**。逐 token 量化会让每个 token 各自的离群通道吃掉码字，而按通道量化只需为这几个通道保留高分辨率。
- Value 没有固定的离群通道，但 token 之间的整体尺度差异大（尤其在早期 token）。逐 token 量化让每个 token 自己定 scale。

因此 KIVI 的配方是「Key per-channel + Value per-token」，与本题签名一致。

## 2. 非对称量化的口径（判题考点）

本题采用 KIVI 的最简 zero-point 形式——**zero-point 就是组的 min**：

\[
\Delta=\frac{x_{\max}-x_{\min}}{2^b-1},\qquad
q=\operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{x-x_{\min}}{\Delta}\right),0,2^b-1\right),\qquad
\hat x=q\Delta+x_{\min}
\]

- \(x_{\min}\) 精确落在码 0、\(x_{\max}\) 精确落在码 \(2^b-1\)——`check_roundtrip_bound` 断言每组的 min/max 往返无损，任何「scale 除 2^b 而非 2^b−1」「zero-point 取整」的写法都会破坏它。
- 若把 zero-point 实现成整数 `round(-x_min/Δ)`，当整组数据远离 0（例如 \([2,3]\)）时它会超出 \([0,2^b-1]\) 被 clamp，覆盖区间整体错位、重建误差爆炸。本题直接用浮点 min 避开这个陷阱，`check_example` 的 \([1,3]\)、\([2,3]\) 两组就是为此设计。
- 退化（`max == min`）：`Δ` 取 1，码恒为 0，`deq = x_min`，精确还原——`check_degenerate_and_validation` 用常量通道/常量 token 断言。

## 3. 粒度断言的写法

`check_granularity` 同时钉死两件事：

1. 参数张量的**长度**：K 的 scale/zero 为 `(D,)`、V 的为 `(T,)`——写反统计方向会立刻 shape 不匹配；
2. 参数张量的**数值**：`ks == (K.amax(0) - K.amin(0))/15`、`kz == K.amin(0)`、`vs == (V.amax(1)-V.amin(1))/15`、`vz == V.amin(1)`——即使 shape 碰巧一致（T == D）也能区分。

## 4. 与其余量化题的分工

- `per-channel-quantize`：**对称**（absmax/127）逐通道，给的是通用权重轴；
- `asymmetric-quantize`：**对称 vs 非对称**的通用对照，单一张量；
- 本题：**两种粒度组合**（K 逐通道 / V 逐 token）+ 浮点 min zero-point，落到 KV cache 场景；
- `kv-cache-memory`：KV cache 的显存公式，不做量化数值。

## 5. 实现要点

- 向量化：`amin/amax(dim, keepdim=True)` → `(x - mn)/scale` 广播，`round().clamp(0, qmax)`，再 `q*scale+mn`；参数用 `squeeze(dim)` 变一维。
- `torch.round` 是银行家舍入（`round(2.5) == 2`），与论文实现一致；不要用 `floor(x+0.5)`。
- 退化组用 `torch.where(span > 0, span/qmax, ones)`，避免 `0/0` 的 NaN 传播。

## 6. 验证方法

- `check_example`：\(2\times2\) 手算 scale/zero 与无损往返。
- `check_granularity`：参数长度 + 数值双断言（T≠D 的 \(5\times7\)）。
- `check_roundtrip_bound`：误差 ≤ Δ/2、组 min/max 精确还原。
- `check_matches_oracle`：\(4\times3\)、6 bit，逐组 python 重算对拍。
- `check_degenerate_and_validation`：常量组精确还原 + `bits` 越界抛 `ValueError`。
