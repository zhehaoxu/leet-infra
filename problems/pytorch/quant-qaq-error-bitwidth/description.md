# 量化：QAQ 误差感知的 KV 位宽反解

## 题目描述

给 KV cache 定一个全局固定位宽是「一刀切」：动态范围小的 (token, layer, head) 单元浪费位数，动态范围大的又不够用。QAQ（arXiv 2403.04643）的注意力感知思路反过来——先给每个量化单元**定一个误差上界** \(\varepsilon_{\max}\)，再**反解**出它需要的位数，最后用这个位数做一次仿射量化-反量化；极端值先挑出来跳过、原值保留。

反解式来自仿射量化的舍入误差。把单元动态范围 \(\Delta\) 铺到 \(2^n\) 个格点，格距是 \(\Delta/2^{n}\)，舍入误差落在 \([-\Delta/2^{n+1},\ \Delta/2^{n+1}]\)、按均匀分布算方差为 \(\Delta^{2}/(12\cdot 4^{n})\)。令它不超过目标误差 \(\tau\) 并解出 \(n\)，实现里写成

\[
n\ =\ \Bigl\lceil \log_2\Bigl(\frac{\Delta}{2\,\varepsilon_{\max}}+1\Bigr)\Bigr\rceil,\qquad
\Delta=\begin{cases}2\max|x|, & \text{symmetric}\\ \max x-\min x, & \text{asymmetric}\end{cases}
\]

两个极值都**只在非 outlier 元素上**取。误差上界本身分两条公式（\(L=\) seq_len、\(\tau=\) 目标误差）：

\[
\varepsilon_{\max}^{\text{key}}=\sqrt{\frac{12}{\lVert q\rVert_2}\,\log\Bigl(\frac{L^{3}}{L-1}\,\tau^{2}+1\Bigr)},\qquad
\varepsilon_{\max}^{\text{value}}=\frac{\sqrt{12/L}\ \tau}{\max|A|}
\]

\(\lVert q\rVert_2\) 是 query 的 2-范数，\(\max|A|\) 是 attention 末 `last_n_attentions` 行沿量化维度取的 amax。于是 key 分支的 \(\varepsilon_{\max}\) 只由 \((L,\tau,\lVert q\rVert_2)\) 决定、与 cache 取值无关，每个量化单元同值；value 分支才逐单元变化。

三个容易写错的口径：掩码低端用 \(\le\)、高端用 \(>\)；位宽是先 \(\lceil\cdot\rceil\) 再 clamp、最后把末 `last_n-1` 个 token 无条件抬到 `n_bits_max`；`method="normal"` 只换量化器，不改上面的 minmax 位宽公式。

## 任务

实现 `solve(cache, attentions, key_or_value, level, symmetric, outliers_ratio, last_n_attentions, target_quantization_error, n_bits_min, n_bits_max, q_norm=None, method="uniform", quantile_upper=None, quantile_center=None)`：

- `cache`：float32/float64，`(n_batch, seq_len, n_layer, n_head, head_dim)`，即上游 permute 之后的布局；
- `attentions`：key 分支传 `None`；value 分支传 `(n_layer, n_batch, n_head, q_len, seq_len)`；
- `key_or_value`：`"key"` 或 `"value"`；`level`：`"token"/"layer"/"head"` 对应量化维 `(-3,-2,-1)/(-2,-1)/(-1,)`；
- `symmetric`：bool；`outliers_ratio`：落在 \([0,1)\)；`last_n_attentions`：int 且 `>= 1`；`target_quantization_error`：`> 0`；
- `n_bits_min`/`n_bits_max`：整数且 `0 <= n_bits_min <= n_bits_max <= 16`；`q_norm`：key 分支必填的正数；
- `method`：`"uniform"` 或 `"normal"`；`quantile_upper`/`quantile_center`：normal 必填，第 \(n\) 项是长度 \(2^{n}\) 的一维表。

记量化维个数 \(k\)、量化单元形状 \(\text{lead}=\text{cache.shape}[:-k]\)。内部一律 float64，返回按 `cache` 的 dtype。返回固定 5 个键：

- `outlier_mask`：bool，形状同 `cache`；
- `max_error`：float，形状为 `lead`（key 分支把标量广播到每个单元）；
- `n_bits_clamped`：int64，形状为 `lead`，ceil+clamp 之后、末行覆盖之前；
- `n_bits`：int64，形状为 `lead`，末行覆盖之后，真正用于量化；
- `quant_dequant`：float，形状同 `cache`，量化再反量化的结果，outlier 位置等于原值。

非法输入抛 `ValueError`：`cache` 不是五维或不是 float32/float64；`key_or_value`/`level`/`method` 取值非法；`symmetric` 不是 bool；`outliers_ratio` 不在 \([0,1)\)；`last_n_attentions < 1`；位宽区间越界或 `n_bits_min > n_bits_max`；`target_quantization_error <= 0`。key 分支缺 `q_norm`、`q_norm <= 0` 或 `seq_len < 2`；value 分支 `attentions` 缺失/不是五维、其 layer/batch/head 与 `cache` 不一致、最后一维不等于 `seq_len` 或 `q_len < last_n_attentions`。normal 缺量化表、表长度没覆盖到 `n_bits_max`（要 `>= n_bits_max+1`）、或第 \(n\) 项长度不是 \(2^{n}\)。某个量化单元的非 outlier 元素太少（uniform 需 `>= 1`，std 需 `>= 2`）。

## 示例

```
cache  shape (1, 3, 1, 1, 4)      # token level: one 4-element unit per (batch, seq_len)
= [[[-3.0, -0.5, 1.0, 4.0],
    [-2.0, 0.25, 0.5, 6.0],
    [-1.0, -0.25, 0.5, 3.0]]]
key_or_value="key"   level="token"   symmetric=True   outliers_ratio=0.5   method="uniform"
last_n_attentions=2  target_quantization_error=0.01   q_norm=3.0   n_bits_min=1  n_bits_max=4

max_error      = [[0.0734599093, 0.0734599093, 0.0734599093]]   # scalar, broadcast to (1, 3)
outlier_mask   = [[True, False, False, True]] * 3               # x <= 1st-smallest | x > 3rd-smallest
n_bits_clamped = [[4, 3, 3]]
n_bits         = [[4, 3, 4]]                                     # last token row forced to n_bits_max
quant_dequant  = [[[-3.0, -0.5, 0.875,  4.0],
                   [-2.0,  0.25, 0.375, 6.0],
                   [-1.0, -0.25, 0.4375, 3.0]]]
```

每行最低与最高的两个元素被判为 outlier、原值保留，中间两个参与量化。第三行反解出来本来是 3 位，但 `last_n_attentions=2` 把最后一个 token 无条件抬到 4 位，于是 0.5 在 4 位格距下顶到正上界、反量化成 0.4375，而不是 3 位时的 0.375。

## 约束

- **双阈值是非对称比较**：`mask = (cache <= lower) | (cache > upper)`，`lower = kthvalue(ceil(ratio/2*M))`、`upper = kthvalue(floor((1-ratio/2)*M))`。`ratio = 0` 时直接全 False，不走 `kthvalue`。
- **两个位宽键要分清**：`n_bits_clamped = clamp(ceil(log2(Delta/(2*eps)+1)), n_bits_min, n_bits_max)`；`last_n > 1` 时再执行 `n_bits[:, -last_n+1:] = n_bits_max`，只有 `n_bits` 参与量化。
- **key 分支的误差与 cache 无关**：广播后每个量化单元同值（测试断言 `min == max`）；value 分支只看 attention 末 `last_n` 行，更早的历史行改了不影响位宽。
- **symmetric 只是把量化中心置 0**：尺度取 `2*max|x|`；但 `method="normal"` 求标准差时仍围绕真实均值，且用 ddof=1。
- **位宽是 ceil，不是四舍五入**；量化取整才是半偶（round-half-even），clamp 到 `[-(2**(n-1)), 2**(n-1)-1]`。`n = 1` 的对称量化只能表示 `[-scale, 0]`，正数一律塌到 0。
- **normal 走落桶**：std 归一化后用 `searchsorted(upper)` 定桶号，再取 `center[bucket]` 反归一化；量化表必须覆盖到 `n_bits_max`。
- **dtype 与副作用**：内部 float64，`max_error`/`quant_dequant` 转回 `cache.dtype`，两个位宽键恒为 int64；不得就地修改输入，非连续输入的结果要与连续副本一致。
