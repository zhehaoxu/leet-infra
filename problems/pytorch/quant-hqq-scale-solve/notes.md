# 量化：HQQ 半二次闭式 scale 求解——算法详解

## 1. RTN 的盲点

RTN 的 scale 由极值先验决定：\(s=(x_{\max}-x_{\min})/(2^b-1)\)，然后 \(q=\operatorname{round}(x/s)\)。极值一旦被一个离群点撑大，整组分辨率就被浪费，而 round 的误差从来没有进入优化的目标函数。

HQQ（Badri & Shaji, 2024, arXiv:2403.07134）把这件事改写成对 \((s, z, q)\) 的联合最小化：

\[
\min_{s,z,q}\ \|W-s(q-z)\|_2^2,\qquad q=\operatorname{round}(W/s)+z
\]

并用 **half-quadratic splitting**（半二次分裂）交替求解：固定 \(q,z\) 求 \(s\)（闭式），固定 \(s,z\) 求 \(q\)（取整）。本题实现这一交替的前半段——**闭式 scale**，`z` 由 min 对齐后固定。

## 2. 闭式 scale 的推导（判题考点）

对固定整数 \(q\) 与 \(z\)，令 \(u=q-z\)，目标对 \(s\) 是二次型：

\[
\|W-su\|^2=\|W\|^2-2s\langle W,u\rangle+s^2\|u\|^2
\]

对 \(s\) 求导置零：

\[
s^\star=\frac{\langle W,u\rangle}{\|u\|^2}
\]

这就是 Wiener 滤波 / 最小二乘的经典解。它保证：**对同一组整数码 \(u\)，不存在比它更小的 scale**。

`check_closed_form_optimality` 用两个角度断言：

1. 从返回的 `deq` 反解整数码 \(u=\operatorname{round}(\hat W/s)\)，验证 \(s\approx\langle W,u\rangle/\langle u,u\rangle\)（浮点往返误差 \(10^{-4}\) 以内）；
2. 把 \(s\) 扰动 \(\pm10\%\)、保持 \(u\) 不变重算 SSE，必须**不下降**——这是「最小二乘最优」的直接证据。若选手偷懒用 `(hi-lo)/(2^b-1)`，第一条就会失败。

## 3. 为什么零点是「min 对齐 + 固定」

完整的 HQQ 还会对 \(z\) 做网格搜索（8 个候选，取误差最小的）。本题刻意砍掉这一步，原因有二：

- 判题确定性：固定 \(z\) 后整个流程是「确定性的迭代」，与实现细节（候选数、平局规则）无关；
- 考点聚焦：闭式 scale 才是与 RTN 的本质差异。`z=\operatorname{clamp}(\operatorname{round}(-W_{\min}/s),0,2^b-1)` 让最小权重落在码 0 附近，保证 \(u\) 的支撑覆盖数据区间。

`check_beats_rtn` 给出宏观保证：同一批偏斜数据上，HQQ 的总 SSE 不高于对称 RTN（`absmax/(2^(b-1)-1)`、zero=0）。这不是恒等式（两条路径的 \(q\) 不同），但在正态偏斜数据上是稳定的经验事实，也是论文的核心卖点。

## 4. 退化与边界

- 常量组（`hi == lo`）：`s=1, z=0, deq=lo`，跳过迭代——否则 `<q,q> = 0` 会让闭式除法爆炸；
- `iters=0`：只做一次初值 round + 一次重建，等价于「非对称 RTN」；本题允许，便于对比；
- `bits ∈ [2,8]`、`iters ≥ 0` 之外抛 `ValueError`；
- 全部内部 float64：`<W,q>` 的累加在 float32 下会丢精度，导致 `check_closed_form_optimality` 的 \(10^{-4}\) 断言不稳。

## 5. 与本合集其他题的分工

- `symmetric-quantize` / `asymmetric-quantize`：固定 scale 的 fake-quant 语义；
- `gptq-column-quantize`：**跨列**误差补偿（Hessian 逆）；
- `quant-squeezellm-sensitive-kmeans`：同时优化**格点位置**（非均匀），本题格点均匀、只优化 scale；
- 本题：组内 **scale 的闭式最优** 与交替迭代结构。

## 6. 验证方法

- `check_example`：\([0,\dots,15]\) 整数格点精确往返。
- `check_closed_form_optimality`：反解 \(u\) 验证 \(s=\langle W,u\rangle/\langle u,u\rangle\) + 扰动不降。
- `check_beats_rtn`：\(4\times128\) 偏斜数据，总 SSE ≤ 对称 RTN。
- `check_matches_oracle`：5 bit、3 组，纯 python 逐标量重算对拍。
- `check_degenerate_and_validation`：常量组退化 + 三类非法入参。
