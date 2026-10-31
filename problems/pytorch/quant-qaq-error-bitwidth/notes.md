# 解题说明 · QAQ 误差感知 KV 位宽反解

## 一句话

先给每个量化单元定一个误差预算 \(\varepsilon_{\max}\)，用位宽公式 \(n=\lceil\log_2(\Delta/(2\varepsilon_{\max})+1)\rceil\) **反解**出该单元要几位，再用这几位做仿射量化-反量化；outlier 先摘掉、原值保留。

## 位宽为什么这么反解

设某单元非 outlier 元素的动态范围是 \(\Delta\)、\(n\) 位仿射量化的格距是 \(\Delta/2^{n}\)，四舍五入带来的误差约是半个格距。于是 \(\Delta/(2\varepsilon_{\max})\) 读作「把动态范围切成误差预算的一半一半，能容下几格」；对它取 \(\log_2\) 再向上取整，就是至少能放下这么多格的位数。式子里的 `+1` 是为了让常量单元（\(\Delta = 0\)）落到 \(\log_2 1 = 0\)、随后被 `n_bits_min` 托住。

两条误差预算是 QAQ 给的口径，本题原样钉死：

\[
\varepsilon_{\max}^{\text{key}}=\sqrt{\frac{12}{\lVert q\rVert_2}\,\log\Bigl(\frac{L^{3}}{L-1}\,\tau^{2}+1\Bigr)},\qquad
\varepsilon_{\max}^{\text{value}}=\frac{\sqrt{12/L}\ \tau}{\max|A|}
\]

区别只在**依赖谁**：key 分支只吃 \((L,\tau,\lVert q\rVert_2)\) 三个标量，与 cache 里存的值完全无关，所以整条 cache 共用一个 \(\varepsilon_{\max}\)；value 分支的分母是 attention 末 `last_n` 行沿量化维的 amax，逐单元不同——amax 越大就除得越小，\(\varepsilon_{\max}\) 收紧，位宽随之升高。

## 三个易错口径

1. **双阈值是非对称比较**。`lower` 取第 \(\lceil \text{ratio}/2\cdot M\rceil\) 小、`upper` 取第 \(\lfloor(1-\text{ratio}/2)M\rfloor\) 小，然后 `mask = (x <= lower) | (x > upper)`——低端含等号、高端不含。写成两侧都含或都不含，边界值上的掩码就会翻。`ratio == 0` 走短路直接返回全 False，根本不进 `kthvalue`（否则 `k = M` 会越界或选出最大元素）。
2. **位宽键有两个**。`n_bits_clamped` 是 ceil + clamp 之后的结果；`n_bits` 再把末 `last_n-1` 个 token 覆盖成 `n_bits_max`（这些 token 的 attention 历史不足 `last_n` 行，误差估不准）。量化用的是后者，前者只是中间量，必须一起返回。
3. **outlier 要同时摘在两个地方**。求 \(\Delta\)（`scale_value`）时只用非 outlier 元素，否则极值把动态范围撑爆、所有元素都跟着升位；量化写回时 outlier 位置用 `torch.where(mask, x, ...)` 原值透传，不参与任何格点映射。

## 判分点

- `max_error.shape == n_bits.shape`，即位宽形状 `lead`；key 分支断言其 `min() == max()`。
- `n_bits_clamped` 与 `n_bits` 分别对上，且只有末 `last_n-1` 个 token 有差异。
- uniform：归一化后 `round`（**半偶**，不是四舍五入）再 clamp 到 `[-(2**(n-1)), 2**(n-1)-1]`，最后 `q*scale + mean`。
- normal：std 归一化（ddof=1）→ `searchsorted(upper_table)` 定桶 → 取 `center_table[bucket]` → 反归一化。`symmetric` 只把量化的中心 `mean` 置 0，**标准差仍然围绕真实均值**。
- `n == 1` 的对称量化格点是 `[-1, 0]`，正数一律塌成 0——这是 clamp 上界 `2**(n-1)-1 == 0` 的直接后果，不是 bug；同时 `quant_dequant[mask] == cache[mask]`，`n_bits.min()/max()` 落在给定区间内。

## 边界

- key 分支 `seq_len < 2` 会撞上分母 `L-1`，直接报错；`q_norm` 缺失或非正也报错。
- value 分支要求 `attentions` 是五维 `(n_layer, n_batch, n_head, q_len, seq_len)`，其 layer/batch/head 与 cache 对齐、最后一维等于 `seq_len`、`q_len >= last_n_attentions`。
- 每个量化单元的非 outlier 元素至少 1 个（否则 \(\Delta\) 无从计算），`method="normal"` 走 std 则至少 2 个（ddof=1）；量化循环只遍历 `n_bits` 里真实出现过的位宽，不是整段 `n_bits_min..n_bits_max`。
- 全程 float64 累加，最后按 `cache.dtype` 返回；float32 与 float64 输入的 `n_bits` 完全一致，只有浮点值在各自容差内比对。

## 复杂度

设量化单元数 \(U\)、每单元元素数 \(M\)：排序类操作（`kthvalue`、amax/amin、mean/std）都是每单元线性，合计 \(O(UM)\)；量化按位宽分组、每组一次向量化的归一化-取整-反归一化，normal 额外付 searchsorted 的 \(O(M\log 2^{n})\)。总体仍是 \(O(UM\log M)\) 量级，没有与位宽上限相乘的项。

## 出处

`ClubieDong/QAQ-KVCacheQuantization`（arXiv 2403.04643）`src/quantizer.py::Quantizer`：

- `__init__` L59-64：`quantize_dims` 三档 token=(-3,-2,-1) / layer=(-2,-1) / head=(-1,)；
- `_calc_quantization_bits` L148-195：key 分支标量公式 L160；value 分支 L163-169 取末 `last_n` 行、permute 成 (n_batch, seq_len, n_layer, n_head, last_n) 后按量化维取 amax，L171 出 \(\varepsilon_{\max}\)；`scale_value` 对称 L177-179、非对称 L180-183；L187 `log2(scale/(2*max_error)+1)`；L191 `ceil → int64 → clamp`；L194-195 `if last_n>1: n_bits[:, -last_n+1:] = n_bits_max`；
- `_calc_outlier_mask` L198-212：L199-200 `ratio == 0` 短路、L204-205 双 `kthvalue`、L210 `(cache <= lower) | (cache > upper)`；
- `_normalize` L215-241 / `_uniform_quantize` L246-253 / `_normal_quantize` L255-262：masked 求中心与尺度（symmetric 的 `mean_value` 置 0，L219；std 走 masked `cache.std()`，L224/L232，即 ddof=1）；uniform 是 minmax 尺度 + `round` + clamp；normal 是 std 尺度 + `searchsorted` 落上界表再索引中心表；两者都以 `torch.where(outlier_mask, cache, ...)` 让 outlier 透传；
- `quantize` L269-284：`cache.permute(1, 3, 0, 2, 4)` L269 换到本题入参布局，L279-281 按位宽分组的 `cache[indices] = method(...)` 写回。

**本题自己的补充**（上游没有、判题环境需要，别记到论文/仓库头上）：**(a)** `max_error` 广播到与 `n_bits` 相同的每单元形状——上游 key 分支它就是个裸 Python 标量（靠除法隐式广播），本题显式铺成张量并断言逐单元同值；**(b)** `method="normal"` 的分位表原本由 `scipy.stats.norm.ppf` 在 `set_dtype_and_device` L117-124 生成，判题环境没有 scipy，故改成显式入参 `quantile_upper`/`quantile_center`；**(c)** `quantize` 的返回值 `average_n_bits` 与 `calc_quantized_cache_size_per_token` L286-298 的 cache 体积折算不在本题范围；**(d)** 上游靠 `assert`（L51-95 一堆参数校验），本题统一改抛 `ValueError`。另外本题入参已是 permute 之后的布局、`attentions` 直接给堆叠好的五维张量，不重跑模型。

## 相关题目

- `kv-quest-page-bound`：同样「用极值替真实计算」，Quest 用它做页选择，本题用它定每单元的存储位数；`quant-kivi-kv-asym` 也是 KV 非对称量化，但位宽固定，本题的位宽是逐单元反解出来的。
- `quant-gear-lowrank-outlier`：GEAR 用低秩项补偿 outlier 的量化误差，本题直接跳过 outlier、原值保留，不做补偿。
