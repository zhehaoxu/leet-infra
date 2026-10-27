# 量化：NF4 分位码本构造与块编码——算法详解

## 1. 为什么是「分位点」而不是均匀格点

4-bit 均匀量化把 \([-1,1]\) 切成 16 等宽区间，但正态权重**中间密、两端稀**：均匀格点在 0 附近分辨率不足、在尾部浪费。NF4 改成**等概率**切分——每段承载 \(1/16\) 的权重质量，段宽自然在 0 附近变窄、尾部变宽，逐值 MSE 最小（Dettmers et al., arXiv:2305.14314 的消融：NF4 的重构误差显著低于 int4/fp4）。

## 2. bitsandbytes 的不对称构造（判题考点）

`create_normal_map(offset=0.9677083, use_extra_value=True)` 的实际做法（本题逐值复现，误差 \(10^{-7}\) 量级）：

- 正半边 **8** 个概率点：`linspace(offset, 0.5, 9)[:-1]`，步长 \((0.5-\text{offset})/8\)；
- 负半边 **7** 个概率点：`linspace(offset, 0.5, 8)[:-1]`，步长 \((0.5-\text{offset})/7\)；
- 补一个精确 \(0\)（正态对称中点），共 16 个；
- 排序后除以 `max|v|` 归一化到 \([-1,1]\)。

`offset` 是「最大分位点」的概率：`ppf(0.9677083)≈1.845`，即覆盖 \(\pm1.845\sigma\) 之外的尾部被饱和到 \(\pm1\)。正侧比负侧多一个点，所以码本**不关于 0 对称**——`check_codebook_anchors` 用「正侧首间距 < 负侧首间距」把这一结构钉死（若误写成对称的 `linspace(0.5,1,9)` 会立刻失败）。

精确表（float64）在 `test_cases.py` 的 `EXACT` 常量里：`-0.6961928 / 0.0795803 / 0.7229568 / 1.0` 等。**注意**：`peft-qlora-nf4-forward` 题面给的是论文附录 E 的四位小数表（`0.7226`），与这里的精确值差 \(3.6\times10^{-4}\)——两题各自题面内自洽，互不引用。

## 3. 块缩放编码（判题考点）

QLoRA 的 4-bit 存储 = 「16 值查表 + 每 64 权重一个 absmax」。本题只做**编码侧**（反量化前向在 `peft-qlora-nf4-forward`）：

\[
s=\max_{j\in\text{block}}|W_{ij}|,\qquad u=W_{ij}/s,\qquad c=\arg\min_k|u-T_k|,\qquad \hat W=T_c\,s
\]

- 并列取小码：`argmin` 的默认行为，题面明写，避免实现差异；
- 块 absmax 为 0（整块全零）：码固定 7、重建 0——`check_matches_oracle` 用全零行断言；
- **缩放线性**：块内权重整体乘 \(\lambda\) 时 \(u\) 不变、码不变、\(\hat W\) 精确乘 \(\lambda\)（浮点乘 2 无舍入）——`check_block_scaling_linearity` 用 \(\lambda=2\) 断言，这是「码本与缩放解耦」的结构不变量，任何把 scale 混进码本或反之的写法都会破坏它。

## 4. 实现要点

- `Φ⁻¹` 用 `torch.erfinv`：`Φ⁻¹(p)=√2·erfinv(2p−1)`；内部 float64 向量化。
- 编码用广播：`(blocks/safe).unsqueeze(-1) - table` 得 `(M, N/64, 64, 16)`，`argmin(-1)`；大张量下内存是 `16×`，判题尺寸（≤ 4×128）无压力。
- 退化块用 `torch.where` 覆盖码，避免 `0/0` 产生 NaN 污染 argmin。

## 5. 与本合集其他题的分工

- `symmetric-quantize` / `per-channel-quantize` / `group-wise-quantize`：线性均匀格点 + absmax 缩放；
- `gptq-column-quantize`：误差补偿的逐列量化；
- `peft-qlora-nf4-forward`：**给定** NF4 码与 absmax 的反量化前向 + double-quant 字节账；
- 本题：码本**怎么构造**（分位点 + 归一化）与**怎么编码**（最近邻查表）。量化误差分析、double quantization 的 scale 二级量化分别在 `quant-double-quant-scale`。

## 6. 验证方法

- `check_example`：\(1\times64\)、absmax=1，手算码 `[12,0,15,12]` 与重建值。
- `check_codebook_matches_reference`：与 bitsandbytes 精确表逐值对拍（容差 \(10^{-5}\)）。
- `check_codebook_anchors`：递增、\(\pm1\)、码 7 为零、不对称结构。
- `check_matches_oracle`：随机 \((4,128)\) + 全零行，逐标量重算对拍（码用 `equal`，重建用 \(10^{-5}\)）。
- `check_block_scaling_linearity`：缩放线性 + `offset` 敏感性 + 非法 `offset` 抛 `ValueError`。
