# PEFT：BitFit 偏置微调参数账——算法详解

## 1. 出处与核心思想

Ben Zakee & Awadalla（ACL 2022, arXiv:2106.10199）：只微调 bias。直觉有三：(1) bias 参数天然稀疏（BERT 量级模型里约占千分之一以下）；(2) bias 是激活的"平移自由度"，对分布偏移敏感——domain adaptation 文献早已发现只调 bias 就能大幅对齐分布；(3) 实现零成本。BERT/GLUE 上接近全量微调，few-shot 与中等数据上甚至更稳（全量微调过拟合）。BitFit 与 prompt 系方法正交，可叠加。

## 2. 参数账逐层口径（判题考点）

| 层型 | 参数总量 | BitFit 可训 | 说明 |
| --- | --- | --- | --- |
| Linear(\(a\to b\), bias=True) | \(ab+b\) | \(b\) | bias 是**输出维** |
| LayerNorm(\(a\)) | \(2a\) | \(a\) | 只有 \(\beta\)；\(\gamma\)(weight) 不可训 |
| Linear(\(a\to b\), bias=False) | \(ab\) | \(0\) | 无 bias 无可训 |

`check_example` 的表：(0,8,4)+(1,4,0)+(2,4,6)+(0,6,6) → 总参数 \(36+8+24+42=110\)、可训 \(4+4+0+6=14\)、ratio \(=14/110\approx 0.1273\)。

易错点（判题全都埋了断言）：

- **LayerNorm 的 \(\gamma\) 不是 bias**：把 \(2a\) 全算成可训是最常见错误——`check_ln_beta_only` 用全 LN 表钉死可训 \(=\Sigma a\)、总量 \(=2\Sigma a\)。
- **bias 是输出维**：Linear(8→4) 的 bias 是 4 个不是 8 个（example 第 1 行覆盖）。
- **无 bias Linear 贡献 0**：`check_no_bias_linear` 用全 kind=2 断言 ratio \(=0\)、可训 \(=0\)。

## 3. 与本合集其他题的分工

- `peft-prompt-tuning`/`peft-prefix-tuning`：参数账在"新加的可学张量"上；BitFit 的账在"基座子集"上——两类口径都要会。
- `peft-lora-forward` 等系：LoRA 的账是 \(r(d_\text{out}+d_\text{in})\)。
- 本题：纯参数审计、无前向——考的是对 Transformer 层型参数结构的记忆精度。

## 4. 验证方法

`check_example` 手算如上；`check_matches_oracle` 随机 20 层表与独立 dict 分派重算对拍（整数精确比较 + ratio 容差 \(10^{-5}\)）；两个专项 check 各覆盖单一层型。
