# PEFT：LoKr Kronecker 因子分解——算法详解

## 1. 为什么 LoKr 需要因式分解

LoKr（Yeh et al., arXiv:2108.07383；PEFT `tuners/lokr/layer.py`）把增量写成 Kronecker 积：

\[\Delta W = s\,(W_1\otimes W_2),\qquad s=\alpha/r\]

若 \(W_1\in\mathbb R^{a\times b}\)、\(W_2\in\mathbb R^{c\times d}\)，则 \(W_1\otimes W_2\in\mathbb R^{(a\,c)\times(b\,d)}\)。要让 \(\Delta W\) 与基座同形，必须 \(a\,c=d_\text{out}\)、\(b\,d=d_\text{in}\)——维度对被拆成两个乘子，拆法不唯一，`factorization(dimension, factor)` 就是选定拆法的确定性规则。

## 2. PEFT 源码规则逐步解读

`factorization`（Linear 分支，源码照录逻辑）：

1. **`factor > 0` 且整除**：`m = factor, n = dimension // factor` 直接返回，**不做** `m<=n` 交换。因此 `factorization(12, 6) == (6, 2)` 是合法输出——调用方只需要乘积对得上，不关心顺序。
2. **`factor == -1`（默认）**：从 `m=1, n=d` 出发，逐步把 `m` 提升到下一个因子，只在 \(m+n\) 不增大时接受——而 \(m+n\) 在 \(m\le\sqrt d\) 段单调下降，所以收敛点就是「最平衡」的因子对；最后保证 \(m\le n\)。等价闭式描述：\(m\) 取不超过 \(\sqrt d\) 的**最大**因子。对质数 \(d\)，唯一非平凡对是 \((1,d)\)。
3. 本题把「`factor>0` 但不整除 / 非法 `factor` / 非法 `dimension`」规范化为 `ValueError`（PEFT 原版此时会静默落到搜索分支，判题上收窄为显式报错，题面已声明）。

## 3. 拆法影响什么

- 参数量：full 形态共 \(m_1 n_1 + m_2 n_2\)（\(W_1,W_2\) 元素数），越平衡越接近 \(2\sqrt{d_\text{out}d_\text{in}}\) 量级、比 \(d_\text{out}d_\text{in}\) 小得多。
- \(W_1\) 还可再低秩化（`use_w1=False` 时 \(W_1=W_{1a}W_{1b}\)，即 `decompose_both`/`use_w_decomp` 开关），本题不展开。
- LyCORIS 风格初始化把 \(W_2\)（或 `w2_b`）置零 → 起点 \(\Delta W=0\)，与 LoHa 的置零退化同一思想。

## 4. 判题要点

- 显式分支考「**原序**」：`(6, 2)` 不等于 `(2, 6)`，顺序交换会被 `check_factor_divides_direct` 抓住。
- 自动分支考「最平衡 + \(m\le n\)」：\((16,16)\)、\((8,16)\)、\((6,8)\)、质数 \((1,d)\)。
- oracle 用全因子暴力枚举 + 字典序最小，与参考解的单调扫描结构独立。
