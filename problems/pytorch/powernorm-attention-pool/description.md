# PowerNorm 与加性注意力池化读出

## 题目描述

归一化与池化是序列模型读出头前的两件常备工具。PowerNorm（Shen et al., *PowerNorm: Rethinking Batch Normalization in Transformers*, ICML 2020, arXiv:2003.07845）把 BatchNorm/LayerNorm 的"减均值、除标准差"砍成只除一个**二阶 power mean**：对每个空间位置，用通道平方的均值 \(S=\frac1C\sum_c x_c^2\) 做分母，\(y=x/\sqrt{S+\varepsilon}\)——不减均值、不估方差，只用一个前向统计量，代价是对非零均值的输入**永不中心化**。池化一侧，Set Transformer 的 PMA 读出（Lee et al., *Set Transformer*, arXiv:1810.00825）用 seed 向量对集合元素做注意力加权求和；本题实现其单查询退化版：\(\mathrm{out}=\mathrm{softmax}(xq/\sqrt D)^\top x\)。两种 mode 二选一，且**互相忽略对方的参数**。CPU、全程确定性、无随机数调用。

## 规则

- 完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。两种 mode：`"powernorm"` / `"attn-pool"`，其他取值抛 ValueError；输入只接受 float32/float64 张量（其他 dtype 抛 ValueError）；两种 mode 对维度数各有要求，不符抛 ValueError。
- **powernorm**（纯二阶矩口径）：输入是 (N,C,L) 3-D 或 (N,C,H,W) 4-D；先展平成 (N,C,M)（4-D 时 \(M=H\cdot W\)，空间维整体摊平），对每个 (n,m) 位置算通道均方 \(S(n,m)=\frac1C\sum_c x[n,c,m]^2\)，再

  \[y[n,c,m]=x[n,c,m]\,/\,\sqrt{S(n,m)+\varepsilon}.\]

  **不做任何减均值/中心化**（判题有专门探针）；\(\varepsilon>0\) 只进分母、防止零输入除零（零输入 → \(0/\sqrt\varepsilon=0\)）。返回与输入同形状、同 dtype。
- **attn-pool**（单查询注意力池化，无温度参数）：输入是 (B,L,D) 3-D，查询向量长度须等于 D 且允许与输入 dtype 不同（先转换到输入 dtype 再参与计算）；打分

  \[\text{logits}[b,l]=\Big(\sum_d x[b,l,d]\,q_d\Big)\big/\sqrt D,\]

  沿 L 维（dim=-1）做减最大值的数值稳定 softmax 得到权重 (B,L)，输出 \(\text{out}[b,d]=\sum_l \text{attn}[b,l]\,x[b,l,d]\)，形状 (B,D)、dtype 同输入。
- **两 mode 互相忽略对方的参数**：powernorm 下查询向量完全不参与（哪怕取值不合法也不影响结果）、attn-pool 下 eps 完全不参与；attn-pool 缺查询向量、查询非一维或长度 ≠ D 抛 ValueError，eps 必须为正数否则抛 ValueError（仅 powernorm 检查）。
- softmax 必须数值稳定（减最大值再 exp 归一化）；数学性质：所有 logits 相等 → 权重均匀 \(1/L\)、out == 逐元素均值；L=1 → out 与该行**逐位相等**。

## 示例

输入 (1,2,4)（即 \(N{=}1,C{=}2,L{=}4\)），eps 取默认 \(10^{-6}\)：

\[x=\begin{bmatrix}1&2&3&4\\5&6&7&8\end{bmatrix}\]

**powernorm**：逐列算 \(S=(x_0^2+x_1^2)/2=[13,\,20,\,29,\,40]\)，分母 \(\sqrt{S+10^{-6}}\approx[3.60555,\,4.47214,\,5.38516,\,6.32456]\)，输出（test_cases.py 回填的 EXAMPLE_PN，逐位到 10 位小数）：

\[y=\begin{bmatrix}0.2773500681&0.4472135901&0.5570859909&0.6324555278\\1.3867504597&1.3416407108&1.2998672724&1.2649110556\end{bmatrix}\]

**attn-pool**：同一矩阵视作 (B=1,L=2,D=4)、查询 \(q=[1,-1,0.5,0]\)：logits \(=[0.25,\,1.25]\)，softmax 后权重 \([1/(1{+}e),\,e/(1{+}e)]=[0.2689414214,\,0.7310585786]\)，输出（EXAMPLE_ATTN）：

\[\mathrm{out}=[3.9242343903,\;4.9242343903,\;5.9242343903,\;6.9242343903]\]

对照：逐元素均值是 \([3,4,5,6]\)——只有打分全等时 out 才塌回均值，此处 logits 差 1.0，权重向第二行倾斜。

## 约束

- mode 只接受 `"powernorm"`/`"attn-pool"`；eps 仅 powernorm 使用且必须为正；attn-pool 的查询向量必须是一维、长度等于 D、dtype 可与输入不同；维度数不符（powernorm 非 3-D/4-D、attn-pool 非 3-D）与非法 dtype 一律 ValueError
- 输出：powernorm 与输入同形状同 dtype；attn-pool 为 (B,D)、dtype 同输入；float64 案例对拍容差 1e-12，float32 案例 1e-6
- 判别探针直接断言：非零通道均值的输入下 powernorm **不减均值**（与 BatchNorm 式 \((x-\mu)/\sqrt{\mathrm{var}+\varepsilon}\) 显著不同，且输出通道均值明显非零）；eps 极小时每个 (n,m) 位置的通道 RMS ≈ 1，eps 变大后按 \(\sqrt{S/(S+\varepsilon)}\) 偏离（理论值逐点对齐）；常数输入 → \(\mathrm{sign}(c)\)、零输入 → 0、逐元素保号；查询与每个元素内积相等 → out == mean；L=1 → out 与该行 torch.equal
