# PEFT：GraLoRA 子块低秩前向与分块合并

## 题目描述

**GraLoRA**（Granular LoRA, arXiv:2505.20355，NeurIPS 2025 Spotlight，官方实现 `SqueezeBits/GraLoRA`，已进入 HuggingFace PEFT 的 `tuners/gralora`）解决 LoRA 的一个结构性缺陷：把秩 \(r\) 加大时 bottleneck 维度 \(r\) 同时变宽，容易过拟合。GraLoRA 把权重矩阵按输出/输入各划 \(k\) 段，每段配一对独立的低秩因子——**bottleneck 被切成 \(k\) 份，而总参数量与同秩普通 LoRA 完全相同**（\(r\cdot(d_\text{out}+d_\text{in})\)）。

本题复刻官方 `layer.py` 的两处确定性账：

- **布局账**（`update_layer`）：可用秩拆成 \(r_\text{gra}=r-r_\text{hyb}\)（可再留一段标准全秩 LoRA 旁路），每段秩 \(\text{sub}=r_\text{gra}/k\)；A 按**输入段**存放 \((k, d_\text{in}/k, r_\text{gra})\)、B 按**输出段**存放 \((k, r_\text{gra}, d_\text{out}/k)\)；scaling \(=\alpha/r\) 用**总秩**（含 hybrid），与普通 LoRA 同口径。
- **前向账**（`Linear.forward` 的双 einsum 链）：输入第 \(n\) 段过 \(A_n\) 升到 \(r_\text{gra}\) 维，rank 轴切成 \(k\) 段重排（permute 交换"段"与"输入块"两个轴后复合 reshape），再与 B 的第 \(j\) 个输出段的对应**行段**内积。

等价的分块公式是理解结构的钥匙：\(\Delta W\) 的 \((j,n)\) 块

\[W^{(j,n)} = B_j[n\cdot s{:}(n{+}1)\cdot s,:]^{\!\top}\, A_n[:,\,j\cdot s{:}(j{+}1)\cdot s],\qquad s=r_\text{gra}/k\]

每块秩只有 \(s\)——B 的第 \(n\) 行段配 A 的第 \(j\) 列段，这就是"granular bottleneck"。

## 规则

- `mode="split_dims"`：返回 `{"gralora_rank", "sub_rank", "sub_in", "sub_out", "scaling"}`；\(r\le 0\) 抛 `ValueError`；\(r_\text{gra}\%k\ne0\) 抛 `ValueError`（官方 assert）。
- `mode="param_layout"`：返回 A/B（及 hybrid 旁路 \(A_g:(r_\text{hyb},d_\text{in})\)、\(B_g:(d_\text{out},r_\text{hyb})\)）的 shape 与零初始化集合——**B 与 \(B_g\) 全零**（训练起点 \(\Delta W=0\)），A 与 \(A_g\) kaiming 非零。
- `mode="forward_delta"`：双 einsum 链直译（dropout=0），输入 \(x:(B,L,d_\text{in})\)，返回 \(scaling\cdot\Delta y:(B,L,k\cdot(d_\text{out}/k))\)。
- `mode="block_delta"`：用分块公式显式组装稠密 \(\Delta W:(d_\text{out},d_\text{in})\)。
- `mode="full_forward"`：gralora 分支 \(+\;scaling\cdot x\,A_g^\top B_g^\top\)；\(r_\text{hyb}>0\) 必须提供旁路因子，\(r_\text{hyb}=0\) 必须不提供。
- 全程 float64。

## 示例

\(d_\text{in}=6, d_\text{out}=8, r=4, k=2, \alpha=8, r_\text{hyb}=0\)：

```
split_dims -> {gralora_rank: 4, sub_rank: 2, sub_in: 3, sub_out: 4, scaling: 2.0}
param_layout -> A:(2,3,4), B:(2,4,4), zero_init=["gralora_B"]
```

取 \(x:(2,3,6)\)、\(A,B\) 为给定值（见测试锚点），`forward_delta` 的输出形状 \((2,3,8)\)，且与 \(x\cdot\Delta W^\top\cdot scaling\)（`block_delta` 组装的 \(\Delta W\)）逐元素一致。

## 约束

- 尺寸须满足 \(k \mid d_\text{in}\)、\(k \mid d_\text{out}\)、\(k \mid r_\text{gra}\)（官方在子块尺寸与 assert 上隐含）。
- hybrid 旁路的两个因子要么同时给、要么同时不给，与 \(r_\text{hyb}\) 一致，违者抛 `ValueError`。
- `block_delta` 结果必须与 `forward_delta` 数学等价（测试里互证）。
