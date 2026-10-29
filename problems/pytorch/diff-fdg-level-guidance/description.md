# 扩散采样：FDG 的逐频层解耦引导

## 题目描述

CFG 对整张 latent 只用一个引导尺度，但**低频和高频对引导的需求正好相反**：低频（色块、构图）推得太狠会丢掉多样性，高频（细节、纹理）推得不够又显得糊。**Frequency-Decoupled Guidance**（FDG）的做法是先把条件/无条件预测都拆成频带，在频带空间里**各自做一次 CFG**，最后再重建回数据空间：

\[ \text{pred}=\sum_l \text{pred}_l,\qquad \text{pred}_l=u_l+s_l\,(c_l-u_l) \]

每个频带有自己的尺度 \(s_l\)（论文推荐高频大、低频小，例如 SDXL 上高频 10、低频 5）。此外 FDG 还有两个可选旋钮：

- **平行分量加权** \(w_l\)：把本层的差分 \(c_l-u_l\) 相对条件预测 \(c_l\) 拆成平行、正交两部分，平行部分乘 \(w_l\) 后再与正交部分相加。\(w_l=1\) 即标准 CFG；取 0 表示只用正交方向，等价于「只改方向、不改幅度」。
- **逐层 rescale** \(\varphi_l\)：按标准差的比值把该层的引导结果拉回条件预测的尺度，再与未缩放结果按 \((1-\varphi_l,\varphi_l)\) 混合，用来抑制高引导尺度下的过曝。

本题把频带拆解/重建的算子（Laplacian 金字塔的 `pyrup` 上采样模糊）当作外部前提：输入就是**已拆好的、同形状的频带**，反变换相应退化为逐层相加。

## 规则

输入为两个等长张量列表 `cond_levels` / `uncond_levels`，按「高频 → 低频」排列，每层形状都是 `(batch, ...)` 且所有层同形状；以及三个可选列表 `parallel_weights`、`level_rescale`、`active_levels`（缺省分别为全 1、全 0、全 `True`）。

对第 \(l\) 层：

- 若该层**未开启**：\(\text{pred}_l=c_l\)。
- 否则：
  1. \(\delta_l=c_l-u_l\)；
  2. 若 \(w_l\) 与 \(1.0\) 不相近（`math.isclose` 默认容差）：
     \[ \hat{c}_l=\frac{c_l}{\max(\lVert c_l\rVert_2,\,10^{-12})},\quad p=(\delta_l\cdot\hat c_l)\,\hat c_l,\quad o=\delta_l-p,\quad \delta_l=w_l\,p+o \]
     范数与内积都在除 batch 外的全部维度上归约；
  3. \(\text{pred}_l=u_l+s_l\,\delta_l\)；
  4. 若 \(\varphi_l>0\)：令 \(\sigma_{\text{ref}}\)、\(\sigma_{\text{pred}}\) 为 \(c_l\)、\(\text{pred}_l\) 在除 batch 外全部维度上的**无偏**标准差（`torch.std` 默认口径），则
     \[ \text{pred}_l=(1-\varphi_l)\,\text{pred}_l+\varphi_l\left(\text{pred}_l\cdot\frac{\sigma_{\text{ref}}}{\sigma_{\text{pred}}}\right) \]

最后返回所有 \(\text{pred}_l\) 之和，float32，形状与单个频带相同。

## 示例

两层频带（高频在前）：

\[ c^{(0)}=\begin{bmatrix}1.0&-2.0&0.5\end{bmatrix},\quad u^{(0)}=\begin{bmatrix}0.5&-1.0&0.0\end{bmatrix},\quad c^{(1)}=\begin{bmatrix}0.2&0.4&-0.6\end{bmatrix},\quad u^{(1)}=\begin{bmatrix}0.1&0.2&-0.2\end{bmatrix} \]

尺度取 \(s=(10,\,5)\)：

```
默认             -> [[6.100000, -9.800000, 2.800000]]
平行权重 [0, 1]  -> [[0.861905, 0.676190, 0.180952]]
逐层 rescale 0.5 -> [[3.608426, -5.666953, 1.507397]]
```

第一栏就是两层各自做 CFG 后相加；第二栏把高频层的平行分量完全丢掉，只剩相对 \(c^{(0)}\) 正交的方向；第三栏把两层各按一半强度拉回条件预测的尺度。

## 约束

- 两个列表非空、等长，每层至少两维（第 0 维是 batch），所有层同形状；同一层的 cond / uncond 也必须同形状。
- `guidance_scales`、`parallel_weights` 每项为有限非负实数；`level_rescale` 每项落在 \([0,1]\)；`active_levels` 每项必须是 `bool`（`0/1` 不接受）。长度不符一律抛 `ValueError`。
- **退化性质**：所有层尺度相同且 \(w_l\equiv1\)、\(\varphi_l\equiv0\) 时，输出必须等于「把各层求和后再做一次 CFG」——即 \(\sum_l u_l+s\,(\sum_l c_l-\sum_l u_l)\)。
- \(\varphi_l>0\) 时该层每个样本的预测不能是常数（否则 \(\sigma_{\text{pred}}=0\)，无法拉回），必须抛 `ValueError` 而不是产生 `NaN`。
- 关闭的频率层透传**它自己的**条件层，而不是相邻层的中间结果。
- 计算内部可用 float64，返回 float32。
