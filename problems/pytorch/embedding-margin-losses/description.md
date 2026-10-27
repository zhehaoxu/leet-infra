# 嵌入 Margin 损失：CosineEmbeddingLoss 与 HingeEmbeddingLoss

## 题目描述

三元组之外，度量学习还有两种**成对**损失。CosineEmbeddingLoss（torch 的 `nn.CosineEmbeddingLoss`；学术渊源一般追到 Jia et al., *Large-scale Video Classification with Convolutional Neural Networks*, CVPR 2014 提出的 cosine embedding loss）只看两个向量的**夹角**：同类要求方向一致（\(1-\cos\)），异类要求夹角足够大（\(\max(0,\cos-\text{margin})\)），因此与向量长度无关，天然适合"把不同长度的句子/图像投到单位球面"。HingeEmbeddingLoss 则是标准 SVM hinge 作用在**已给定的成对分数**上（正对直接把分数当损失、负对要求分数低于 margin），常用于孪生网络的标量相似度输出。本题做一个 mode 分派器，把这两种损失 + 三种 reduction 一次写全。CPU、全程确定性、无随机数调用。

## 规则

- 完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。mode ∈ {`"cosine"`, `"hinge"`}，其余抛 ValueError；reduction ∈ {`"mean"`, `"sum"`, `"none"`}，其余抛 ValueError。
- **cosine 分支**：x1、x2 形 (N,D)，target 形 (N,) 且取值 ∈ \(\{1,-1\}\)（不校验）；`margin=None` 按 **0.0** 处理；x2 为 None 时抛 ValueError。记
  \[\cos_i=\frac{\langle x1_i,x2_i\rangle}{\max(\lVert x1_i\rVert_2\cdot\lVert x2_i\rVert_2,\;10^{-8})},\qquad
    \ell_i=\begin{cases}1-\cos_i,&\text{target}_i=1\\[2pt]\max(0,\ \cos_i-\text{margin}),&\text{target}_i=-1\end{cases}\]
  **clamp 作用在"两个模长的乘积"整体上**（不是各自先 clamp）；任一向量为零向量时分母被兜到 \(10^{-8}\) 而分子为 0，故 \(\cos_i=0\)、target=1 支损失恰为 1。
- **hinge 分支**：x1 是形 (N,) 的**成对分数**（不是向量），target (N,) ∈ \(\{1,-1\}\)；`margin=None` 按 **1.0** 处理；x2 约定传 None 且**一律忽略、不做校验**。
  \[\ell_i=\begin{cases}x1_i,&\text{target}_i=1\\[2pt]\max(0,\ \text{margin}-x1_i),&\text{target}_i=-1\end{cases}\]
  注意 target=1 支是**恒等**（不套 relu），所以负分数会产生**负损失**——这是 `F.hinge_embedding_loss` 的真实行为，判题用 `torch.equal` 钉死。
- **两个 margin 默认值不同**（cosine 0.0、hinge 1.0），都通过"传 None 即用各自默认"表达，与原生签名一致。
- **N=0 兜底**：`"none"` 返回 (0,) 空张量；`"sum"` 返回 0；`"mean"` 返回**标量 0**（原生对空集求均值给 NaN，本题约定为 0）。
- 返回 dtype 跟随 x1；`"none"` 时形状 (N,)，`"mean"/"sum"` 时 0 维标量。

## 示例

**cosine**：x1 \(=[[1,0,2],[0,1,1]]\)，x2 \(=[[1,1,0],[0,-1,2]]\)，target \(=[1,-1]\)，margin 默认 0.0，reduction="none"。
两对的点积都是 1、模长乘积都是 \(\sqrt5\cdot\sqrt2=\sqrt{10}\) → \(\cos=[0.3162278,\,0.3162278]\)：

- i=0（target=1）：\(\ell=1-0.3162278=\) **0.6837722063**
- i=1（target=−1）：\(\ell=\max(0,\,0.3162278-0)=\) **0.3162277639**

**hinge**：x1 \(=[2.5,-0.5]\)，target \(=[1,-1]\)，margin 默认 1.0，reduction="none"：

- i=0（target=1，恒等支）：**2.5**
- i=1（target=−1）：\(\max(0,\,1-(-0.5))=\) **1.5**

即 EXAMPLE_COSINE = [0.6837722063, 0.3162277639]、EXAMPLE_HINGE = [2.5, 1.5]。

## 约束

- mode 只接受 `"cosine"`/`"hinge"`；reduction 只接受 `"mean"`/`"sum"`/`"none"`；cosine 下 x2=None、非法 mode、非法 reduction 都抛 ValueError
- target 只约定 ∈ \(\{1,-1\}\)（long 张量），不做校验；其他取值按 `-1` 支处理（判题不考）
- 分母 clamp 阈值固定 \(10^{-8}\)，作用在模长**乘积**上；零向量与双零向量都必须给出有限值
- 对拍：cosine 与 `F.cosine_embedding_loss`、hinge 与 `F.hinge_embedding_loss` 在 margin ∈ {0.0,0.2,0.7}/{1.0,0.5} × reduction 三种下逐元素 \(10^{-6}\) 全等；`"none"` 与 `"mean"/"sum"` 的 shape/dtype 都是判题点
