# RoPE（旋转位置编码）详解

## 核心思想

不给 token 加位置向量，而是把每对相邻维度看成二维平面上的向量，按其**位置** $m$ 旋转一个角度。旋转是正交变换，保持模长，且两个向量做点积时只依赖**相对角度差**——于是天然编码**相对位置**。

$$ \text{RoPE}(x,m)=\big(x_{2i}\cos m\theta_i - x_{2i+1}\sin m\theta_i,\ \ x_{2i}\sin m\theta_i + x_{2i+1}\cos m\theta_i\big)_{i} $$

$$ \theta_i = \text{base}^{-2i/d},\qquad i=0,\dots,\tfrac d2-1 $$

## 为什么是「相对位置」

同一头里 $q$ 在位置 $m$、$k$ 在位置 $n$ 旋转后做点积，旋转矩阵性质给出 $R_m q\cdot R_n k = q^\top R_{n-m} k$ —— 结果只含 $n-m$。这是 RoPE 让注意力对「距离」敏感、对「绝对位置」不敏感的关键，也是它能外推/配合 NTK-aware、YaRN 缩放的原因。

## 实现坑（本关易错点）

1. **配对方式**：本实现把维度按**相邻两两**配对（偶数维 `x[...,0::2]` 作 $x_1$、奇数维 `x[...,1::2]` 作 $x_2$）。注意还有「前后半配对」（GPT-NeoX/LLaMA 常见 `half` split）另一种约定，两者需配套不同的权重排列，别混用。
2. **角度频率**：$\theta_i=\text{base}^{-2i/d}$，指数是 $-2i/d$，别写成 $-i/d$（那是每半维不乘 2 的错误）；低频维变化慢、高频维变化快。
3. **位置要广播到批/头**：`positions.view(-1,1)*theta.view(1,-1)` 得 $(L,\text{half})$，再与 $x$ 的 $(...,L,\text{half})$ 相乘广播。
4. 拼回时用 `stack(...,dim=-1).flatten(-2,-1)` 保证维度交错顺序与切分时一致。

## 应用

LLaMA、GPT-NeoX、PaLM 等主流 LLM 全用 RoPE。理解「按对旋转 + 相对性」后再看 ALiBi（纯加性距离偏置，见「ALiBi」关）对比鲜明：一个改 Q/K、一个改 scores。
