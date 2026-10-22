# GLU（Gated Linear Unit）详解

## 定义

把输入沿末维**劈成两半**，一半当值、一半当门：

$$ \text{glu}(x)=A\otimes\sigma(B),\qquad x=[A,B]\ \text{沿 dim 切分} $$

$\sigma$ 用 sigmoid（本关），也有用 ReLU/tanh 门的变体。

## 为什么重要

门控是 Transformer 现代变体的核心：GLU 家族（GeGLU、SwiGLU）把 FFN 的激活换成门控形式，效果优于普通 ReLU/GELU FFN。**LLaMA 用 SwiGLU**——门用 SiLU、并砍 $2/3$ 隐藏维以保持参数量。理解 GLU 是理解这些结构的前提。

## 实现与切分坑

1. **切分维度与形状**：要求 `x.shape[dim]` 为偶数，$A=x[\dots,0{:}h]$、$B=x[\dots,h{:}2h]$，$h=\text{shape}[dim]/2$。`chunk(2, dim)` 一步到位。
2. **门只作用在自己那半**：被门的是前半，门函数取后半，别弄反——反了数值全变。
3. **输出维度减半**：这是 GLU 的固有特性，不是 bug。
4. sigmoid 数值稳定：$\sigma$ 已内建保护，大负→0、大正→1。

## 对拍建议

参考实现用 `torch.sigmoid` 加 `chunk`，与手写一致。注意 `dim` 可为负数（如 `-1`），切片要用真实维度。
