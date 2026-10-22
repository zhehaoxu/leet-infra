# 批量矩阵乘（Batched GEMM）详解

## 定义

`torch.matmul` 的批处理语义：对高维（>2D）张量，把**最后两维当矩阵**、前面所有维当 batch 广播：

$$ (\dots,m,k)\ @\ (\dots,k,p)\ \to\ (\dots,m,p) $$

一维情形有升/降维规则（1D@1D→标量、1D@2D→补行/列），本关聚焦 ≥2D。

## 广播规则（本关考点）

batch 维按**普通 broadcasting** 对齐：`torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])`。于是：

- $(2,3,4)@(2,4,5)\to(2,3,5)$
- $(3,4,5)@(1,5,6)\to(3,4,6)$（batch 维 $1$ 广播到 $3$）
- 单个矩阵 $(m,k)@(k,p)\to(m,p)$（batch 为空）

内维必须相等（$a_{\text{last}}=b_{\text{second-last}}$），否则报错——这是唯一硬约束。

## 逐元素实现（本参考解写法）

不依赖 `@`，用外积求和定义矩阵乘：

$$ C_{ij}=\sum_k A_{ik}B_{kj}\ \Rightarrow\ (a.\text{unsqueeze}(-1)*b.\text{unsqueeze}(-3)).\text{sum}(-2) $$

先 `expand` 把 batch 广播到公共形状，再 $a(\dots,m,k,1)\times b(\dots,1,k,p)$ 沿 $k$ 求和。

## 为什么单独练

所有注意力、FFN、反向的梯度计算都建立在 batched matmul 上；`einsum('...ik,...kj->...ij')` 是它的可读等价。理解广播与内维约束，是读懂 `bmm/matmul` 差异（bmm 严格要求 3D 且 batch 不广播）的基础。
