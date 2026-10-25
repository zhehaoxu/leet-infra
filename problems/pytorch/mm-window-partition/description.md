# 多模态：Swin 窗口划分 Window Partition

## 题目描述

Swin Transformer（Liu et al., *Swin: Hierarchical Vision Transformer using Shifted Windows*, arXiv:2111.03861）把特征图切成互不重叠的 \(\text{window}\times\text{window}\) 局部窗口，注意力只在窗口内计算，从而把复杂度从全局平方降到线性。本题实现 **window partition**（及其可逆性检查）。

## 规则

实现 `solve(x, window)`。输入 `x`（float32，形状 \((B,H,W,C)\)，\(H,W\) 可被 `window` 整除）。返回形状 \((B\cdot \frac{H}{\text{window}}\cdot \frac{W}{\text{window}},\ \text{window},\ \text{window},\ C)\)。窗口顺序按 **(batch, 窗口行, 窗口列)** 行主序；每个窗口保留内部 \((\text{row},\text{col},C)\) 的空间布局。是纯重排，禁止 padding。

## 示例

`x`（\(1\times4\times4\times2\)，值 \(0..31\)，`x[0,i,j]=[(i*4+j)*2,(i*4+j)*2+1]`）、`window=2`：

```
共 4 个窗口，按 (块行, 块列) 行主序；窗口0 覆盖 x[0, 0:2, 0:2, :]（2×2×2）：
  行0: [[0,1],[2,3]]   行1: [[8,9],[10,11]]
out shape = (4, 2, 2, 2)
```

## 约束

- CPU float32；\(H,W\) 是 `window` 的整数倍。
- 双射重排：可用逆操作（window merge）还原原图，测试据此校验。
