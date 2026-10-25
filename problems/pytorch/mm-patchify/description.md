# 多模态：ViT 图像分块 Patchify

## 题目描述

Vision Transformer（Dosovitskiy et al., *An Image is Worth 16×16 Words*, arXiv:2010.11929）把一张 \(H\times W\) 的图像切成不重叠的 \(\text{patch}\times\text{patch}\) 小块，展平成一串「视觉 token」再送进 Transformer。本题实现这一步 **patchify**（不含线性投影，只做几何重排）。

## 规则

实现 `solve(image, patch)`。输入 `image`（float32，形状 \((B,C,H,W)\)，\(H,W\) 可被 `patch` 整除）。返回形状 \((B, N, C\cdot \text{patch}^2)\)，其中 \(N=(H/\text{patch})\cdot(W/\text{patch})\)。

排布约定：
- token 顺序按**块行优先**（先块行 `block_row`、再块列 `block_col`）；
- 每个 token 内的 \(C\cdot\text{patch}^2\) 维按 **(通道 c, 块内行 i, 块内列 j)** 展平（通道在最外）。

禁止用 `torch.nn.functional.unfold`/im2col 直接调包（可 `reshape`+`permute`）。

## 示例

`image`（\(1\times1\times4\times4\)）为 \(0..15\) 排成的 \(4\times4\) 矩阵、`patch=2`：

```
输入网格            输出 4 个 token（每个 4 维）
[[ 0  1 | 2  3]     [0,1,4,5] [2,3,6,7]
 [ 4  5 | 6  7]     [8,9,12,13] [10,11,14,15]
 ----------------
 [ 8  9 |12 13]
 [12 13|14 15]]
out shape = (1, 4, 4)
```

## 约束

- CPU float32；\(H,W\) 是 `patch` 的整数倍。
- 是纯重排（双射），信息无损——测试会用逆操作 unpatchify 还原对比。
