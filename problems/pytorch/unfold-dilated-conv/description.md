# Unfold 膨胀卷积取块（Dilated Patch Extraction）

## 题目描述

卷积的 im2col 视角：把输入特征图上每个输出位置对应的感受野"块"抽出来。加上膨胀率 `dilation` $d$ 后，块内相邻采样点在原图上相距 $d$ 个像素（中间跳过 $d-1$ 个）。

$$ \text{block}[r,w,i,j] = x[..., \; r + i\cdot d,\; w + j\cdot d],\quad 0 \le i < k_H,\ 0 \le j < k_W $$

输出位置数（stride=1、无 padding）：

$$ o_H = H - (k_H - 1)d, \qquad o_W = W - (k_W - 1)d $$

## 规则

- 实现 `solve(x, kernel_size, dilation=1)`，`x` 形状 `(B, C, H, W)`，`kernel_size=(kH, kW)`。
- 返回形状 `(B, C, oH, oW, kH, kW)` 的张量：最后两维是每个块内部的 $k_H \times k_W$ 采样，遵循上面的膨胀采样公式。
- `dilation=1` 时退化为普通连续取块。
- 可用 `Tensor.unfold`：先 unfold 出跨度 $(k-1)d+1$ 的连续窗口，再在窗口内以步长 $d$ 抽样。

## 示例

```
输入：x = 1×1×3×4, 值 [[0,1,2,3],[4,5,6,7],[8,9,10,11]], kernel=(2,2), dilation=2
   oH=3-(2-1)*2=1, oW=4-2=2 → 形状 (1,1,1,2,2,2)
   块(0,0) 采样 行{0,2}×列{0,2} = [[0,2],[8,10]]
   块(0,1) 采样 行{0,2}×列{1,3} = [[1,3],[9,11]]
```

## 约束

- float32/float64 输入；$(k_H-1)d < H$、$(k_W-1)d < W$
