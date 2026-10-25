# 多模态：Qwen2-VL 2×2 视觉 Token 合并

## 题目描述

Qwen2-VL（Bai et al., arXiv:2409.12191）的 **PatchMerger** 用「2×2 空间合并」把视觉 token 数压到 1/4、同时把通道数扩到 4 倍：把 \(h\times w\) 的 patch 网格每相邻 \(2\times2\) 块的嵌入沿特征维拼接，之后再过 MLP 投影（本题只做这一步确定性的 **空间 shuffle + concat**，不含带权重的 Linear）。

## 规则

实现 `solve(image_embeds, grid_h, grid_w)`。输入 `image_embeds`（float32，形状 \((g_h\cdot g_w,\ D)\)，按行主序展平的 patch 序列，\(g_h,g_w\) 为偶数）。返回 \((\frac{g_h}{2}\cdot\frac{g_w}{2},\ 4D)\)：第 \((i,j)\) 个输出 token ＝ 网格位置 \((2i,2j),(2i,2j\!+\!1),(2i\!+\!1,2j),(2i\!+\!1,2j\!+\!1)\) 四个 patch 的 \(D\) 维嵌入按此顺序拼接。输出 token 之间按 (合并行, 合并列) 行主序。禁止 `einops`。

## 示例

\(g_h=g_w=2,\ D=2\)，`image_embeds=[[0,1],[2,3],[4,5],[6,7]]`（依次是网格 (0,0),(0,1),(1,0),(1,1)）：

```
唯一合并块 = 四个 patch 依序拼接
out = [[0,1, 2,3, 4,5, 6,7]]   shape = (1, 8)
```

## 约束

- CPU float32；`grid_h`、`grid_w` 均为偶数。
- 合并顺序固定为 左上→右上→左下→右下（即行主序的 2×2）。
- 纯重排 + 拼接：输出元素是输入的置换、无信息损失。
