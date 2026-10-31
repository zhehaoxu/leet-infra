# LayoutLM 家族 bbox 归一化与 2D 位置索引账

## 题意

LayoutLM 家族（LayoutLM/LayoutLMv2/LayoutLMv3，官方 `microsoft/unilm`，HF `layoutlm*`）把文档版面的文本框坐标作为模型输入：坐标先归一化到 0–1000，再查 x/y/h/w 位置嵌入，另有基于网格的 2D 位置索引。本题把这条 **bbox → 嵌入 id / 网格 id** 的纯索引账抽出来：给定像素坐标 boxes 与页宽高，输出归一化四角、宽高嵌入 id 与四角/中心的网格 id。

## 求解

口径题面钉死（与 HF `_calc_spatial_position_embeddings` 一致）：

1. **归一化到 0–1000**：x 方向 `round(1000 * x / page_w)`、y 方向 `round(1000 * y / page_h)`（Python `round`，即 **banker's rounding**），再 **clamp 到 [0, 1000]**（HF 无 clamp 只 raise；unilm 训练管线在喂入前 clamp，题面钉 clamp）；
2. **宽高嵌入 id**：`w_id = x1 - x0`、`h_id = y1 - y0`（归一化后；官方 `bbox[:,:,2]-bbox[:,:,0]` 与 `bbox[:,:,3]-bbox[:,:,1]`）；
3. **2D 网格索引**：`grid(x, y) = y * 1001 + x`（0..1000 共 1001 列）。每 box 输出四角网格 id `[g(x0,y0), g(x1,y0), g(x0,y1), g(x1,y1)]` 与中心 `g((x0+x1)//2, (y0+y1)//2)`。

返回 `{"norm_boxes", "wh_ids", "corner_grid", "center_grid"}`。

## 数据规模

- 1 ≤ len(boxes) ≤ 10^4，坐标可为任意有限数（越界靠 clamp）。
- boxes 空/条目非 4 元/含非数或 NaN/Inf、归一化后 x0 > x1 或 y0 > y1、page_w/page_h 非正数或 bool → `ValueError`。
