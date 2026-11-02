# PaddleOCR DB 后处理单框账：box_score_fast 与 unclip

## 题意

DBNet（Differentiable Binarization）的检测头输出概率图，后处理把二值化区域变回文本框：阈值二值化 → 轮廓 → minAreaRect → 区域得分门 → unclip 外扩（Shapely/Clipper）→ 尺寸门 → 坐标缩放。本题手写 PaddleOCR `DBPostProcess.boxes_from_bitmap` 的 quad 单框全账：contour 已 `get_mini_boxes` 化为轴对齐矩形四点，不出真图、不调 cv2/shapely/pyclipper——三库语义全部实测钉死后纯标量复现。

## 求解

官方口径（PaddlePaddle/PaddleOCR `ppocr/postprocess/db_postprocess.py` 逐条钉死，shapely 2.2 + pyclipper 1.4 + cv2 5.0 实测）：

- **二值化**：`segmentation = pred > thresh`（严格 >，逐点）；
- **sside 门先行**：`get_mini_boxes` 的 `sside = min(span_x, span_y)`；`sside < min_size`（默认 3）→ skip，score/distance/expanded 均不算；
- **box_score_fast**：bbox 四向 `xmin=clip(floor(min x),0,W-1)`、`xmax=clip(ceil(max x),0,W-1)`（y 同 H）；mask 由 `cv2.fillPoly` 光栅化——**坐标 `astype(np.int32)` 向零截断**（-3.7→-3，非 floor 的 -4）后闭多边形（轴对齐矩形 = 闭矩形，含右/下边界），与 bbox 区域求交；`score = sum(prob·mask)/mask_sum`，**mask 全 0 时 cv2.mean = 0.0**（框完全出图侧）；
- **score 门**：`box_thresh > score` → skip（严格 >，score 恰等过门）；
- **distance**：`|shoelace| × unclip_ratio / 闭合周长`（shapely `Polygon.area` 顶点序无关取绝对值、`Polygon.length` 含末→首边）；
- **pyclipper JT_ROUND 外扩**：**输入坐标向零截断**（10.7→10），**输出坐标 round-half-away**（`floor(v+0.5)`；50.5→51，与 numpy 银行家的 50 相反）；轴对齐矩形外扩 = `x_lo = r(trunc(min x) − distance)`、`x_hi = r(trunc(max x) + distance)`（y 同）；
- **尺寸门**：unclip 后 `sside < min_size + 2` → skip；
- **dest 缩放**：`clip(round(x/W*dest_width), 0, dest_width)`——numpy round = **银行家舍入**（6.5→6、19.5→20），W 为 prob map 宽度。

## 数据规模

- 手算锚点 10 个：basic（dist=1.5 整、expanded [0,6]²）、cw 顶点序（|shoelace|）、thresh 恰等不计、浮点矩形（bbox floor/ceil 与 mask 截断两套坐标共存：bbox=[2,7,1,6] 而 mask [2,6]×[1,5]）、负坐标（bbox clip 0 + fillPoly 截断向零交集 2×3 px）、贴边、框全出图（score=0.0 → box 门）、单点退化（sside=0 → min_size_in）、细长框 sside_out 门（span 3/恰 4 两档）、score 恰等 box_thresh 过门、dest 银行家 0.5→0 锚点（1/6*3=0.5）与双界 clip；
- 随机对拍 200 组（oracle 为 numpy 向量化独立实现，坐标限官方可达域：整数或 .5 半整）。

## 评分

十二键账逐项核对：pos_count/sside_in/bbox/score/distance/expanded/sside_out/gate/passed/scaled 精确相等，score/distance 1e-9；bitmap 矩阵逐点相等。

## 红线

- 不得 import cv2 / shapely / pyclipper（AST 节点级检查）；纯标量账；
- 坐标入参限官方可达语义（整数或截断后相同的浮点——非整浮点坐标的 JT_ROUND 圆弧离散有 ±1 误差，不入题记 notes）。
