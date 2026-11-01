# YOLOv3 anchor 系 box 编解码账

YOLOv2 起引入 anchor 先验：网格只回归相对 anchor 的修正量。本题转写
pjreddie/darknet `f86901f6`（master 最终版）`src/yolo_layer.c` 的完整账——
`get_yolo_box`（decode）、`delta_yolo_box`（encode target）、forward 内联的
best-anchor 匹配——重点钉死官方口径里 encode 与 decode 的**不对称**。

## solve 接口

```python
solve(lw, lh, img_w, img_h, anchors, truths, preds)
```

- `anchors`：像素单位 `[pw, ph]` 列表（darknet biases）。
- `truths`：归一化 `[x, y, w, h]`（中心格式）。
- `preds`：与 truths 等长，每项 `[tx, ty, tw, th]` 是该 truth 落格处、
  匹配 anchor 的**激活前 raw 输出**（官方 tx/ty 在 forward 里做 LOGISTIC、
  tw/th 不激活，因此这里统一给 raw，激活在 decode 内做）。

## 1. best-anchor 匹配（forward 内联）

```c
i = (truth.x * l.w); j = (truth.y * l.h);
truth_shift.x = truth_shift.y = 0;      // 中心移原点
pred.w = l.biases[2*n]/net.w; ...       // anchor 归一化
iou > best_iou 才更新                     // 严格 >：并列取小下标
```

wh-only、中心对齐：`inter=min(w1,w2)·min(h1,h2)`，`best_iou` 初值 0。
（`best_iou` 改 −1 是**等价变异**：IoU≥0 恒成立，选择与输出完全相同。）

## 2. 落格

`i=(int)(truth.x·lw), j=(int)(truth.y·lh)`——int 截断，官方无 clamp。

## 3. encode target（delta_yolo_box）

```
tgt_x = truth.x·lw − i              # 线性偏移，不是 logit(σ⁻¹)！
tgt_w = log(truth.w·img_w / pw)     # log 比
```

**核心不对称**：decode 用 `σ(tx)`，encode target 却是线性差——xy 的
encode/decode 往返**不闭合**（官方 loss 就是对 σ(tx) 做 L2，target 在
σ 输出空间）；wh 的 log/exp 严格互逆（`e^tgt_w·pw/img_w == truth.w`）。

## 4. decode（get_yolo_box）

```
bx = (i + σ(tx)) / lw
by = (j + σ(ty)) / lh
bw = e^tw · pw / img_w              # biases 像素 / 图尺寸归一化
bh = e^th · ph / img_h
```

通道布局：`stride = lw·lh` 通道分离（本题输出按逻辑值，不展开 stride）。

## 5. delta

`delta = scale · (target − output)`，`scale = 2 − truth.w·truth.h`（归一化
truth）；xy 的 output 是 `σ(tx)`、wh 的 output 是 raw。

## 判题细节

- anchor 并列：严格 `>`，首个最大者胜（小下标）。
- `basic_single` 锚点：raw 全 0 → σ=0.5 → `bx=(i+0.5)/lw`；wh delta
  即 `scale·tgt_w`（raw=0）。
- oracle 为函数式直算（`max(range, key=(iou, -n))` 组织），250 组随机
  对拍（lw/lh∈{4,7,13}、1–5 anchor、1–4 truth）。

## 划界

- `vision-yolov1-grid-responsibility`：无 anchor 网格系（sqrt 编码、
  线性偏移、先到先得）；本题 anchor 系（exp/log 编码、IoU 匹配、
  sigmoid 偏移）。
- `vision-yolo-anchor-kmeans`：anchor 先验的聚类来源。
- mask（多尺度 anchor 子集）、objectness/classes 通道、NMS 后处理不入题。

## 变异记录

12 条非等价全抓：并列改后到覆盖、匹配改 1−|log 比|（k-means 口径混淆）、
decode 不激活、bw 除网格宽、tgt_x+0.5、tgt_w 括号错、scale 改 1−wh、
delta 加法、落格 round、分母互换、wh encode 不除 anchor、匹配取最后。
另有 1 条**等价变异**（`best_iou=−1` 初值）记 notes 不收。
