# YOLOv3 anchor 系 box 编解码账

## 题面

YOLOv2/YOLOv3 用 anchor 先验：预测框 = anchor 加上网络修正量。官方 darknet
实现里，encode（训练 target）与 decode（推理还原）是**不对称**的——xy 的
target 是线性偏移而 decode 用 sigmoid，wh 才是 log/exp 严格互逆。给定网格、
anchor 先验与一组 truth/网络 raw 输出，请逐行复现官方账。

给定：
- `lw, lh`：网格列/行数（int ≥1）
- `img_w, img_h`：图像尺寸（归一化基准）
- `anchors`：像素单位 `[pw, ph]` 列表
- `truths`：归一化 `[x, y, w, h]`（中心格式）
- `preds`：与 truths 等长，每项 `[tx, ty, tw, th]` 为该格匹配 anchor 的
  raw（激活前）输出

## 处理链（对每条 truth 依序）

1. **best-anchor 匹配**：truth 中心移到原点，与每个 anchor 的归一化
   `(pw/img_w, ph/img_h)` 做 wh-only 中心对齐 IoU
   （`inter=min(w1,w2)·min(h1,h2)`，`union=w1·h1+w2·h2−inter`）；
   `iou > best_iou`（严格 >，初值 0）才更新 → 并列取小下标。
2. **落格**：`i=(int)(truth.x·lw), j=(int)(truth.y·lh)`（int 截断）。
3. **encode target**：`tgt_x=truth.x·lw−i`、`tgt_y=truth.y·lh−j`（线性
   偏移）；`tgt_w=log(truth.w·img_w/pw)`、`tgt_h=log(truth.h·img_h/ph)`
   （log 比，pw/ph 为匹配 anchor）。
4. **decode**：`bx=(i+σ(tx))/lw`、`by=(j+σ(ty))/lh`、
   `bw=e^tw·pw/img_w`、`bh=e^th·ph/img_h`。
5. **delta**：`scale=2−truth.w·truth.h`（归一化 truth）；
   `delta=[scale·(tgt_x−σ(tx)), scale·(tgt_y−σ(ty)),
   scale·(tgt_w−tw), scale·(tgt_h−th)]`。

## 输出

`{"matches": [{...}, ...]}`，按 truth 顺序，每项含：
`truth_index`、`anchor`（匹配 anchor 下标）、`cell: [i, j]`、
`target: [tgt_x, tgt_y, tgt_w, tgt_h]`、`pred_box: [bx, by, bw, bh]`、
`delta: [4 值]`、`anchor_iou`（匹配 IoU）。

## 输入约束

- `lw/lh` 为 int ≥1；`img_w/img_h > 0`；`anchors` 非空且每项为正二元；
  `truths/preds` 等长；truth 有限且 `w,h>0`；pred 为 4 个有限 float；
  违规抛 `ValueError`。

## 样例

`lw=lh=7, img=416, anchors=[[32,64]], truth=[0.55,0.6,0.2,0.3],
pred=[0,0,0,0]`：格 `[3,4]`；`tgt_x=0.85, tgt_w=log(0.2·416/32)`；
`bx=(3+0.5)/7=0.5, bw=32/416`；`scale=2−0.06=1.94`；
`delta[0]=1.94·(0.85−0.5)`。
