# YOLOv1 网格责任编码账

YOLOv1 不用 anchor：全图划 S×S 格，GT 中心落在哪格、哪格就对该目标负责，
格内编码相对偏移，宽高走 sqrt 压缩。本题转写官方实现的完整账——不是论文
公式的理想化版本，而是 pjreddie/darknet commit `9db61832`（2015-07-20，
CVPR camera-ready 前后）里 `src/data.c` 的 `fill_truth_detection` 与
`src/detection.c` 的 `convert_detections` 逐行口径。

## solve 接口

```python
solve(side, classes, labels, jitter=None, flip=False)
```

- `labels`：`[id, x, y, w, h]` 列表，`x,y` 是**归一化中心**、`w,h` 是归一化
  宽高（`read_boxes` 口径）。列表顺序即官方 `randomize_boxes` 洗牌后的顺序
  ——随机序显式入参，本题不考随机。
- `jitter = [dx, dy, sx, sy]`：官方训练时的平移/缩放抖动，作用于角。
- 输出 `{"grid": {"r,c": {...}}, "decoded": {...}, "dropped": [...]}`。

## encode 侧（fill_truth_detection）

对每条 label 依序执行：

1. **角展开**：`left=x−w/2, right=x+w/2, top=y−h/2, bot=y+h/2`。
2. **jitter 变换**：`left=left·sx−dx`（右/下角同式）。
3. **flip**：`left, right = 1−right, 1−left`（先取 1− 右，再取 1− 左；
   注意新 left 来自旧 right）。
4. **constrain(0,1) 逐角截断**。
5. **重新计算**：`x=(left+right)/2, y=(top+bot)/2, w=right−left,
   h=bot−top`。核心账点：`w,h` 是**截断后两角之差**——某角被截断时宽高
   随之变化，绝不是 `clamp(w, 0, 1)`。
6. **中心半开门**：`x<=0 or x>=1 or y<=0 or y>=1 → continue`。恰在 0 或
   1 的中心直接丢弃（半开区间 (0,1)，不是 clamp）。
7. **落格**：`col=(int)(x·S), row=(int)(y·S)`——C 的 int 截断。第 6 步
   已保证 `x·S ∈ (0,S)`，所以官方**没有 clamp**；中心恰在格线上（如
   x·S=1.0）合法落下一格、偏移 0。
8. **格内偏移**：`tx=x·S−col, ty=y·S−row`——先乘后减，浮点口径敏感
   （不等于 `x−col/S`）。
9. **宽高门**：截断后 `w<0.01 or h<0.01 → continue`（严格 `<`，0.01
   本身保留）。
10. **sqrt 编码**：`tw=w^0.5, th=h^0.5`（官方 `pow(w,1./2.)`；被注释的
    `sqrt(w)` 与之等价）。
11. **同格仲裁**：`if(truth[index+classes+2]) continue;`——该格已写过
    object（truth 槽非零）则后来 GT 丢弃，**先到先得**，顺序即洗牌序。

truth 槽布局 `(4+classes)`：one-hot classes + `[tx, ty, tw, th]`。
**confidence 目标恒为 1**——truth 数组里没有 conf 槽，v1 的 conf 监督由
loss/objectness 分支统一处理；官方还有 background 标志槽（1=背景默认）
与 rescore（conf 回写 IoU）两套口径，均不入题。

## decode 侧（convert_detections）

行主序展开：`row=i//S, col=i%S`，然后

```
cx = (px + col) / S      # px = 网络输出 = tx
cy = (py + row) / S
bw = px^2                # 平方还原 sqrt 编码
bh = py^2
```

与 encode 互逆（本题 decode 输出归一化坐标，不乘图像像素 w/h）。

## 判题细节

- 浮点精确边界：`0.505−0.495 = 0.010000000000000009 > 0.01`，所以"w
  恰 0.01 保留"这条锚点需要专门构造——`l=0.0, r=0.01`（如
  `x=0.005, w=0.01`）才是浮点精确命中。
- oracle 为数学等价的直算式实现：decode 直接写截断后的 `x,y,w,h`
  （往返平方的浮点差 ~1e-17，tol 1e-9 内一致），与 solution 的 C 链
  重放互证；300 组随机（S∈{1,2,3,5,7}、含 jitter/flip）对拍。

## 划界

- `vision-anchor-match`：IoU 阈值 anchor 匹配，有 anchor 先验；本题无
  anchor 网格系。
- `vision-yolo-anchor-kmeans`：anchor 先验的聚类来源；本题是网格责任。
- background 标志槽、rescore、loss 侧 IOU 还原不入题。

## 官方 loss 侧 quirk（记录，未入题）

`detection_layer.c` forward 里 `truth.x = truth[j+0]/7`——把格内偏移
直接除以 S 当图坐标，**没加 col**（S 硬编码 7）。这是 IOU/diou 估计的
粗糙近似，且 x/y 分量梯度随后被 `4*(truth−output)` 覆盖，实为死码级
quirk；检测输出侧 `convert_detections` 才是正确的 `(px+col)/S`。

## 变异记录（12 条全抓）

constrain 后不重算 w、中心门改闭开、col 四舍五入、偏移 +0.5、宽高门
改 `<=`（靠浮点精确边界锚点抓）、encode 改平方、同格后到覆盖、flip
不交换、decode 行列互换、decode 不平方、int 改 round、conf 改 0.5。
