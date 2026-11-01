# YOLOv1 网格责任编码账

## 题面

YOLOv1 把检测写成回归：全图划 S×S 格，GT 中心落在哪个格，哪个格就对该
目标负责。但"编码"这件事在官方 C 实现里是一条容易踩坑的链——jitter 抖动
之后角落要截断，截断之后宽高必须**重新计算**；中心门是半开的；落格用
int 截断且没有 clamp；宽高门 0.01 是严格小于；同格冲突先到先得。

给定 `side`（S）、类别数 `classes`、label 列表（`[id, x, y, w, h]`，中心
与宽高均为图像归一化坐标，顺序即官方洗牌后的顺序）、jitter 参数
`[dx, dy, sx, sy]` 与 `flip` 开关，请逐行复现官方 encode 链，并按
`convert_detections` 的口径给出每格 decode 结果。

## 处理链（对每条 label 依序）

1. 角展开：`left=x−w/2, right=x+w/2, top=y−h/2, bot=y+h/2`。
2. jitter：`left=left·sx−dx` 等（右/下角同式）。
3. flip（若开）：`left, right = 1−right, 1−left`。
4. `constrain(0,1)`：逐角截断到 [0,1]。
5. **重新计算**中心与宽高：`x=(left+right)/2, y=(top+bot)/2, w=right−left,
   h=bot−top`——宽高是截断后两角之差，不是对原 w clamp。
6. 中心门：`x<=0 or x>=1 or y<=0 or y>=1` → 丢弃（半开区间，恰在边界
   也丢）。丢弃记 `{"index": i, "reason": "center_out"}`。
7. 落格：`col=(int)(x·S), row=(int)(y·S)`（int 截断；由于第 6 步，无需
   clamp）。格内偏移：`tx=x·S−col, ty=y·S−row`（先乘后减）。
8. 宽高门：截断后 `w<0.01 or h<0.01` → 丢弃（严格 `<`，0.01 保留）。记
   `reason: "too_small"`。
9. sqrt 编码：`tw=w^0.5, th=h^0.5`。
10. 同格仲裁：格 `(row, col)` 已占用 → 丢弃，先到先得。记
    `reason: "occupied"`。
11. 写格：`grid[f"{row},{col}"] = {"id", "x": tx, "y": ty, "w": tw,
    "h": th, "conf": 1.0}`（conf 目标恒为 1）。

## decode（互逆还原）

对每个占用格：`cx=(col+tx)/S, cy=(row+ty)/S, bw=tw², bh=th²`，存入
`decoded`（同 key，list 形式 `[cx, cy, bw, bh]`）。

## 输入约束

- `side`、`classes` 为 int ≥1；`jitter` 为 4 个有限 float（缺省
  `[0,0,1,1]`）；`flip` 为 bool；每条 label 的 `id ∈ [0,classes)` 且
  `w,h > 0`；违规抛 `ValueError`。

## 样例

`side=7, classes=20, labels=[[3, 0.55, 0.6, 0.2, 0.4]]`：中心 (0.55,0.6)
→ col=3, row=4；`tx=0.55·7−3≈0.85, ty≈0.2`；`tw=√0.2, th=√0.4`；输出格
`"4,3"`；decode 回 `[0.55, 0.6, 0.2, 0.4]`。
