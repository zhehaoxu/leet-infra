# YOLOv5 build_targets 双扩展分配账

## 题面

YOLOv5 的正样本分配比"中心落格"宽得多：每个 GT 先按 anchor 形状粗筛，
再把落在格点的小数一侧**向邻居格扩展**（offset 0.5），一个 GT 最多同时
落到 5 个格点上。给定 GT 列表、各层 anchor 与特征图尺寸，复现官方
`build_targets` 的完整账（单图口径）。

给定：
- `targets`：nt×5 `[class, x, y, w, h]`（归一化；x,y∈[0,1]、w,h∈(0,1]）
- `anchors`：nl 层 × na anchor × `[w, h]`（像素单位）
- `shapes`：nl 层 `[H, W]`（特征图格数）
- `anchor_t`：wh 比阈值（默认 4.0）

## 处理链（每层独立）

1. **复制 + gain**：每 GT 复制 na 份并附 anchor 索引；gain =
   `[1, 1, W, H, W, H, 1]` 乘入格空间（x 用 W、y 用 H）。
2. **wh 匹配**：`r = 格空间 wh / 像素 anchor`（官方混搭口径），
   `max(r, 1/r) < anchor_t` 逐维取 max 后判（**恰等也过滤**）。
   ⚠️ 同一目标跨层 r 逐层 ×1/4（格宽 ÷2、anchor ×2），匹配资格逐层剧变。
3. **五方向扩展**：offsets = `[0,0], [+0.5,0], [0,+0.5], [−0.5,0],
   [0,−0.5]`。方向 fire 条件（x/y 独立）：
   - 同侧（j/k）：`坐标 % 1 < 0.5` **且** `坐标 > 1`；
   - 对侧（l/m）：`(W−x) % 1 < 0.5` **且** `(W−x) > 1`（y 用 H）。
   靠边（坐标或逆坐标 ≤1）不扩展；小数恰 0.5 不扩展。
4. **格点还原**：`gij = (gxy − offset).long()`——**截断非 round**。
   x 恰整数时 dir0 与 dir3 落同一格。
5. **clamp**：`gj∈[0,H−1]、gi∈[0,W−1]`（官方防御分支，双门下不触发）。
6. **输出**：indices = `[a, gj, gi]`；tbox = `[gx−gij_x, gy−gij_y, gw, gh]`
   （残差以截断格点为基准）；anch = 该 anchor 的 `[w, h]`。

## 输出 ledger（每层一项）

```json
{
  "gain":     [1, 1, W, H, W, H, 1],
  "wh_keep":  [bool × (na·nt)],                    // anchor-major
  "expanded": [[a, cls, gx, gy, gw, gh, dir], …],  // dir-major 展开序
  "indices":  [[a, gj, gi], …],
  "tbox":     [[dx, dy, gw, gh], …],
  "anch":     [[aw, ah], …]
}
```

## 输入约束

- targets 行 5 列有限；x,y∈[0,1]、w,h∈(0,1]；class 非负整数；
- anchors 每项 2 个正有限值；shapes 每项 2 个 ≥1 整数；anchor_t>0；
  违规抛 `ValueError`；空 targets 合法（全层空 ledger）。

## 样例

`anchors=[[[8,8]]], shapes=[[16,16]]`，target `[0, 14.6/16, 8.5/16,
0.2, 0.4]`：格空间 gx=14.6、gy=8.5、gw=3.2、gh=3.2（r=0.4 keep）。
gx%1=0.6 → j False；gxi=1.4（%1=0.4 且 >1）→ l True；y 侧双 False
→ dirs=[0,3]。dir0：gij=int(14.6)=14、残差 0.6；dir3：
int(14.6+0.5)=15、残差 0.1 → indices `[[0,4,14],[0,4,15]]`。
