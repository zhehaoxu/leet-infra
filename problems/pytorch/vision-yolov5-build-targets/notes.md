# YOLOv5 build_targets 双扩展分配账 — 实现笔记

## 回源

- 官方仓：ultralytics/yolov5 tag v7.0，`utils/loss.py::build_targets`。
- 论文无（工程实现），口径以代码为准；v6.x 与 v7.0 此函数一致。
- `self.anchors = m.anchors`：像素单位（相对输入图）。targets 归一化
  [img, class, x, y, w, h]；本题单图化取 [class, x, y, w, h]。

## 官方代码逐行账

```python
gain = torch.ones(7)                       # [1,1,gx?,gy?,gw?,gh?,1]
ai = arange(na).view(na,1).repeat(1,nt)
targets = cat((targets.repeat(na,1,1), ai[...,None]), 2)   # 复制+anchor idx
off = tensor([[0,0],[1,0],[0,1],[-1,0],[0,-1]]).float() * 0.5
for i in range(nl):
    anchors, shape = self.anchors[i], p[i].shape
    gain[2:6] = tensor(shape)[[3,2,3,2]]   # [W,H,W,H]（注意 3,2 顺序）
    t = targets * gain
    r = t[..., 4:6] / anchors[:, None]     # 格空间 wh / 像素 anchor
    j = torch.max(r, 1/r).max(2)[0] < anchor_t    # 双向比、严格 <
    t = t[j]
    gxy = t[:, 2:4]; gxi = gain[[2,3]] - gxy      # [W,H] − gxy
    j, k = ((gxy % 1 < g) & (gxy > 1)).T
    l, m = ((gxi % 1 < g) & (gxi > 1)).T
    j = stack((ones_like(j), j, k, l, m))
    t = t.repeat((5,1,1))[j]                      # dir 外层、行内层
    offsets = (zeros_like(gxy)[None] + off[:, None])[j]
    gij = (gxy - offsets).long()                  # 截断
    gi, gj = gij.T
    indices.append((b, a, gj.clamp_(0, shape[2]-1), gi.clamp_(0, shape[3]-1)))
    tbox.append(cat((gxy - gij, gwh), 1))
```

## 关键账点

1. **gain 顺序**：shape = (b, na, H, W, ·)，`shape[[3,2,3,2]] = [W,H,W,H]`；
   x 乘 W、y 乘 H。gain[0]/gain[1]/gain[6] 恒 1（img/class/anchor idx）。
2. **wh 匹配双向比**：`max(r, 1/r).max(2) < anchor_t`（默认 4.0）。
   - 恰等也过滤（`>=` 判否）。变异测试需 `max(r,1/r)==anchor_t` 边界锚点
     （格宽 32 vs anchor 8、anchor_t=4）。
   - **官方混搭 quirk**：r = 格空间宽 / 像素 anchor。同一物理目标跨层
     格宽 ÷2、anchor ×2 → r 逐层 ×1/4 → 匹配资格逐层剧变（很多目标
     只在一个层有资格）。与"每层独立合理"的直觉相反，照抄官方。
3. **offsets 双门**：方向 d（x+） firing = `gx % 1 < 0.5 且 gx > 1`；
   方向 l（x−） = `(W−gx) % 1 < 0.5 且 (W−gx) > 1`。x、y 各自独立。
   - x 恰整数：`%1=0<0.5` 且坐标与逆坐标都 >1 → 四方向全扩展（+dir0）。
   - 恰 =1.0 的坐标：`>1` False → 不扩展（`>=1` 变异需此锚点）。
   - `%1 = 0.5` 恰边界：不扩展（严格 <）。
   - 双门数学上保证扩展不越界（gx−0.5>0.5、gx+0.5<W+0.5 且参与 dir3
     的框 gx<W−1 → int ≤ W−1），**clamp 是防御死码**——`clamp 上界
     改 W` 变异等价，记 notes。
4. **gij 截断**：`.long()` 是 truncation（正数 floor）非 round：
   x=14.6、off=−0.5 → int(15.1)=15；x=4.5、off 无 → 4。x 恰整数时
   dir0 与 dir3 落同一格（int(1.0)=int(1.5)=1），tbox 残差不同。
5. **展开顺序**：`t.repeat((5,1,1))[j]` → dir 外层、行内层；本题
   expanded 记账同序。mask 行序决定输出顺序（变异可抓倒序）。
6. **tbox**：`(gxy − gij, gwh)`——残差以**截断后的格点**为基准
   （非 offsets）：x=16.0、dir1 → 16−15=1.0（格右半）；dir3 → 0.0。
7. **nt=0**：`t = targets[0]`（空）、offsets=0 → 全层无正样本。

## 出题口径

- 输入：targets（nt×5 [class,x,y,w,h] 归一化，class 非负整数）、
  anchors（nl×na×[w,h] 像素，正有限）、shapes（nl×[H,W]，≥1）、
  anchor_t（>0）。
- 输出 per layer：gain、wh_keep（na·nt，anchor-major）、expanded
  （[a, cls, gx, gy, gw, gh, dir]，dir-major）、indices（[a, gj, gi]
  clamp 后）、tbox（[dx, dy, gw, gh]）、anch。
- 校验：行长/有限/wh 正且 ≤1、x,y∈[0,1]、class 非负 int、anchor 正、
  shape≥1、anchor_t>0；违规 ValueError。
- oracle：按 (anchor, dir) 直接枚举（无复制-过滤结构）+ 200 组随机
  对拍（seed 707，anchor_t ∈ {1,2,4,8}）。

## 等价变异说明

- **clamp 上界改 W 等价**：双门保证 gij 界内，clamp 分支永不触发
  （官方防御性代码）。记 notes 不收。

## 变异测试

13 抓 + 1 等价：wh 比单向、anchor_t 边界（恰等锚点）、gain H/W 交换
（非方形锚点 [8,16]）、小数门 0.4、边门 >=1（恰 1.0 锚点）、逆坐标
W−1−x、offset 1.0、gij round、tbox 残差基准、方向倒序、anchor 索引、
gwh 不乘 gain、y 门用 gxy 判。
