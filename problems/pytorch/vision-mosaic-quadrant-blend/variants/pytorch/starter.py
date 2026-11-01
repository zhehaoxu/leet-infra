"""YOLOv4 Mosaic 增强 truth 侧账（starter）。

口径钉死（AlexeyAB/darknet src/data.c blend_truth_mosaic + mosaic 图像
拷贝分支；shifts/cut_x/cut_y 显式入参，不考随机生成）：

- **象限平移**（对每条 old truth 的归一化中心）：四分支方向不对称——
  ```
  0(左上): xb −= (w−cut_x−right_shift)/w, yb −= (h−cut_y−bot_shift)/h
  1(右上): xb += (cut_x−left_shift)/w,    yb −= (h−cut_y−bot_shift)/h
  2(左下): xb −= (w−cut_x−right_shift)/w, yb += (cut_y−top_shift)/h
  3(右下): xb += (cut_x−left_shift)/w,    yb += (cut_y−top_shift)/h
  ```
- **像素角**：`left=(int)((xb−wb/2)·w)` 等四角，C int 截断。
- **mosaic_bound=True**：按象限界 clamp（left<lb 收至 lb、right>rb 收至
  rb、top/bot 同）后**由角重算** xb,wb,yb,hb（(right+left)/2/w 等）。
- **mosaic_bound=False**：越界侧收缩修正（**先于终门执行，可把越界框
  救回**）——`left<0: diff=left/w（负）, xb−=diff/2, wb+=diff`（收缩，
  等效把框左缘裁到 0）；right>w / bot>h 用 −diff；top<0 用 +diff；
  四向修正后角重算（再截断）。
- **终门（单 AND）**：left≥0 且 right≤w 且 top≥0 且 bot≤h 且
  0<wb<1 且 0<hb<1 且 0<xb<1 且 0<yb<1 且 wb>1/w 且 hb>1/h；
  通过写 `[xb,yb,wb,hb,id]`（id 透传），否则只记下标。
- **copy_plan**（memcpy 反推，当前象限）：
  ```
  0: dst (0,0)                src (w−cut_x−right_shift, h−cut_y−bot_shift)  size (cut_x, cut_y)
  1: dst (cut_x,0)            src (left_shift,          h−cut_y−bot_shift)  size (w−cut_x, cut_y)
  2: dst (0,cut_y)            src (w−cut_x−right_shift, top_shift)          size (cut_x, h−cut_y)
  3: dst (cut_x,cut_y)        src (left_shift,          top_shift)          size (w−cut_x, h−cut_y)
  ```
- 输出 {"boxes": [...], "dropped_indices": [...], "copy_plan": {...}}。
- 校验：w/h int ≥1；cut_x ∈[0,w]、cut_y ∈[0,h]；i_mixup ∈0..3；shifts
  为 4 个非负 int；truth [x,y,w,h,id] 有限且 w,h>0；mosaic_bound bool。
"""

import math


def solve(w, h, cut_x, cut_y, i_mixup, shifts, old_truths,
          mosaic_bound=False):
    raise NotImplementedError
