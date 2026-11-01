# YOLOv4 Mosaic 四象限拼接 truth 账

## 题面

Mosaic 增强把四张图各裁互补象限拼成一张训练图，truth 侧要做的不只是
"平移"——象限方向不对称、像素角要 int 截断、越界框有两种修正分支
（bound 时按象限 clamp 重算；非 bound 时收缩修正且**先于终门执行**，
能把越界框救回），最后还要过 14 条件的大 AND 终门。给定拼接参数与一张
图的 box 列表，请复现官方 `blend_truth_mosaic` 的完整账。

给定：
- `w, h`：net 输入尺寸（int ≥1）
- `cut_x, cut_y`：拼接中心（0≤cut_x≤w、0≤cut_y≤h）
- `i_mixup`：当前图象限 0..3（左上/右上/左下/右下）
- `shifts = [left_shift, right_shift, top_shift, bot_shift]`：非负 int
- `old_truths`：`[x, y, w, h, id]` 归一化列表
- `mosaic_bound`：bool

## 处理链（对每条 box 依序）

1. 象限平移（四分支方向不对称）：
   `0: xb−=(w−cut_x−right_shift)/w, yb−=(h−cut_y−bot_shift)/h`；
   `1: xb+=(cut_x−left_shift)/w, yb−=(h−cut_y−bot_shift)/h`；
   `2: xb−=(w−cut_x−right_shift)/w, yb+=(cut_y−top_shift)/h`；
   `3: xb+=(cut_x−left_shift)/w, yb+=(cut_y−top_shift)/h`。
2. 像素角：`left=(int)((xb−wb/2)·w)` 等四角。
3. 越界修正（两分支互斥）：
   - mosaic_bound=True：按象限界单侧 clamp（left<lb→lb、right>rb→rb、
     top/bot 同）后由角重算 `xb=(right+left)/2/w`、`wb=(right−left)/w` 等；
   - mosaic_bound=False：`left<0: diff=left/w, xb−=diff/2, wb+=diff`
     （diff<0 即收缩）；`right>w: diff=(right−w)/w, xb−=diff/2, wb−=diff`；
     top/bot 同理；四向修正后角重算。
4. 终门（单 AND）：`left≥0 且 right≤w 且 top≥0 且 bot≤h 且 0<wb<1 且
   0<hb<1 且 0<xb<1 且 0<yb<1 且 wb>1/w 且 hb>1/h`。
5. 通过 → `boxes.append([xb, yb, wb, hb, id])`（id 透传）；否则记下标。

## copy_plan（memcpy 反推，当前象限）

```
0: dst (0,0)         src (w−cut_x−right_shift, h−cut_y−bot_shift)  size (cut_x, cut_y)
1: dst (cut_x,0)     src (left_shift,          h−cut_y−bot_shift)  size (w−cut_x, cut_y)
2: dst (0,cut_y)     src (w−cut_x−right_shift, top_shift)          size (cut_x, h−cut_y)
3: dst (cut_x,cut_y) src (left_shift,          top_shift)          size (w−cut_x, h−cut_y)
```

## 输出

`{"boxes": [...], "dropped_indices": [...], "copy_plan": {dst, src, size}}`。

## 输入约束

- w/h int ≥1；cut 在界内；i_mixup∈0..3；shifts 4 个非负 int；truth
  `[x,y,w,h,id]` 有限且 w,h>0；mosaic_bound bool；违规抛 `ValueError`。

## 样例

`w=h=416, cut=(208,208), i_mixup=1, shifts=0`，truth `[0.3,0.6,0.1,0.1,7]`
→ 平移 (xb,yb)=(0.8,0.1)，原样过门，输出 box `[0.8, 0.1, 0.1, 0.1, 7]`；
copy_plan `dst=[208,0], src=[0,208], size=[208,208]`。
