# YOLOv4 Mosaic 四象限拼接 truth 账

Mosaic 把四张图按拼接中心 (cut_x, cut_y) 各裁互补象限拼进一张 net 尺寸
画布。官方实现分两层：图像侧是 `load_data_detection` 里四分支 memcpy
（把当前图的对应子块拷到目标象限），truth 侧是 `blend_truth_mosaic`——
把每张图的 box 经象限平移、像素角截断、越界修正后过终门写入。本题转写
AlexeyAB/darknet `src/data.c` 两条链；随机量（cut/shift 的生成）全部显式
入参，只考坐标账不读图。

## solve 接口

```python
solve(w, h, cut_x, cut_y, i_mixup, shifts, old_truths, mosaic_bound=False)
```

- `w, h`：net 输入尺寸（lowest 门也用它：`lowest_w = 1/w`）。
- `i_mixup ∈ {0,1,2,3}`：当前图所在象限（0 左上 / 1 右上 / 2 左下 / 3 右下）。
- `shifts = [left_shift, right_shift, top_shift, bot_shift]`：官方由
  `min(cut, max(0, −pleft·w/ow))` 生成的裁剪补偿，显式入参。
- `old_truths`：该图经 fill_truth_detection 后的归一化 `[x, y, w, h, id]`。

## 1. 象限平移（四分支方向不对称）

```
0: xb −= (w−cut_x−right_shift)/w ; yb −= (h−cut_y−bot_shift)/h
1: xb += (cut_x−left_shift)/w    ; yb −= (h−cut_y−bot_shift)/h
2: xb −= (w−cut_x−right_shift)/w ; yb += (cut_y−top_shift)/h
3: xb += (cut_x−left_shift)/w    ; yb += (cut_y−top_shift)/h
```

wb/hb 平移不变。

## 2. 像素角

`left=(int)((xb−wb/2)·w)` 等四角（C int 截断）。

## 3. 越界修正（两分支互斥）

- **mosaic_bound=True**：按象限界 clamp（0:(0,cut_x,0,cut_y)、
  1:(cut_x,w,0,cut_y)、2:(0,cut_x,cut_y,h)、3:(cut_x,w,cut_y,h)）——
  left<lb 收至 lb、right>rb 收至 rb（单侧条件），然后**由角重算**
  `xb=(right+left)/2/w, wb=(right−left)/w` 等。
- **mosaic_bound=False**：四向收缩修正，**先于终门执行、可把越界框救回**：
  ```
  left<0:  diff=left/w（负）; xb −= diff/2; wb += diff   # 收缩
  right>w: diff=(right−w)/w;  xb −= diff/2; wb −= diff
  top<0 / bot>h 同理作用于 yb/hb
  ```
  全部修正后四角重算（再截断）。中心恰 (0,0)/(1,0) 的框会被修正抬进
  界内而非丢弃——这是本题最容易手算错的点。

## 4. 终门（单 AND，14 条件）

`left≥0 && right≤w && top≥0 && bot≤h && 0<wb<1 && 0<hb<1 && 0<xb<1 &&
0<yb<1 && wb>1/w && hb>1/h` —— 全过才写 `[xb, yb, wb, hb, id]`
（id 透传），否则该 box 丢弃（只记下标）。

## 5. copy_plan（memcpy 反推，当前象限）

```
0: dst (0,0)         src (w−cut_x−right_shift, h−cut_y−bot_shift)  size (cut_x, cut_y)
1: dst (cut_x,0)     src (left_shift,          h−cut_y−bot_shift)  size (w−cut_x, cut_y)
2: dst (0,cut_y)     src (w−cut_x−right_shift, top_shift)          size (cut_x, h−cut_y)
3: dst (cut_x,cut_y) src (left_shift,          top_shift)          size (w−cut_x, h−cut_y)
```

（图 0 贡献其右下子块、图 1 左下子块、图 2 右上子块、图 3 左上子块。）

## 输出

`{"boxes": [[x,y,w,h,id]...], "dropped_indices": [...], "copy_plan": {...}}`。

## 判题细节

- wb==1（满宽框）被 `wb<1` 门拦——`<=` 变异的抓点（truth x=0、wb=1、
  象限 1 → xb=0.5 满框全在界内）。
- oracle 用先合并 dx/dy 再逐框直算的组织，250 组随机（shifts 与象限
  对齐、mosaic_bound 对半）对拍。

## 划界

- `vision-yolov1-grid-responsibility`：v1 的 constrain 是「截断后重算」，
  本题 bound 分支同精神但按象限界；非 bound 的收缩修正与半开终门是
  mosaic 专属。
- mixup=1（两图 0.5 混合 blend_truth）与 cutmix 不入题；图像像素拷贝
  不读图，只考 copy_plan 坐标。

## 变异记录

13 条全抓：象限平移符号、shift 用错侧、bound clamp 方向、bound 后不
重算、收缩改扩张、top 修正符号、终门 `wb<=1`（靠 wb==1 锚点抓）、
lowest 减半、像素角 round、重算中心 (r−l)/2、id 不透传、copy_plan
shift 用错、平移不除 w。
