# SPP-Net 金字塔池化账

SPP-Net（1406.4729）把任意尺寸特征图池化成固定长度（1+4+16=21）。官方
实现不在 SPP 层内直接算 bin 边界，而是把每级翻译成一个 max-pooling 层的
`kernel / stride / pad`，bin 边界由 pooling 层的通用公式给出。本题转写
BVLC/caffe `spp_layer.cpp` 的 `GetPoolingParam` 与 `pooling_layer.cpp` 的
`MaxPoolForward`——**caffe 式与论文伪代码的窗口划分不同**，这是本题核心。

## solve 接口

```python
solve(feature, bin_counts)
```

- `feature`：H×W 二维 list（float）；`bin_counts`：升序不必、逐级处理的
  每边 bin 数（如 [1,2,4]）。
- 约束：feature 规则非空、元素有限；H、W ≥ max(bin_counts)（入口校验；
  注意这只保证不退化，**不保证无空 bin**，见下）。

## GetPoolingParam（每级 k）

```
kernel = ceil(H / k)          # H、W 各自算，矩形图 kh≠kw
pad    = (kernel·k − H + 1) // 2   # remainder 对称分摊（ceil 整数式）
stride = kernel
```

## MaxPoolForward（bin (ph, pw) 的窗口）

```
hs = max(0, ph·kernel − pad)   # 起点 pad 前移，可负 → clip 0
he = min(hs + kernel, H)       # 终点用未 clip 的 hs+kernel 再 clip H
```

- **重叠**：pad>0 时 bin(0,·) 与 bin(1,·) 窗口可重叠——论文式
  `floor(i·H/k)` / `ceil((i+1)·H/k)` 无重叠。H=5,k=4 论文式 bin1=(1,3)，
  caffe 式 bin(1,·) 起点=max(0,2−2)=0 与 bin(0,·) 同窗。
- **空 bin**：H=6,k=5 → kernel=2, pad=2，bin(4,·) 起点 6 ≥ H → 窗口空，
  官方输出 `−FLT_MAX`（本题 −inf）。约束 H≥k **不能**排除这种情况。
- 窗口 max。

## 输出

`{"pooled": 按级按 (ph,pw) 行主序拼接的 flatten, "shapes": [[k,k]...],
"kernels": [[kh,kw]...], "pads": [[padh,padw]...]}`。

## 判题细节

- 5×5 + [1,2,4] 是论文经典例：全局 21 维，k=2 级 kernel=3、pad=1。
- oracle 逐 bin 显式枚举 `max(..., default=−inf)` 兜空窗；200 组随机
  （1–12 边长、1–3 级）对拍；另设论文式窗口对照断言两者可不同。

## 划界

- `adaptive-maxpool`/`adaptive-avgpool`：单级自适应池化；本题是**多级
  金字塔 + caffe 式 kernel/pad 划分账**（重叠窗口与空 bin quirk 只在
  SPP 组合下出现）。
- AVE / STOCHASTIC pooling 不入题；SPP 后接的 FC 拼接只考 flatten 顺序。

## 变异记录

12 条非等价全抓：ceil 改 floor、pad 改 (r)//2、起点/终点不 clip、拼接
倒序、max 改 min、论文式窗口、终点用 clip 前 hs、H/W kernel 互换、空 bin
输出 0、pad 全给上侧、pad 乘进 ph、窗口 off-by-one。1 条**等价重构**
（`min(hs+kernel,H)` 改写为 `min(max(0,...)+kernel, H)`）记 notes。
