# SPP-Net 金字塔池化账

## 题面

SPP-Net 把任意 H×W 特征图池化成固定长度：级联 [1,2,4,...] 级空间金字塔。
官方 caffe 实现不直接算 bin 边界，而是把每级翻译成 max-pooling 的
kernel/stride/pad——这带来与论文伪代码不同的窗口划分：窗口可能重叠，
甚至可能出现空 bin。给定特征图与每级 bin 数，请复现官方账。

给定：
- `feature`：H×W 二维浮点 list
- `bin_counts`：每边 bin 数列表（如 [1,2,4]，按给定顺序逐级处理）

## 每级账（bin 数 k）

1. `kernel = ceil(H/k)`（H、W 各自算，可不同）；`stride = kernel`；
   `pad = (kernel·k − H + 1) // 2`（remainder 对称分摊）。
2. bin `(ph, pw)` 窗口：
   `hs = max(0, ph·kernel − pad)`（起点可负，clip 0）；
   `he = min(hs + kernel, H)`（终点用**未 clip** 的 hs+kernel 再 clip H）。
   W 轴同理（用 kernel_w/pad_w）。
3. 窗口空（hs≥he 或 ws≥we）时该 bin 输出 −inf（官方 −FLT_MAX）。
4. 窗口内取 max，按级按 (ph,pw) 行主序 flatten 拼接。

## 输出

`{"pooled": [...], "shapes": [[k,k]...], "kernels": [[kh,kw]...],
"pads": [[padh,padw]...]}`。

## 输入约束

- feature 规则非空、元素有限；bin_counts 非空、元素为 int ≥1；H、W ≥
  max(bin_counts)；违规抛 `ValueError`。

## 样例

5×5 + [1,2,4] → 21 维：k=1 级 kernel=5,pad=0，全局 max=24；k=2 级
kernel=3、pad=1，bin(0,0) 窗口 (0..3)×(0..3) → 12；k=4 级 kernel=2、
pad=2，bin(0,0) 窗口 (0..2)×(0..2) → 6。
