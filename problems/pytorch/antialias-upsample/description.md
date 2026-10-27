# 抗锯齿 resize 与 trilinear 3D

## 题目描述

`F.interpolate` 是视觉/视频管线里最常被"隐式"调用、也最容易被调错参数的算子：同一组 `size`，`align_corners` 与 `antialias` 各开各关，输出像素就完全不同。本题要求逐点复刻其 **CPU** 行为，把三件正交的事拼起来：

1. **坐标映射**：half-pixel（\(src=(dst+0.5)\cdot\text{scale}-0.5\)）还是 align_corners（\(src=\text{scale}\cdot dst\)，\(\text{scale}=\frac{n_{in}-1}{n_{out}-1}\)）；
2. **插值核**：线性 2-tap，或 Keys cubic 卷积核 4-tap（双三次）；
3. **抗锯齿**：缩小先把窗宽按 \(1/\text{invscale}\) 加宽，权重按和归一化（等价于先做低通再采样）。

三处"只有实测原生才知道"的口径是本题的真正考点：**非 AA 的 bicubic 用 Keys \(a=-0.75\)（OpenCV 口径），AA 却统一用 \(a=-0.5\)（PIL 口径）**；**AA 路径即使 `align_corners=True` 也仍用 half-pixel 中心**；**float32 输入的 scale/中心/权重逐步按 float32 舍入，只有累加在 float64 里做**。三者各自都足以让结果与原生对不上。

CPU、全程确定性、无随机数调用。

## 规则

- 完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。`mode` 取 `"bilinear" | "bicubic" | "trilinear"`；`size` 是长度 2（4-D 输入）或 3（5-D 输入）的正整数序列；`x` 只接受 float32/float64。
- **坐标映射**（每维独立）：`align_corners=False` 时 \(\text{scale}=n_{in}/n_{out}\)、\(src=(dst+0.5)\,\text{scale}-0.5\)，线性核下 \(src<0\) clamp 到 0，**bicubic 不 clamp**（负中心只改变权重相位，边界 tap 靠 clamp 索引处理）；`align_corners=True` 时 \(\text{scale}=(n_{in}-1)/(n_{out}-1)\)、\(src=\text{scale}\cdot dst\)、不 clamp，且 \(n_{out}=1\) 时 \(\text{scale}=0\)——整维塌到源索引 0（实测原生如此）。
- **非 AA 权重**：bilinear/trilinear 取 \(k_0=\lfloor src\rfloor\)、\(t=\text{clamp}(src-k_0,0,1)\)，2-tap \((1-t,\ t)\)（\(k_0=n_{in}-1\) 时两个 tap 同为最后一个）；bicubic 取 \(k_0-1..k_0+2\) 四个 tap，用 Keys cubic \(a=-0.75\)（\(|x|<1\): \(((a+2)x-(a+3))x^2+1\)；\(1\le|x|<2\): \(((ax-5a)x+8a)x-4a\)），tap 索引 clamp 到 \([0,n_{in}-1]\) 后**同索引权重相加**。
- **AA 权重**（实测原生 CPU 口径）：仅当 \(\text{scale}\ge 1\)（该维缩小）时加宽窗口，\(\text{support}=\frac{\text{interp\_size}}{2}\cdot\text{scale}\)、\(\text{invscale}=1/\text{scale}\)（interp_size：bilinear 2、bicubic 4）；\(\text{scale}<1\) 时 \(\text{support}=\frac{\text{interp\_size}}{2}\)、\(\text{invscale}=1\)。中心取 \(\text{center}=\text{scale}\,(dst+0.5)\)（**`align_corners=True` 下也是这个 half-pixel 式**），tap 范围 \([\max(\lfloor c-s+0.5\rfloor,0),\ \min(\lfloor c+s+0.5\rfloor,n_{in})-1]\)，权重 \(=\text{filter}((tap-c+0.5)\cdot\text{invscale})\)，其中 bicubic 用 **\(a=-0.5\)**，最后**按权重和归一化**（和为 0 的退化情形输出记 0）。
- **trilinear** 就是线性核跑在三个空间维上，且 `antialias=True` 与 trilinear 组合抛 ValueError。
- **恒等尺寸**：某维 \(n_{out}=n_{in}\) 时该维逐位透传，整条 resize 的输入输出在该维逐位相等（AA 两态皆然）。
- **数值口径**：float32 输入时 scale/中心/权重逐步按 float32 舍入（与原生 separable/AA 路径一致），累加在 float64 完成，最后回投 float32；float64 输入全程 float64。
- **错误约定**：非法 mode、非法秩（bilinear/bicubic 要 4-D、trilinear 要 5-D）、`size` 长度不符或含非正整数、非浮点 dtype、AA+trilinear **一律抛 ValueError**。

## 示例

\(x\) 为 (1,1,3,4)、值 \(1..12\)，`size=(2,2)`、`mode="bilinear"`（默认 `antialias=False, align_corners=False`）：

\[x=\begin{bmatrix}1&2&3&4\\5&6&7&8\\9&10&11&12\end{bmatrix}
\;\longrightarrow\;
\mathrm{out}=\begin{bmatrix}2.5&4.5\\8.5&10.5\end{bmatrix}\]

逐步：\(scale_y=3/2\)、\(scale_x=4/2=2\)。
- \(dst_y=0\): \(src=0.5\cdot1.5-0.5=0.25\) → 行 0/1 权重 \((0.75,0.25)\)；\(dst_y=1\): \(src=1.75\) → 行 1/2 权重 \((0.25,0.75)\)。
- \(dst_x=0\): \(src=0.5\cdot2-0.5=0.5\) → 列 0/1 权重 \((0.5,0.5)\)；\(dst_x=1\): \(src=2.5\) → 列 2/3 权重 \((0.5,0.5)\)。
- \(\mathrm{out}[0][0]=0.75\cdot\frac{1+2}{2}+0.25\cdot\frac{5+6}{2}=0.75\cdot1.5+0.25\cdot5.5=2.5\)；\(\mathrm{out}[1][1]=0.25\cdot7.5+0.75\cdot11.5=10.5\)。

所有权重（0.5/0.25/0.75）都是二进制精确值，所以手算与 float32 输出、与原生**逐位一致**。

对照口径差异（用判题的确定性花纹输入实测）：4×4 输入放大到 (8,8) 时，`mode="bicubic"` 的 `antialias=False` 与 `antialias=True` 输出最大差 \(0.277\)（\(a=-0.75\) vs \(a=-0.5\)）；同一尺寸的 `mode="bilinear"` 放大到 (6,8) 时两者**数学上完全等价**（同 tap 同权重），原生两条 CPU 路径只差 float32 舍入（实测最大 \(4.8\times10^{-7}\)），本题参考实现则让它逐位相等。另外 \(6\times6\to9\times9\) 的 `align_corners=True` 放大里，AA 与非 AA 最大差 \(1.23\)——因为 AA 分支仍用 half-pixel 中心，压根没用 \(\text{scale}\cdot dst\)。

## 约束

- 返回与 `x` 同 dtype、形状 \((N,C,*size)\) 的新张量；不修改入参。
- 判题以原生 `F.interpolate` 为主 oracle：float32 用 \(10^{-5}\)（`atol`/`rtol`），float64 用 \(10^{-12}\)；恒等尺寸与 align_corners + \(n_{out}=1\) 两支用 `torch.equal` 断言逐位一致。
- 实现必须是**可分离**的（每维独立算 tap 与权重，再逐维作用），且不使用随机数；batch 与 channel 维只是旁观者。
