# 详解：抗锯齿 resize 与 trilinear 3D

## 出处

- 直接对标的算子：`torch.nn.functional.interpolate(x, size, mode="bilinear"|"bicubic"|"trilinear", antialias=..., align_corners=...)` 的 **CPU** 路径（PyTorch 官方文档 + 本仓库在 torch 2.14.1 CPU 上的实测）。`antialias` 官方只允许 4-D 的 bilinear/bicubic，本题把这条限制写进错误约定。
- **cubic 卷积核**：Keys, *Cubic Convolution Interpolation for Digital Image Processing*, IEEE Trans. Acoustics, Speech and Signal Processing 29(2), 1981（期刊，无 arXiv 版本）。同一族核有两个常用参数：\(a=-0.75\)（Keys 原文推荐的"通用插值"取值）与 \(a=-0.5\)（即 Catmull-Rom 核）。本题实测到原生**非 AA bicubic 用 \(-0.75\)、AA bicubic 用 \(-0.5\)**（在放大配置下两者输出最大差 \(0.277\)，量级远超任何容差，因此这个归属判定是可靠的）。starter 里把 \(-0.5\) 记作"PIL 口径"，属于库归属的转述；本仓库只验证**数值口径**，不去追认第三方库的实现细节。
- **抗锯齿缩小的动机**：Zhang et al., *Making Convolutional Networks Shift-Invariant Again*, ICML 2019, arXiv:1904.11486——下采样前先做低通（加宽支持域并按和归一化），否则缩小会引入混叠，破坏平移不变性。本题复刻的是这条思想在 PyTorch CPU kernel 里的具体离散化口径（tap 范围、\(0.5\) 偏移、归一化时机），不是它的连续域推导。
- 与 `upsample-bilinear`（同合集）：那题只做非 AA 的线性放大；本题把 2-tap/4-tap、AA、align_corners、3-D 四件事合成一个可分离 kernel。

## 完整推导与下标约定

**可分离性**：resize 的核对空间维是张量积的，即对输出 \((o_1,o_2)\)
\[\text{out}[o_1,o_2]=\sum_{y}\sum_{x} w_1[o_1,y]\,w_2[o_2,x]\,x[y,x],\]
\(w_d\) 由该维独立的 \((n_{in},n_{out},\text{align},\text{kernel},\text{AA})\) 决定。所以实现按维作用两次（参考实现构造 \((n_{out},n_{in})\) 稠密权重矩阵后做 matmul）与逐输出像素双循环（判题 oracle 的做法）严格等价；`check_channels_batch_independence` 进一步断言"整批 == 逐 \((b,c)\) 切片"。

**half-pixel 映射**：把像素看成宽度 1 的区间，源区间 \([0,n_{in}]\) 线性映射到目标 \([0,n_{out}]\)，目标像素 \(dst\) 的中心是 \((dst+0.5)\cdot\text{scale}\)，换算回"以源像素 0 中心为原点"的坐标就是
\[src=(dst+0.5)\cdot\text{scale}-0.5,\qquad \text{scale}=n_{in}/n_{out}.\]
`align_corners=True` 则把首末**像素中心**钉住：\(src=\text{scale}\cdot dst\)，\(\text{scale}=(n_{in}-1)/(n_{out}-1)\)。退化支路：\(n_{out}=1\) 时该比值分母为 0，原生取 \(\text{scale}=0\)，于是 \(src\equiv 0\)——整维塌到源索引 0（判题用 `torch.equal(out, x[..., 0:1, :])` 断言，实测原生同样如此）。

**非 AA 的 tap 与权重**：线性核 \(k_0=\lfloor src\rfloor\)、\(t=src-k_0\)（clamp 到 \([0,1]\)）、权重 \((1-t,t)\)；\(k_0=n_{in}-1\) 时第二个 tap 不越界而是复用最后一个索引。half-pixel 放大时首/末 \(src\) 为负（如 \(4\to8\)：\(src_0=0.5\cdot0.5-0.5=-0.25\)），**线性核把 \(src\) clamp 到 0**，故首行权重恰是 \([1.0,0.0]\)（复刻源像素 0）；**bicubic 不 clamp \(src\)**，而是 \(k_0=\lfloor -0.25\rfloor=-1\)、\(t=0.75\)，四个 tap \([-2,-1,0,1]\) 再各自 clamp 索引 → \([0,0,0,1]\)，权重 \([-0.03515625,\,0.26171875,\,0.87890625,\,-0.10546875]\)。合并同索引后有效权重是 \(w_0=1.10546875,\ w_1=-0.10546875\)（和仍为 1）——**这就是双三次的边界过冲（ringing）**，也正是"tap 索引 clamp 到 \([0,n_{in}-1]\)、同索引权重累加"这条规定的意义：若把同索引写成"覆盖"而不是"累加"，\(w_0\) 只剩最后一个 tap 的 0.87890625，权重和变成 0.7734，面积守恒与边界值同时崩掉。

**AA 的窗宽与归一化**：令 interp_size 为核的基础支撑（bilinear 2、bicubic 4）。
\[\text{scale}\ge 1\ (\text{缩小}):\ \text{support}=\tfrac{\text{interp\_size}}{2}\cdot\text{scale},\ \text{invscale}=1/\text{scale};\qquad
\text{scale}<1\ (\text{放大}):\ \text{support}=\tfrac{\text{interp\_size}}{2},\ \text{invscale}=1.\]
中心 \(\text{center}=\text{scale}(dst+0.5)\)（源空间中心 = \(c-0.5\)），tap 范围
\[t\in\big[\max(\lfloor c-s+0.5\rfloor,0),\ \min(\lfloor c+s+0.5\rfloor,n_{in})-1\big],\]
权重 \(\text{filter}((t-c+0.5)\cdot\text{invscale})\)，再整体除以权重和。实测 tap 数：bilinear \(12\to5\) 得 \([4,5,4,5,4]\)，bicubic \(13\to5\) 得 \([7,9,11,9,6]\)（上限 \(\lceil s\rceil\cdot2+1\) 分别 7 与 13，未被触顶）。归一化在 float32 下只做到精度级：实测权重和落在 \([0.9999999,\,1.0000001]\)，所以"面积保持"是 float32 精度级的性质，不是精确等式。

**为什么 scale<1 时 AA 对 bilinear 恰好等价于非 AA**：此时 \(\text{invscale}=1\)、\(\text{support}=1\)，tap 落在 \([\lfloor c-0.5\rfloor,\ \lfloor c+1.5\rfloor-1]\)，即至多 \(c-0.5\) 与 \(c+0.5\) 对应的两个源索引（越界时被夹回）；权重 \(1-|t-c+0.5|\) 恰好是线性插值的 \((1-t_\text{lin},t_\text{lin})\)。边界首行：非 AA 走 \(src<0\) clamp → \([1.0,0.0]\)，AA 走单 tap \([0]\) 归一化 → \([1.0]\)，**两条路给出同一个数**，这就是放大等价性成立的原因（实测 \(3\times4\to6\times8\)：AA tap 数为 \([1,2,2,2,2,2,2,2,1]\)）。bicubic 在放大时 support 仍是 2，但核的 \(a\) 从 \(-0.75\) 换成 \(-0.5\)，等价性破裂；`align_corners=True` 时 AA 仍用 half-pixel 中心而 scale 用 \((n_{in}-1)/(n_{out}-1)\)，与非 AA 的 \(src=\text{scale}\cdot dst\) 根本不是同一组相位，等价性也破裂（实测 \(6\to9\) 放大两支最大差 \(1.23\)）。

**float32 逐步舍入**：参考实现用 `struct.pack('<f', v)` 把每个中间量（scale、center、\(t\pm s+0.5\) 的 floor、每个 filter 值、归一化除法、累加和）圆整回 float32，复刻原生 CPU 的 `float` 计算链。实测对照：同一批配置下全程用 double 计算权重再降回来，与原生最大差 \(1.5\times10^{-6}\)（仍在判题 \(10^{-5}\) 容差内），而逐步 float32 舍入把它压到 \(2.4\times10^{-7}\)。也就是说**舍入链不是"能否通过"的分水岭**；真正的分水岭是 tap 集合与中心口径——tap 少取/多取一个必然把误差推到 \(10^{-1}\) 量级。
**累加在 float64**：权重已按输入精度定形，求和用 float64 再回投 float32，避免多 tap（bicubic AA 可达 11 个）相加时的次序性损失；这也是恒等尺寸能精确透传的前提之一（判题用 `torch.equal` 断言四组 mode/AA 组合，实测原生同尺寸也逐位返回原值——它对 \(n_{out}=n_{in}\) 直接短路）。

**trilinear**：同一套线性口径跑在 \((D,H,W)\) 三维上，\(n_{in}\to n_{out}\) 每维独立；`antialias=True` 与它组合直接 ValueError（原生同样拒绝）。

## 核心与易错点

1. **两套 Keys 参数**：非 AA bicubic \(a=-0.75\)、AA bicubic \(a=-0.5\)；写成一个常数就会在 bicubic 放大/缩小时整片偏掉。
2. **AA 路径无视 align_corners 的坐标口径**：center 永远是 \(\text{scale}(dst+0.5)\)，scale 却仍按 align_corners 取 \((n_{in}-1)/(n_{out}-1)\)。
3. **非 AA 的 clamp 只作用于线性核的 \(src\)**：bicubic 保持负 \(src\)，靠 clamp tap 索引处理边界。
4. **tap 索引 clamp 后必须合并同索引权重**： bicubic 边界行出现 \([0,0,0,1]\) 这种重复索引，漏合并等于丢权重。
5. **AA 的 \(0.5\) 偏移出现在三处**（\(\lfloor c\pm s+0.5\rfloor\)、filter 参数 \((t-c+0.5)\)），少加一个就整体平移半像素。
6. **\(\text{scale}<1\) 时不要加宽窗口**：support 固定为 interp_size/2、invscale=1；把它写成"永远乘 scale"会在放大时混进远距离 tap。
7. **恒等尺寸必须逐位透传**：\(n_{out}=n_{in}\) 时短路（或用单位阵），不要让浮点权重 \([1.0]\) 之外的东西参与。
8. **align_corners=True 且 \(n_{out}=1\) 的 scale 是 0**，不是 1 也不是 NaN；整维塌到索引 0。
9. **float32 输入要走 float32 计算链**：double 计算不会挂掉判题（容差吸收），但与原生 `torch.equal` 级别的对照、以及 tap 边界情形都会不一致。
10. **累加 dtype 与最终回投**：先 float64 累加、最后 cast 回输入 dtype；输出 dtype 必须等于输入 dtype（判题断言）。
11. **错误全部是 ValueError**：包括 AA+trilinear、秩不符、`size` 非法、非浮点 dtype——不要照抄原生的 RuntimeError 文案类型。

## 与邻近题的区别

- **vs `upsample-bilinear`**：那题只有非 AA、线性核、放大方向的坐标映射；本题的 AA 分支、bicubic 与 3-D 都是它的超集。
- **vs `upsample-nearest`**：最近邻用 \(\lfloor\cdot\rfloor\) 取单点、没有权重与归一化，也不存在"AA 等价性"这类问题。
- **vs `grid-sample`**：`grid_sample` 的源坐标由外部 \([-1,1]\) 网格给定（含非仿射映射），本题的坐标由尺寸比值生成；两者都涉及边界 clamp，但一个 clamp 到 0、一个按 padding_mode。
- **vs `adaptive-maxpool` / `global-pooling`**：那些把 \(o\) 给定后按 floor/ceil 划区间，聚合是 max/mean；本题聚合是加权求和且权重带负瓣。
- **vs `conv2d-naive` / `unfold-dilated-conv`**：卷积的权重是学习参数、与位置无关（平移等变）；本题的权重由几何决定、逐输出位置不同（因此 resize 本身**不是**平移不变的，AA 正是为补救缩小方向的失配而来）。
- **vs `soft-nms` / `topk-select` 等**：同为"逐位置算权重再合并"的形态，但没有坐标映射与核函数概念。

## 对拍口径与复杂度

- **主 oracle 是原生 `F.interpolate`**：float32 共 18 组配置（bilinear/bicubic × AA 两态 × align_corners 两态 × 放大/缩小/混合/恒等 + trilinear 4 组），float64 共 5 组；float32 用 \(10^{-5}\)、float64 用 \(10^{-12}\) 的 `allclose`。
- **独立 oracle 是逐输出像素的纯 Python 双精度循环**（`_oracle_bilinear_2d`）：它自己重算每维 tap/权重、用两重 for 循环做 \(w_y w_x x[y][x]\) 累加，与解法的"权重矩阵 + matmul"完全不同路径；覆盖 6 组 bilinear 配置（含 AA+align_corners=True 的 half-pixel center 特殊口径与 \(7\times5\to4\times6\) 的单维缩小/放大混合）。
- 判别性用例：恒等尺寸 4 组 mode/AA 组合 `torch.equal` 透传 + trilinear 一组；bilinear 放大 (6,8)/(7,9) 两支**逐位相等**且各自对齐原生；bicubic 放大两支必须不同（断言 `not allclose(atol=1e-4)`）；align_corners=True + \(n_{out}=1\) 的三维塌缩（`torch.equal(out, x[...,0:1,:])`，并断言原生同样）；整批与逐 \((b,c)\) 切片的独立性（bicubic AA 与 trilinear 各一组）；7 条 ValueError。
- 复杂度：可分离实现为 \(\sum_d O(n_{out,d}\cdot n_{in,d})\) 的单维权重构造，加上逐维矩阵乘 \(O(N C \prod_d n_{out,d}\cdot n_{in,d})\)；AA 时每行的 tap 数由 \(\lceil\text{support}\rceil\cdot2+1\) 界定（判题最大观测到 11）。参考实现把每维权重存成稠密 \((n_{out},n_{in})\) 矩阵，牺牲一点内存换实现简洁与 float64 累加。
