# 详解：LpPool 与 maxpool 索引/unpool

## 出处

- 直接对标的算子：`torch.nn.functional.lp_pool2d` / `lp_pool1d`（`nn.LPPool2d` 的函数式）、`F.max_pool2d(..., return_indices=True)`、`F.max_unpool2d`（PyTorch 官方文档 + 本仓库在 torch 2.14.1 CPU 上的实测）。
- **"p 范数池化"与"最大值池化"的关系**：\(p\to\infty\) 时 \(\big(\sum|x_i|^p\big)^{1/p}\) 收敛到 \(\max|x_i|\)，所以 LpPool 常被当作 max-pool 的平滑版本（梯度会分给窗口内所有元素，而不是只走 argmax 一条）。本题把 \(p\) 限定为有限值 \(\ge 1\)，\(\infty\) 情形交给 `maxpool_idx` 模式。
- **"把最大值写回源位置"** 是 Deconvnet 的 unpooling（Zeiler & Fergus, *Adaptive Deconvolutional Networks for Mid and High Level Feature Engineering*, ICCV 2011, arXiv:1311.2901）：记录 max 的位置（`return_indices`）、再按位置稀疏还原，正是本题 `maxpool_idx` + `maxunpool` 两态的合成。
- **与原生不一致的两处，题面都改成了本仓库口径**（都经实测确认，见下节）：其一，LpPool 对每个元素取 \(|x|\)（原生 `p` 为奇整数或分数时对负输入给带符号值或 NaN）；其二，非法入参统一抛 **ValueError**（原生 `padding > kernel/2` 抛 RuntimeError、`F.max_unpool2d` 的越界 index 也抛 RuntimeError）。
- `F.lp_pool2d` **没有 padding 参数**，所以带填充的 LpPool 分支只能与独立 oracle 对拍；`F.max_pool2d` 有 padding，可以对拍原生（含 indices）。

## 完整推导与下标约定

**窗口几何**：一维上 \(o=\lfloor (n+2p_{\text{pad}}-k)/s\rfloor+1\)（二维逐维独立）。窗口 \(i\) 覆盖填充坐标 \([is,\ is+k-1]\)，映射回原图坐标为 \(r = is + a - p_{\text{pad}}\)（\(a\in[0,k)\)）。要求 \(o\ge 1\) 与 \(2p_{\text{pad}}\le k\)：后者保证**每个窗口至少含一个真实元素**（窗口跨度 \(k\)、左右各填 \(p_{\text{pad}}\le k/2\)，故中心半段必落在原图内）。这条约束对 `maxpool_idx` 至关重要——若允许一个窗口全是填充，\(-\infty\) 就会成为输出，indices 也失去定义。

**indices 的坐标链**（这是本题最容易写错的一行）：窗口 \((i,j)\) 内把 \(k_h\times k_w\) 展平成行优先局部序号 \(l = a\,k_w + b\)，`argmax` 取首个最大值（CPU 上 `torch.argmax` 并列取最小下标，与原生 `return_indices` 一致）；还原成原图坐标
\[\text{row} = i\,s_h + \lfloor l/k_w\rfloor - p_{\text{pad},h},\qquad
\text{col} = j\,s_w + (l \bmod k_w) - p_{\text{pad},w},\qquad
\text{idx} = \text{row}\cdot W + \text{col}.\]
两处细节：**减 padding 偏移**（否则 indices 落在填充图上，padding>0 时全体偏右偏下）；**乘以原图宽度 \(W\)** 而不是填充后的 \(W+2p_w\)。参考实现把这两步写成 `(rows - ph) * w + (cols - pw)`，实测在 \((k,s,\text{pad})=((2,2),(2,2),(0,0)),((3,3),(1,1),(1,1)),((3,3),(2,1),(1,1)),((2,3),(1,2),(1,0))\) 四组配置上与 `F.max_pool2d(..., return_indices=True)` **`torch.equal` 逐位一致**。

**\(|x|\) 口径与原生分歧的确切范围**：设窗口内有负值。偶数 \(p\) 时 \(|x|^p = x^p\)，两支重合（实测 \(p=2,4,6\) 与原生全等）；奇数整数 \(p\) 与分数 \(p\) 时分歧可观察——窗口 \([-1,-2,3,4]\)：
- \(p=1\)：本题 \(1{+}2{+}3{+}4=10.0\)，原生 `F.lp_pool2d(x,1,2,2)` 给 **4.0**（带符号和）；
- \(p=3\)：本题 \((1+8+27+64)^{1/3}=100^{1/3}=4.6415887\)，原生给 \(82^{1/3}=4.3444815\)；
- \(p=1.5\)：本题 6.6178598，原生给 **NaN**（负数的分数次幂）。
反过来，输入非负时 \(p=1\)、分数 \(p\) 也与原生一致（判题就用这组对照把"abs 口径"与"原生口径"的边界划清）。

**p=1 不是均值**：`avg_pool2d(|x|, k)` 给 \(\frac{1}{k^2}\sum|x_i|\)（判题用探针断言本题输出恰为其 \(k^2\) 倍）；`kernel_size=1` 时窗口只有一个元素，输出 \(|x_i|\) 逐元素——注意这是**数学等价**：float32 下 \(p=2\) 恰好逐位还原（实测 `torch.equal` 为真），\(p=3\) 有 \(1.2\times10^{-7}\) 的 \(\text{pow}(p)\to\text{pow}(1/p)\) 往返误差，所以判题用 `allclose` 而不是 `equal`。

**零填充与全零窗口**：LpPool 补 0，且 \((0)^{1/p}=0\)，全零窗口（含纯填充边角）输出 0——不会出现 \(\sum=0\) 后开方变 NaN（\(p\ge1\) 时 \(0^{1/p}=0\) 良定义）。这与 maxpool 补 \(-\infty\) 形成对照，也是两种 mode 必须分开填充的原因。

**maxunpool 的散射与单射条件**：实现是 `zeros(N,C,H·W).scatter_(2, indices, x).view(N,C,H,W)`，即"覆盖写 + 其余为 0"。`stride ≥ kernel` 保证窗口不重叠，从而 indices 在不同窗口间互不相同（单射）；一旦重叠，同一个源位置可能被两个窗口命中，原生 `F.max_unpool2d` 对此**没有定义**取哪个值，题面直接抛 ValueError 而不是猜一个。`output_size=None` 时按 \(H=(h-1)s_h+k_h\) 推断，这是"无残边"情形的逆变换；有残边时（如 3×3 输入、k=s=2 → 1×1 输出）反推得 2×2，而真实原图是 3×3，indices=4 会越界——判题示例正是这条，必须显式传 `output_size=(3,3)`。

**indices 的合法域是输出画布，不是小图**：校验用 \(0\le\text{idx}<H\cdot W\)（\(H,W\) 为 unpool 输出尺寸）。原生在同情形抛 `RuntimeError: Found an invalid max index: 4 (output volumes are of size 2x2)`，本题按 ValueError 抛出（消息含同样的区间提示）。

**秩与参数校验顺序**：`lppool1d` 要 3-D、其余 4-D；`lppool1d` 还额外要求 kernel/stride/padding 逐维一致（传二元组不等即抛错），因为 1-D 只有一个空间维，二元组只能是它的重复。`maxpool_idx`/`maxunpool` 忽略 `p`（判题断言传 `p=123.0` 结果不变），非法 `p` 也就不在那里检查。

## 核心与易错点

1. **每个元素取 \(|x|\)**：负输入 + 奇/分数 \(p\) 是与原生的**口径差异**，不是 bug；判题用非负输入对拍原生、用负输入对拍 oracle 与探针。
2. **`p=1` 是 \(|x|\) 求和**，不是均值：多除一个 \(k^2\)（写成 `avg_pool2d(|x|)`）是最常见错误。
3. **maxpool 填充补 \(-\infty\)**（本题沿用 `maxpool2d` 题的口径）：补 0 会让全负窗口输出 0。
4. **indices 要减 padding 偏移**：`(row - ph) * W + (col - pw)`，写成 `row*W + col` 在 padding>0 时全体错位。
5. **indices 的 stride 是原图宽 \(W\)**，不是填充后的宽度。
6. **并列最大取行优先最前者**：`argmax` 展平后取首个，与原生一致；用 `amax` + `==` 反查会拿到别的坐标。
7. **unpool 未命中位置是 0**，不是原值、不是 \(-\infty\)、也不是窗口内其它元素。
8. **unpool 的画布尺寸**：`output_size=None` 的推断只在无残边时还原原图；判题示例（3×3→1×1）必须显式传 \((3,3)\)。
9. **重叠配置禁止**（`stride < kernel` 抛 ValueError）：原生未定义写回顺序，别试图用 `scatter_reduce(amax)` 去"猜"原生行为。
10. **错误类型统一 ValueError**：不要照抄原生的 RuntimeError/IndexError；秩、mode、\(p\)、padding、indices、output_size 全部 ValueError。
11. **不修改入参**：`unfold` 返回的是视图，必须在 pad 之后的新张量上算，写 `out[p]=...` 前先 `clone`（参考实现全程只读）。

## 与邻近题的区别

- **vs `maxpool2d`**：那题只返回最大值（padding 补 \(-\infty\) 的口径在那里建立）；本题在它之上加 `return_indices` 的坐标链和 unpool 的逆过程。
- **vs `avgpool2d`**：均值口径 + `count_include_pad` 的分母游戏是它的难点；本题 \(p=1\) 刻意**不**除 \(k^2\)，判题有专门的倍数探针。
- **vs `adaptive-maxpool` / `global-pooling`**：那两题的窗口边界由输出尺寸反推（\(o\) 给定、区间用 floor/ceil 划分）；本题是经典固定 \(k/s/\text{pad}\) 的前向几何。
- **vs `roi_pool`**：ROI 池化的窗口是浮坐标、按双线性/取整落在任意区域；本题窗口始终在整数网格上。
- **vs `unfold-fold-roll`**：那题给出一般 im2col（含 dilation）与 `fold` 的**重叠累加**；本题的 `unfold` 只是内部手段，`maxunpool` 是单射写回（覆盖）而不是累加，重叠情形被题面禁止。
- **vs `scatter-reduce-aggregate`（同批次）**：unpool 恰是其 `reduce` 为"覆盖写"、`include_self=False`、目标位置由 indices 给定的一维化特例；那里的重复坐标聚合正是本题刻意排除的情形。
- **vs `conv2d-transposed`**：转置卷积也用"写回/累加"的对偶视角，但每个窗口值是加权求和而非 argmax 的搬运。

## 对拍口径与复杂度

- **主 oracle 是原生**：`F.lp_pool2d` / `F.lp_pool1d`（\(p=2\) 与偶数 \(p\)，含负输入）、非负输入下的 \(p=1\) 与分数 \(p\)、`F.max_pool2d(..., return_indices=True)`（out 用 \(10^{-6}\)，indices 用 `torch.equal`）、`F.max_unpool2d`（`torch.equal`，逐位一致）、float64 组用 \(10^{-12}\)。
- **独立 oracle 是多重 Python 循环**（`_oracle_lppool2d`/`_oracle_lppool1d`/`_oracle_maxpool`/`_oracle_unpool`）：逐 \((n,c,i,j)\) 遍历窗口、按坐标边界判断取原值或 \(-\infty\)、用纯 Python `**` 求幂累加。它覆盖原生没有的能力（LpPool 的 padding、负输入的奇/分数 \(p\)），也覆盖 indices 的负坐标边角。
- 判别性用例：示例三态（\(\sqrt{46}\) / 5.0 / indices 4 / unpool 只有一个非零位）、\(p=1\) 的"求和不是均值"与 `avg_pool2d(|x|)` 的 \(k^2\) 倍关系、kernel=1 的逐元素 \(|x|\)、全零窗口与零填充边角、并列最大（把输入按 \(1/3\) 舍入制造重复值，实测 `torch.argmax` 取最小下标、原生 `return_indices` 同口径）、`maxpool_idx` 忽略 `p`、`stride=None → kernel`、roundtrip 的非零位集合与 indices 完全一致（`scatter` 构造期望掩码）、21 条 ValueError。
- 复杂度：\(O(N\cdot C\cdot o_h\cdot o_w\cdot k_h k_w)\) 时间；参考实现用两次 `unfold` 得到 \((N,C,o_h,o_w,k_h,k_w)\) 的视图后向量化归约，内存是窗口张量的展开规模（判题规模 \(H,W\le 8\)、\(L\le 11\)、\(k\le 4\)，可忽略）。maxunpool 是 \(O(N C h w)\) 的散射。
