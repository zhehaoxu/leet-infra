# 详解：检测：FPN 自顶向下融合与层尺寸账

## 出处

- 算法：Lin et al., *Feature Pyramid Networks for Object Detection*, CVPR 2017, arXiv:1612.03144。第 3 节给出 top-down pathway：每级 1×1 侧接把通道统一到 256，上一级结果 2× 最近邻放大后与侧接逐元素相加，再过一个 3×3 卷积消除上采样的混叠。
- 本题**略去**最后的 3×3 输出卷积：它不改变任何尺寸、步长或 anchor 数，也不进入本题的参数量账本；保留它只会额外引入一组权重。本题的侧接参数量因此只算 1×1 卷积（\( \text{fpn\_dim}\cdot C_i \) 再加偏置）。
- P6 的构造在真实实现里并不统一：RetinaNet（arXiv:1708.02002）在 P5 上接 3×3 stride-2 卷积得到 P6；Detectron2 的 `LastLevelMaxPool` 用 1×1 核、stride 2 的最大池化。本题固定为 `kernel_size=2, stride=2, padding=0` 的最大池化，是一个自洽、可手算的第三种口径——数值上与上面两者都不同，判题以本题约定为准。
- 放大下标与 `torch.nn.functional.interpolate(..., size=(H_t,W_t), mode="nearest")` 的规则一致（`scale = S_up / S_t`，向下取整）；`mode="nearest-exact"` 是另一套取整规则，实测 3×3→5×5 时源下标为 \( [0,0,1,2,2] \)，与本题的 \( [0,0,1,1,2] \) 不同。
- 步长默认 \(2^{2+i}\)（4/8/16/32）是从 C2 起算的约定；若特征列表从 C3 开始，必须显式传 `[8, 16, ...]`——本题不让实现从 \(H_i\) 反推。
- CPU、确定性、无随机数；1×1 卷积用 `F.conv2d`（或等价的通道矩阵乘），放大用 `index_select` 的整数下标。

## 完整推导与下标约定

**侧接（1×1 卷积）**：对第 \(i\) 级特征 \( F\in\mathbb{R}^{N\times C_i\times H_i\times W_i} \) 与权重 \( W_i\in\mathbb{R}^{\text{fpn}\times C_i} \)、偏置 \( b_i\in\mathbb{R}^{\text{fpn}} \)（可缺省），

\[
\text{lateral}_i[n,o,y,x]=\sum_{c=0}^{C_i-1}W_i[o,c]\,F[n,c,y,x]+b_i[o],
\]

即逐像素对通道做矩阵乘 \( W_i\,F[n,:,y,x] \)。参数量 \( \text{fpn}\cdot C_i+(\text{fpn}\ \text{若有偏置}) \)。把二维权重补成 \( (\text{fpn},C_i,1,1) \) 的核直接调 `F.conv2d`，与手写矩阵乘等价。

**最近邻放大的整数规则**：设上一级（更粗）尺寸为 \( (H_u,W_u) \)，本级目标为 \( (H_t,W_t) \)，则

\[
\text{src}(d)=\left\lfloor \frac{d\cdot H_u}{H_t}\right\rfloor,\qquad d=0,\dots,H_t-1 .
\]

两条性质让它不需要夹取：\( d\le H_t-1 \Rightarrow d\cdot H_u/H_t<H_u \)，所以 \( \text{src}<H_u \)；\( H_u\ge1 \) 时 \( \text{src}\ge0 \)。整数倍 \( H_t=2H_u \) 退化为 \( \lfloor d/2\rfloor \)，即每行复制两次；倍率不整时（\( H_u=3,H_t=5 \)）得到 \( [0,0,1,1,2] \)，中间的源行被重复两次、最后一行重复一次，不做任何插值。实测 `F.interpolate(size=(5,5), mode="nearest")` 正是这个映射；换成浮点 `scale = H_u/H_t` 再四舍五入会得到 \( [0,1,1,2,2] \)，`nearest-exact` 则是 \( [0,0,1,2,2] \)，两者都与本题不同。

**相加的对齐**：目标尺寸取自本级侧接输出 \( (H_i,W_i) \)，所以放大后的张量与 `lateral[i]` 形状天然一致，不需要裁剪或广播。若实现先相加再放大（或者把上一级直接加到本级而不放大），形状根本对不上。

**P6**：对融合后的 P5 做

\[
\text{P6}[y,x]=\max_{dy,dx\in\{0,1\}}\text{P5}[2y+dy,\,2x+dx],\qquad
y< \lfloor H_5/2\rfloor,\ x<\lfloor W_5/2\rfloor .
\]

窗口按 `stride=2` 不重叠地铺，因此奇数边长会丢掉最后一行/最后一列（`ceil_mode=False` 的行为）。\(3\times3\) 的 P5 只覆盖左上 \(2\times2\)，最大值不会包含第 3 行/列。P6 没有侧接卷积，所以 `lateral_params` 的长度是 \(L_{\text{in}}\) 而不是 \(L\)。

**账本**：

- `sizes`：融合后每级的 \( (H,W) \)，含 P6；
- `strides`：默认 \(2^{2+i}\)，显式给出时原样记录；P6 \( =2\cdot\text{strides}[-1]\)；
- `lateral_params`：逐级 \( \text{fpn}\cdot C_i+(b_i\ ?\ \text{fpn}:0) \)，只到 \(L_{\text{in}}\)；
- `num_upsample`：\(L_{\text{in}}-1\)，P6 的池化不算「放大 + 相加」；
- `anchors_per_level`：\(A\cdot H_lW_l\)，含 P6；总和即 `num_anchors`。

这套账本里最容易「自洽地算错」的是步长：用 \( \text{stride}=\lceil H_{\text{img}}/H_i\rceil \) 反推，在非 2 的幂尺寸上会得到 4/8/16 之外的值，而 FPN 的步长是网络结构决定的常量。本题因此把「按给定值原样记账」写进契约，并专门用 16×16 特征配 `strides=[3,7,13,29]` 的用例抓这种实现。

**dtype/device**：融合输出跟随输入特征；1×1 卷积要求权重与特征同 dtype（判题只用 float32/float64），放大与相加都是同 dtype 运算。偏置为 None 时不加，等价于零偏置但**不计参数**。

## 核心与易错点

1. **权重形状是 \( (\text{fpn},C_i) \)，不是 \( (C_i,\text{fpn}) \)**：转置的权重在通道数不等的级上形状对不上，直接报错；若恰好 \(C_i=\text{fpn}\) 则会静默算错。
2. **放大必须先于相加**：`fused[i] = lateral[i] + up(fused[i+1])`；先加后放大会得到完全不同（且形状错误）的结果。
3. **整数下标、向下取整**：用浮点 `scale` 乘完再 `round` 在 3→5 上给出 \( [0,1,1,2,2] \) 而不是 \( [0,0,1,1,2] \)。
4. **非整数倍必须支持**：真实 FPN 的相邻级通常是 2×，但本题允许 3→5；写死「每隔一行取一次」的实现会越界。
5. **P6 向下取整**：奇数边长丢最后一行/最后一列；用 `ceil_mode=True` 会多出一行/列，P6 的尺寸、anchor 数全错。
6. **P6 的前置条件**：\(H_5\) 或 \(W_5\) 为 1 时池化窗口放不下，必须抛 `ValueError` 而不是返回空张量。
7. **P6 不计侧接参数**：`lateral_params` 长度只有 \(L_{\text{in}}\)，`num_lateral_params` 不含 P6；但 P6 要进 `sizes`/`strides`/`anchors_per_level`。
8. **步长是声明值**：不从 \(H_i\) 反推；P6 的步长 = 2× 最粗一级。
9. **`num_upsample = L_in - 1`**：单级时为 0；P6 的池化不计入。
10. **偏置逐级可选**：`None` 与零偏置在数值上相同，但参数量不同；`lateral_bias` 长度必须等于级数。
11. **`fpn_dim` 级间一致**：每级权重第 0 维都必须等于同一个 `fpn_dim`，否则后面无法相加。
12. **尺寸取自输出而非输入**：`sizes` 是按融合后的张量记的；P6 之后的级数比输入多 1。
13. **不改输入**：卷积、`index_select`、加法都不原地写；`features` 列表本身也不应被追加 P6（返回的是新列表）。
14. **dtype 跟随**：float64 输入要得到 float64 输出，不能因为 Python 标量或默认 dtype 被降成 float32。

## 与邻近题的区别

- **vs `upsample-nearest`**：那题只做单张特征图的最近邻放大，考的是 `scale`/`size` 两套参数与下标；本题把放大当作多级融合里的一步，重点是通道统一、逐级相加与整本账。
- **vs `vision-anchor-match`（同批次）**：那题从特征图尺寸与步长**生成 anchor 框**并做 IoU 匹配；本题只产出供其消费的账本（尺寸、步长、每级 anchor 数），不生成任何框坐标。
- **vs `vision-fcos-centerness`（同批次）**：FCOS 用 `regress_ranges` 把不同尺度的 GT 分给不同层，是本账本的下游用法；本题不涉及标签分配。
- **vs `vision-deformable-sample`（同批次）**：可变形采样在连续偏移处做双线性插值；本题的放大是整数最近邻，没有亚像素位置。
- **vs `unfold-dilated-conv` / `unfold-fold-roll` 等算子题**：那些题聚焦单层张量的展开/折叠；1×1 卷积在本题只是通道矩阵乘，不是难点。

## 对拍口径与复杂度

- **主 oracle**：`_oracle` 用五重 Python 循环做 1×1 卷积（逐 \(n,o,y,x\) 对通道求和）、用整数下标做最近邻拷贝、用显式 `max` 做 P6 池化，账本也在 Python 里逐项累加；参考解用 `F.conv2d`、`index_select`、`F.max_pool2d` 与向量化的尺寸/参数计算，结构完全不同。覆盖 4 组配置：3 级 N=2 非方形（6→3、3→2）、2 级非方形含 P6 与自定义步长（2→5）、单级、4 级含 P6。
- **判别性用例**：示例的逐元素数值与全部账本；3→5 与 2→4 的放大映射；P6 在 3×3（最大值 5，不含 7/8/9）与 4×4（`[[6,8],[14,16]]`）上的窗口；参数量 \( [25,15,15,5] \) 与总和 60；锚数 \( [768,192,48,12,3] \) 与总和 1023；自定义步长 `[3,7,13,29]` 与 P6 的 58；单级时 `num_upsample=0`；float64 输入不被降精度；19 组非法输入。
- **实测非显然事实**：示例里 `lateral1` 的第二个通道是 \( 6+7+1=14 \) 而不是 13（bias 参与），导致 P4 的第二个通道是 `[[24,34],[44,54]]`；\(3\times3\) 的 P5 做 P6 时丢掉最后一行/列，最大值是 5 而不是 9；默认步长与示例特征尺寸（2×2 配 stride 4）并不自洽，这正是「步长不可由尺寸反推」的演示。
- **复杂度**：侧接 \( O(N\cdot\text{fpn}\cdot C_i\cdot H_iW_i) \)（矩阵乘），放大 \( O(\text{fpn}\cdot H_tW_t) \)，P6 \( O(\text{fpn}\cdot H_5W_5) \)。判题规模最大 \(16\times16\times5\)，远低于时限。真实 FPN 在 GPU 上用 cuDNN 卷积与 `upsample_nearest2d`，账本在构建期算一次即可。
