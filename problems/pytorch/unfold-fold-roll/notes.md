# 详解：unfold/fold/roll：im2col、col2im 重叠累加与循环移位

## 出处

- `F.unfold`：torch 官方文档（`torch.nn.Unfold` / `F.unfold`）——"Extracts sliding local blocks from a batched input tensor"，输出 \((N,\,C\times\prod(\text{kernel\_size}),\,L)\) 并给出 \(L\) 的整除公式；文档明确二维空间维情形 "this operation is sometimes called `im2col`"。im2col 谱系源自 Caffe 把卷积摊成 GEMM 的同名预处理。
- `F.fold`：torch 官方文档（`torch.nn.Fold` / `F.fold`）——"Combine an array of sliding local blocks into a large containing tensor"，与 `Unfold` 的输出规格互为镜像，"by summing the overlapping values"；文档同样给出 \(L\) 与 output_size 的匹配式，二维情形 "sometimes called `col2im`"（Caffe 谱系）。
- `torch.roll`：torch 官方文档——"Elements that are shifted beyond the last position are re-introduced at the first position. If `dims` is `None`, the tensor will be flattened before rolling and then restored to the original shape."（后半句正是本题约定刻意偏离的原生行为。）
- 本题自己的口径（starter docstring）：stride=None → kernel_size 的默认、roll 的 dims=None 不展平、int+元组只 roll dims[0]。这三条与原生默认不同，判题以 starter 为准。

## 完整推导与下标约定

**下标约定**：输入 \(x\ (N,C,H,W)\)，核 \((k_h,k_w)\)，步幅 \((s_h,s_w)\)，膨胀 \((d_h,d_w)\)，填充 \((p_h,p_w)\)；\(H_{out}=(H+2p_h-d_h(k_h-1)-1)//s_h+1\)，\(W_{out}\) 同理，\(L=H_{out}W_{out}\)。

**unfold 的元素账**：

\[cols[n,\; c\cdot k_hk_w+i\cdot k_w+j,\; hi\cdot W_{out}+wi] = xp[n,c,\; hi\cdot s_h+i\cdot d_h,\; wi\cdot s_w+j\cdot d_w]\]

其中 \(xp\) 是零填充后的图（画布坐标从 \(-p_h,-p_w\) 起），越界即 0。行序把 \((c,i,j)\) 摊成 \(c\cdot k_hk_w+i\cdot k_w+j\)——**通道最外、核内先高后宽**，与权重 \((C_{out},C_{in},k_h,k_w)\) 的展平序对齐，这正是 im2col 矩阵乘能一次成立所需的排列（呼应 `im2col` 题）；列序 \(hi\cdot W_{out}+wi\) 让 W 方向相邻窗口在列上相邻。

**fold 的散射账**（同一账反向）：

\[out[n,c,\; hi\cdot s_h+i\cdot d_h-p_h,\; wi\cdot s_w+j\cdot d_w-p_w] \;\mathrel{+}=\; cols[n,\; c\cdot k_hk_w+i\cdot k_w+j,\; hi\cdot W_{out}+wi]\]

落在画布内才累加，画布含 padding 带、最后裁掉。**重叠求和的来源**：输出像素 \((h,w)\) 被多少组满足两式的 \((i,j,hi,wi)\) 命中就累加多少次——stride<kernel 时必然多次；全 1 cols、k=2、s=1 的账是角 1、边 2、中心 4，可以逐格手验。

**fold∘unfold 的可逆性**：stride=kernel、padding=0（且空间维可整除）时每个输入位置恰属一个窗口，重排是双射，`torch.equal` 恒等；padding>0 时只要 pad 后尺寸仍被 kernel 整除，fold 裁边后同样严格恒等。stride<kernel 时窗口重叠、多个输入位置映射到同一输出位置，fold 只能"求和恢复"，信息已不可逆——原生文档强调 overlapping 值相加、无法精确逆 unfold，正是此意。

**roll 的索引账**：逐维独立地 \(out[pos]=x[(pos-s)\bmod n]\)（等价地说：元素 \(i\) 的新位置是 \((i+s)\bmod n\)）。负 shifts 与超长 shifts 全部落入 \(\bmod\,n\) 的同一等价类：长度 5 上 shift 7 ≡ 2、shift −2 ≡ 3。长度 0 的维无元素可移，原样返回（先做 \(n>0\) 守卫，否则 mod 0 除零）。负维先 \(+ndim\) 归一化，越界抛 `IndexError`。

**与原生 `torch.roll` 的两处刻意差异**：① dims=None 时本题 roll **最后一维**，原生先**展平**再 roll 再复原——对 \((2,3)\) 张量 shifts=1，本题 \([[3,1,2],[6,4,5]]\)、原生 \([[6,1,2],[3,4,5]]\)；② shifts 为 int、dims 为元组时本题只 roll dims[0]，原生不接受该组合。测试把原生结果一并 assert，钉死"这是约定、不是 bug"。

**unfold 的 padding 语义**：窗口左上角从 \((-p_h,-p_w)\) 开始——第一个窗口一半在画布外；采样查零填充图，而不是跳过越界元素。dilation 在切片上表现为 \(i\cdot d_h\) 的额外偏移。

## 核心与易错点

1. **行列双序**：\(row=c\cdot k_hk_w+i\cdot k_w+j\)、\(col=hi\cdot W_{out}+wi\)。写成"窗口最外层"或先 \(j\) 后 \(i\)，2 通道/多核案例立刻翻车。
2. **fold 必须 `+=` 累加**：写成覆盖赋值，重叠区只剩最后一个窗口的贡献。
3. **fold 的 L 匹配按 output_size 反算**：output_size 给大或给小都得到 \(L\ne H_{out}W_{out}\) → `RuntimeError`；不能拿 cols 的 L 反推 output_size。
4. **stride=None → kernel_size**（unfold/fold 都生效），与原生默认 stride=1 不同——对拍原生时必须显式传 stride。
5. **roll 约定**：dims=None 不展平；int+tuple 只 roll dims[0]。别顺手调 `torch.roll` 当实现。
6. **错误分类**：非 4 维/非 3 维输入、非法 mode、缺 output_size、shifts 缺失或与 dims 长度不配 → `ValueError`；窗口过小（unfold）、fold 形状不配 → `RuntimeError`；dims 越界 → `IndexError`。
7. **shift 取模与空维**：先 `int()` 再取模；长度 0 的维跳过。
8. **返回新张量**：unfold/fold 用新画布写入、roll 用 `index_select`，天然不改输入；不要用原地操作"优化"。

## 与邻近题的区别

- **im2col**：那题考"卷积 = 一次 GEMM"，unfold 只是它用的工具（且布局是 \((N,\,oh\cdot ow,\,C\cdot k_hk_w)\)：行 = 输出位置、列 = 感受野，与本题 cols 的 \((N,\,C\cdot k_hk_w,\,L)\) 呈转置关系）；col2im 不在那题范围。本题把 im2col/col2im 成对实现并钉死官方行列排列。
- **unfold-dilated-conv**：只做膨胀取块，输出保留窗口维 \((B,C,oh,ow,k_H,k_W)\)，无 padding/stride、无 fold 侧；本题是全参数 unfold + fold + roll。
- **concat-stack-split**：拼接/拆分是"沿既有维堆叠"，元素按块复制、维长做加法；本题三模式是纯置换（无重叠时 unfold/fold 与 roll 都是双射），元素总数不变。
- **原生算子**可以当 oracle，但三处约定差异（stride 默认、dims=None、int+tuple）必须按本题口径写实现，否则判别用例会抓住。

## 对拍口径与复杂度

- **双 oracle**：原生 `F.unfold`/`F.fold`（7 组参数 × 2 形状，`allclose(1e-6)`；stride=None 约定单独一组）与 `torch.roll`（13 组 shifts/dims，`torch.equal`）；再叠独立纯 python 多重循环 oracle（unfold 逐窗口逐核元素带边界判断、fold 逐窗口 `+=`、roll 逐元素 \((pos-s)\bmod n\)），与向量化实现分路径。
- **判别口径**：2 通道窗口排列手写案例、零填充窗口、dilation 跳格、重叠累加 ones 账、stride=kernel 与 padding 可整除两种往返恒等、roll 两处约定（含与原生展平结果的对照 assert）、全错误分类。
- **复杂度**：unfold/fold 的输出含 \(N\cdot C\cdot k_hk_w\cdot L\) 个元素，逐元素一次写入/累加，时间与空间均 \(O(N\cdot C\cdot k_hk_w\cdot L)\)（窗口数 \(L=H_{out}W_{out}\)；stride=kernel 时为 \(O(NCHW\cdot k_hk_w/(s_hs_w))\) 量级，即输出体积本身）；roll 是 \(O(\prod \text{shape})\) 的索引重排。核元素数 \(k_hk_w\) 是常数级小量，主成本与输出体积同阶。
