# 详解：pixel_shuffle 与 space-to-depth/depth-to-space

## 出处

- pixel_shuffle：Shi et al., *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network*（ESPCN），CVPR 2016，**arXiv:1609.05158**。论文提出 sub-pixel convolution：先在低分辨率上算出 \(C\cdot r^2\) 个通道，再用 "periodic shuffling" 重排成高分辨率——`torch.nn.PixelShuffle` 官方文档直接引用该文。注意文献对应关系是 ESPCN（1609.05158）；社区材料里偶见把它与 EDSR 混引的写法，本题不采用（该编号不对应本算子的原始文献，标待回源）。
- space-to-depth / depth-to-space：TensorFlow 标准算子（`tf.nn.space_to_depth` / `tf.nn.depth_to_space`），无论文编号。通道块排序在各框架间并不统一——ONNX 的 `DepthToSpace`/`SpaceToDepth` 默认 DCR 序、另设 CRD 模式吸收差异（torch 的 pixel_shuffle 族取 CRD 一侧的排序）；本题 s2d/d2s 认准与 `F.pixel_unshuffle`/`F.pixel_shuffle` 完全相同的通道序 \(c\cdot r^2+i\cdot r+j\)，测试以 `torch.equal` 钉死等价。
- `F.pixel_unshuffle`：torch 官方文档——"Reverses the PixelShuffle operation"，语义即 \(out[n,c\cdot r^2+i\cdot r+j,h,w]=x[n,c,h\cdot r+i,w\cdot r+j]\)；本题的 unshuffle/s2d 与它逐位对拍。

## 完整推导与下标约定

**一张双射表、四个方向**。核心恒等式（shuffle/d2s 视角）：

\[out[n,\,c,\,h\cdot r+i,\,w\cdot r+j]=x[n,\,c\cdot r^2+i\cdot r+j,\,h,\,w]\qquad(0\le i,j<r)\]

通道下标 \(c\cdot r^2+i\cdot r+j\) 的读法：\(c\) 是逻辑输出通道（外层），\((i,j)\) 是该通道在 \(r\times r\) 块内的行/列偏移（内层，先行后列）。空间下标 \(h\cdot r+i\)、\(w\cdot r+j\)：\((h,w)\) 是块坐标（慢变），\((i,j)\) 是块内偏移（快变）。unshuffle/s2d 是同一张表的反向读取：从输出的通道号里除 \(r^2\) 得 \(c\)、取余拆出 \((i,j)\)，从空间坐标里除 r 得块坐标、取余得偏移。四个 mode 共用这一张表，方向两两相逆。

**reshape/permute 实现路径**（以 d2s 为例）：\((N,C\cdot r^2,H,W)\) → view \((N,C,r,r,H,W)\)（把通道拆出块内 \((i,j)\) 两轴）→ permute \((0,1,4,2,5,3)\) 成 \((N,C,H,i,W,j)\) → reshape \((N,C,H\cdot r,W\cdot r)\)。permute 的目标是把 \(i\) 插到 \(h\) 后、\(j\) 插到 \(w\) 后——行主序内存里 reshape 合并的是**最后两维**，只有 \((h,i)\) 相邻、\((w,j)\) 相邻才能合并出 \(h\cdot r+i\) 与 \(w\cdot r+j\)。s2d 是同一路径取逆：\((N,C,H,W)\) → \((N,C,H/r,i,W/r,j)\) → permute \((0,1,3,5,2,4)\) → \((N,C,i,j,H/r,W/r)\) → reshape \((N,C\cdot r^2,H/r,W/r)\)。

**为什么 s2d ≡ pixel_unshuffle**：torch 的 `pixel_unshuffle` 官方语义就是上面那张反向表——channels-first 布局下与 TF space_to_depth 的「块塞回通道」是同一双射，**前提是通道块按 \((c\) 外层、\(i\) 行、\(j\) 列\()\) 排序**。框架差异只在块内排序约定（TF/ONNX 默认 DCR 把深度放最内，ONNX 的 CRD 模式才对齐 torch 的序）；实现时必须认准 \(c\cdot r^2+i\cdot r+j\) 这一种，测试对 6 组配置 `torch.equal` 钉死。

**r=1 退化**：块退化为 \(1\times1\)，\((i,j)\equiv(0,0)\)，重排表是恒等置换——四种 mode 都应 `torch.equal` 原样返回。判题只看值，允许任何值等价的实现路径。

**整除校验的分侧性**：shuffle/d2s 要求 \(C\bmod r^2=0\)（通道里藏着 \(r^2\) 个子通道），空间维随便放大；unshuffle/s2d 要求 \(H,W\bmod r=0\)（空间里藏着 \(r\times r\) 块），通道随便。校验放错侧会在另一个 mode 上漏检或误报。空间维只放大（shuffle）或只整除（unshuffle）的另一面是：不需要输出尺寸公式——这与 unfold/fold 的 \(H_{out}\) 公式形成对照。

**非连续输入**：transpose 得到的视图内存布局与逻辑形状不一致；实现要么先 `.contiguous()`（或让 `reshape` 自动拷贝），要么用对任意 stride 都正确的寻址——判题要求与同值连续输入 `torch.equal`。

## 核心与易错点

1. **permute 轴序**：d2s 的 \((0,1,4,2,5,3)\) 与 s2d 的 \((0,1,3,5,2,4)\) 方向相反，混用得到「块内转置」的错图——16 值手写案例立刻抓住。
2. **通道序认准 \(c\cdot r^2+i\cdot r+j\)**（c 外层）：写成 \(i\cdot r^2+c\cdot r+j\) 一类与原生对不上。
3. **拆轴要在 reshape 前完成**：直接 reshape \((N,C,r^2,H,W)\) 再 permute 会把 h 与块内偏移交错错位；必须先 view 出独立的 \(i,j\) 轴。
4. **r 的类型校验**：`isinstance(r, int)` 且排除 bool（Python 的 bool 是 int 子类，`True` 会冒充 1）；\(r<1\) 拒绝。
5. **整除校验分侧**：shuffle/d2s 查 C、unshuffle/s2d 查 H 与 W，两侧不要写串。
6. **双射自检**：输出是输入值的置换（展平后排序相等）——实现时用它快速抓「丢元素/重复元素」类 bug。
7. **不调用 F.pixel_\***：这是实现约束（考点），只在测试对拍侧允许出现原生。
8. **dtype 与入参保护**：纯重排不改 dtype、不修改入参；reshape/permute 路径天然满足，显式索引路径别顺手 `to()` 改类型。

## 与邻近题的区别

- **concat-stack-split**：那是「沿既有维堆叠/拆开」，元素按块复制、维长做加法，不是双射；本题体积严格不变、每个元素恰出现一次，只是换了地址——机制完全不同。
- **unfold-fold-roll**：unfold/fold 是「滑窗 ↔ 列」，窗口可重叠、fold 要累加，一般不可逆；本题重排永远无重叠、永远可逆（往返 `torch.equal`）。roll 则是一维循环移位，粒度是「整维转圈」而非「通道↔空间换位」。
- **im2col**：展开成列矩阵服务 GEMM，输出多出感受野维、体积膨胀 \(k_hk_w\) 倍；本题体积严格不变。
- 一句话记忆：concat = 复制堆叠，unfold/fold = 窗口↔画布（重叠区多对一），pixel_shuffle 族 = 严格换位。

## 对拍口径与复杂度

- **主判据**：与 `F.pixel_shuffle` / `F.pixel_unshuffle` `torch.equal`（纯重排无浮点误差，不用 allclose），6 组 \((n,c,h,w,r)\) 配置、四种 mode 同一配置全查；s2d≡unshuffle、d2s≡shuffle 的等价断言同为 `torch.equal`。
- **独立 oracle**：四层显式循环逐目标坐标算来源下标（d2s：来源通道 \(c\cdot r^2+(ho\bmod r)\cdot r+wo\bmod r\)；s2d：输出通道拆 \(c,i,j\) 后取 \(x[\cdot,\,c,\,hi\cdot r+i,\,wi\cdot r+j]\)），与向量化 reshape/permute 完全分路径；另加双射断言（展平排序相等）、往返恒等、r=1 恒等、非连续视图。
- **确定性数据**：黄金比例哈希伪随机张量（无 RNG），seed 按形状参数组合派生。
- **复杂度**：时间 \(O(N\cdot C\cdot H\cdot W\cdot r^2)\)——即 \(O(\text{numel})\)，每个元素一次寻址（输入输出元素数相等，重排不增值）；空间 \(O(\text{numel})\)（新张量；reshape/permute 路径含一次强制拷贝）。
