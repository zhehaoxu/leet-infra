# 详解：视觉检测：多尺度 anchor 生成与 IoU 匹配

## 出处

- anchor 机制与「多尺度特征图上按尺度/宽高比铺框、再按 IoU 阈值分配正负样本」的原始工程口径：Liu et al., *SSD: Single Shot MultiBox Detector*, arXiv:1512.02325（ECCV 2016），第 2.2 节 matching strategy 给出 \( \mathrm{IoU}>0.5 \) 为正、按最大 IoU 匹配的两条规则。
- 保底规则与 ignore 带：Lin et al., *Focal Loss for Dense Object Detection*（RetinaNet）, arXiv:1708.02002，第 4 节 anchor assignment——正样本 \( \mathrm{IoU}\ge 0.5 \)、负样本 \( \mathrm{IoU}<0.4 \)、\( [0.4,0.5) \) 不参与 loss（本例的 ignore），并且每个 GT 至少保底一个 anchor。
- 中心点约定 \( ((x+0.5)\cdot\mathrm{stride}, (y+0.5)\cdot\mathrm{stride}) \) 与「先成框再裁剪」沿用 SSD/RetinaNet 系实现（Detectron2 `anchor_generator.py`、torchvision `AnchorGenerator` 的 `_generate_anchors`）。本题不依赖这些库，判题环境只有 torch。
- CPU、纯确定性：anchor 坐标只由整型格坐标与浮点常数算出，IoU 只由四则运算算出，没有随机数，也不读图像。

## 完整推导与下标约定

**枚举顺序决定下标**。设第 \(l\) 层特征图 \(H_l\times W_l\)、步长 \(s_l\)，尺度列表 \(S\)、宽高比列表 \(R\)。anchor 的总顺序是

\[
l \;\to\; y=0..H_l-1 \;\to\; x=0..W_l-1 \;\to\; a\in S \;\to\; r\in R,
\]

即同一格内的 \( |S||R| \) 个 anchor 连续排列，行优先于列。参考实现先在每层构造 \( (H_l,W_l,|S||R|,4) \) 的 `combos`（对每个 \((s,r)\) 各算一张 \( (H_l,W_l,4) \) 坐标图，`stack(dim=2)` 后一起 `reshape(-1,4)`），再按层 `cat`。下标是**格优先**而不是尺度优先——这一点直接决定保底规则挑中哪个 anchor，也决定 `anchor_enumeration` 用例里 `anchors[2]` 已经跳到 \(x=1\)。

**中心与尺寸**。格 \((x,y)\) 的中心是

\[
c_x=(x+0.5)\,s_l,\qquad c_y=(y+0.5)\,s_l,
\]

与 \(H_l,W_l\) 无关（只与 stride 有关，这是「同一位置在各层共享中心」的常见做法）。对 \((s,r)\)：

\[
w=s\sqrt r,\qquad h=\frac{s}{\sqrt r},\qquad \text{area}=wh=s^2 .
\]

面积恒等于 \( s^2 \)，与宽高比无关——这条恒等式是后面「并列 IoU」的来源。

**裁剪**。先算完整框，再逐坐标 `clamp(0, W_img)` / `clamp(0, H_img)`。裁剪会让离图像边缘的格子失去部分宽高，甚至退化为零面积；这些 anchor **仍然保留**，只是面积变小、IoU 变小。示例里格 \((0,0)\) 的框是 \((-4,-4,12,12)\)，裁剪后为 \((0,0,12,12)\)，面积从 \(16\times16=256\) 降到 \(12\times12=144\)。

**IoU 口径**。交集用裁剪后的坐标：

\[
\text{inter}=\max(0,\min(x_2^a,x_2^g)-\max(x_1^a,x_1^g))\cdot
             \max(0,\min(y_2^a,y_2^g)-\max(y_1^a,y_1^g)),
\]
\[
\mathrm{IoU}=\frac{\text{inter}}{S_a+S_g-\text{inter}},\qquad \text{若 } S_a+S_g=0 \text{ 则记 } 0 .
\]

对示例 GT \((4,4,20,20)\)（面积 256）实测四个 IoU：

```
anchor 0 (0,0,12,12)    inter=64   S_a=144  union=336  IoU=0.19047619
anchor 1 (4,0,20,12)    inter=128  S_a=192  union=320  IoU=0.4（float32 算出 0.40000000596）
anchor 2 (0,4,12,20)    inter=128  S_a=192  union=320  IoU=0.4
anchor 3 (4,4,20,20)    inter=256  S_a=256  union=256  IoU=1.0
```

**三档划分与等号归属**。规则是

\[
\text{label}=\begin{cases}
\arg\max_g \mathrm{IoU}, & \mathrm{best\_iou}\ge \mathrm{pos\_thr}\\
-2, & \mathrm{neg\_thr}\le \mathrm{best\_iou}<\mathrm{pos\_thr}\\
-1, & \mathrm{best\_iou}<\mathrm{neg\_thr}
\end{cases}
\]

等号归「忽略」。示例里 anchor 1、2 恰好是 0.4：按浮点算出来是 0.40000000596，\( \ge \mathrm{neg\_thr} \) 且 \( <\mathrm{pos\_thr} \)，所以是 -2。把 `neg_thr` 降到 0.2 时它们仍是 -2，而 0.190476 的 anchor 0 仍 \( <0.2 \) 为 -1；把 `pos_thr` 也降到 0.4 时它们翻成正样本，得到 `[-1,0,0,0]`、counts `[3,1,0]`——这组三重阈值的实测正是「等号归属」的判别点。

**保底规则**。对每个 GT \(g\)，在 \( \mathrm{best\_gt}=g \) 的 anchor 里取 \( \mathrm{best\_iou} \) 最大者，强制置正。它必须**在三档划分之后**执行：先算 `best_iou >= pos_thr`、再算忽略带、最后覆盖，否则被忽略带写过的位置会互相打架。示例的极小 GT \((15.9,15.9,16.1,16.1)\)：四个 IoU 是 \(0,0,0,0.00015625\)（前三个框与 GT 无交集），全部低于 0.4，阈值规则给出 `[-1,-1,-1,-1]`；保底把 anchor 3 改成 0，得 `[-1,-1,-1,0]`、counts `[1,3,0]`。若漏掉保底，这类「小目标」会一个正样本都分不到，训练直接失效——这就是它存在的理由。

**并列的两个层次**。GT 之间并列时，`best_gt` 取**下标最小**的 GT（`max(dim=1)` 的默认行为）；保底时若多个 anchor 的 IoU 完全相等，取**下标最小**的 anchor。二者都要显式确定，因为判题用 `torch.equal` 逐位比较标签。

**无 GT**。\(G=0\) 时没有 `best_gt`、没有保底步骤，但 anchors 必须照常生成（下游要按 \(A\) 建分类头），标签全 -1，counts 为 `[0, A, 0]`。`check_no_gt` 还断言「有 GT 与无 GT 两次调用返回的 anchors 逐位相同」，防止实现把 anchor 生成写在 GT 分支里。

**dtype**。anchors 跟随 `gt_boxes`（float32/float64），labels 与 counts 恒为 int64。float64 下锚点坐标与 IoU 都按双精度算，`check_float64_and_dtype` 用 \(10^{-12}\) 容差验证。

## 核心与易错点

1. **枚举顺序是格优先（层→行→列→尺度→宽高比）**，不是尺度优先。用 `meshgrid` 得到 \((H,W)\) 的坐标图后，必须对 \((s,r)\) 用 `stack(dim=2)` 再 `reshape(-1,4)` 才能得到「同格 \( |S||R| \) 个连续」；若先串尺度再串格，anchor 数一样但下标全错，保底规则挑中的 anchor 也随之改变。
2. **中心只由 stride 决定**，不要用 `W_img / W_l` 之类的反推；示例里 `W_img=32, W_l=2, stride=8` 恰好自洽，掩盖不了错误，所以 `check_anchor_enumeration` 特意用 stride 16 + 1×1 特征图 + 32×32 图像来拉开差距。
3. **先成框再裁剪**，不是先裁中心；裁剪后坐标为 0 面积的 anchor 不能丢，否则 `counts` 总数与 \(A\) 对不上。
4. **等号归忽略带**：`>= neg_thr` 而不是 `> neg_thr`。示例中恰好等于 0.4 的 anchor 是专门构造的，`[-1,-2,-2,0]` 与 `[-1,-1,-1,0]` 一字之差就判错。
5. **保底规则的执行时机**在三档之后；并且只在「该 GT 已有 anchor 指向它」时生效（`cand` 为空要 `continue`，不能用 `argmax` 在空张量上炸）。
6. **GT 并列与 anchor 并列都要取小下标**：`torch.max(dim=1)` 默认返回第一个最大值，`argmax` 也是；但如果你用 `sort` 或 `topk`，稳定性要自己保证。
7. **无 GT 仍要生成 anchors**，且与有 GT 时完全一致；不要提前 `return empty`。
8. **零面积 IoU 记 0**，不能产生 NaN；`union > 0` 的 `torch.where` 两支都会求值，所以除法结果可能含 NaN/Inf，必须靠 `where` 选掉。
9. **阈值校验是 \(0\le \mathrm{neg\_thr}\le \mathrm{pos\_thr}\)**，`neg_thr > pos_thr` 必须抛 `ValueError`（否则忽略带为空，语义不成立）。
10. **精确并列的保底选择对浮点噪声敏感**：实测在 \((3,4)\) 网格、\(s\in\{8,12\}\)、\(r\in\{0.5,1,2\}\) 的配置里，anchor 69（\(s=12,r=0.5\)）与 anchor 70（\(s=12,r=1\)）对 GT \((20,20,40,40)\) 的 IoU 数学上都是 \(72/472\)（因为面积恒为 \(s^2=144\)、交集都是 72），float32 与 float64 的末位差异会让两者分别胜出。题目只承诺「每 GT 恰有一个正样本」，不承诺并列时挑哪一个；对拍口径见下节。
11. **counts 必须与 labels 自洽**：三者之和 \(=A\)，且分别等于 `>=0`、`==-1`、`==-2` 的计数；实现里直接从 labels 计数，不要另算一遍。

## 与邻近题的区别

- **vs `vision-fcos-centerness`（同批次）**：那道题在已经确定为前景的位置上算 FCOS 的 center-ness 软标签 \( \sqrt{\min(l,r)/\max(l,r)\cdot\min(t,b)/\max(t,b)} \)，输入是「位置 → GT」的分配结果；本题的产物正是那个分配结果（硬标签 + 保底），输出是离散标签而不是连续质量分数。
- **vs `vision-fpn-levels`（同批次）**：FPN 层选择按 GT 尺寸的对数尺度把目标指派到某一层（\(k=\lfloor k_0+\log_2\sqrt{wh/224}\rfloor\) 之类），只决定「哪一层负责」，不生成框、不算 IoU；本题同时在多层铺框，并用 IoU 决定正负与忽略。
- **vs `vision-rotated-nms`（同批次）**：那道题在检测后处理阶段对候选框做旋转 IoU 抑制，输入是已打分的框；本题在前处理阶段决定哪些 anchor 参与训练。
- **vs `vision-rle-mask-iou`（同批次）**：那是分割掩码的 RLE 编码与掩码 IoU（像素级集合运算）；本题是矩形框 IoU。
- **vs `iou-variant-losses`（既有题）**：GIoU/DIoU/CIoU 只算两个**已给定**框之间的广义 IoU，不涉及 anchor 网格与标签分配；本题的 IoU 是轴对齐基础版，重点是枚举顺序与阈值语义。

## 对拍口径与复杂度

- **主 oracle**：`_oracle` 是纯 Python 四重循环（层→行→列→尺度→宽高比）逐个生成 anchor，再逐 anchor 逐 GT 用 `_iou_loop` 算 IoU、逐 GT 做保底扫描；与参考解的「`meshgrid` 张量坐标图 + `_iou_matrix` 广播矩阵 + `argmax`」在结构上完全不同。测试覆盖 4 组配置：单层单 GT、多 GT + 多尺度多比例的边界情况、两层不同 stride、以及 8×8 图像下的重度裁剪。
- **并列容差**：`check_pointer_oracle` 先算 oracle 的 IoU 矩阵与该 GT 的最大 IoU \(g_{\max}\)，把 \( \mathrm{IoU}\ge g_{\max}-10^{-6} \) 的 anchor 视为「保底并列候选」；只有当某一侧为正、另一侧为负/忽略**且该位不是并列候选**时才失败。同时断言「每 GT 至少一个正样本」「每个正样本标签对应的 IoU 要么是 \(g_{\max}\) 要么 \( \ge \mathrm{pos\_thr}\)」「正样本总数与 oracle 相同」「counts 与 labels 自洽」。这样既锁死了算法语义，又不会因为浮点末位差异误伤。
- **判别性用例**：示例常量（`EXAMPLE_*` 实测回填）、恰好 0.4 的等号归属与三组阈值（`[-1,-2,-2,0]` / `[-1,-2,-2,0]` / `[-1,0,0,0]`）、极小 GT 的保底（`[-1,-1,-1,0]`）、两 GT 各自保底、重复 GT 的并列取小下标、无 GT 时 anchors 一致、8×8 图像的裁剪、float64 dtype、10 组非法输入。
- **实测非显然事实**：anchor 面积恒为 \(s^2\)，所以不同 \((s,r)\) 组合可以在特定 GT 上产生数学上完全相等的 IoU（实测 \(72/472\)），保底选择因此依赖浮点末位；float32 下 0.4 被算成 0.40000000596，等号归忽略带的判定因此依赖「\( \ge\mathrm{neg\_thr}\)」而不是浮点相等；裁剪会把 256 面积的框压成 144，IoU 从 1.0 掉到 0.190476。
- **复杂度**：生成 \(O(A)\)；IoU 矩阵 \(O(AG)\) 时间、\(O(AG)\) 内存（广播）；保底 \(O(A+AG)\) 的 `nonzero`/`argmax`。判题规模 \(A\le 72\)、\(G\le 2\)，远低于时限。工程实现里会把 \(A\) 做成十万级并分块算 IoU，本题不要求。
