# 详解：searchsorted 分桶与 bincount 直方图

## 出处

- 直接对标的算子：`torch.searchsorted(boundaries, x, right=False)`（要求 `boundaries` 沿最后一维升序，逐元素二分）与 `torch.bincount(input, weights=None, minlength=0)`（要求 1-D 非负整数张量）。本题把两步合成一次调用，并把 `counts` 长度**钉死**为 \(M+1\)——相当于原生调用必须显式传 `minlength=M+1`。语义以 PyTorch 官方文档 + 本仓库实测为准。
- 应用出处（都是「查桶 + 计数」这一对）：
  - 反 CDF 采样：`top-p-sampling`、`rl-per-priority-replay` 里"取前缀和**严格大于** \(u\) 的最小位置"就是 `right=True` 口径（那题的推导已确认与 `searchsorted(cdf, u, right=True)` 等价）；
  - MoE 负载统计：`moe-switch-aux-loss` 用 `bincount(selected.reshape(-1), minlength=E)` 数每个专家被选几次，`minlength=E` 的作用与本题"尾部空桶也要占位"完全相同；
  - 码本编码：`quant-nf4-quantile-grid` 的 16 段分位码本 + 最近码，其"按 boundary 查段号"正是 `searchsorted`，只是题面用 \(\arg\min_k|u-T_k|\) 的显式距离写法。
- CPU、纯确定性：查桶只依赖比较，计数只依赖累加。

## 完整推导与下标约定

**两条口径等价于二分结果的证明**：设 `boundaries` 升序、长度 \(M\)。记 \(k(x)=\#\{j: b_j < x\}\)。由升序性，严格小于 \(x\) 的下标集合恰是前缀 \(\{0,\dots,k-1\}\)，于是
\[b_{k-1} < x \le b_k \quad(\text{约定 } b_{-1}=-\infty,\ b_M=+\infty),\]
即 \(k\) 就是"最小 \(i\) 使 \(b_i \ge x\)"——这正是 `side="left"` 的二分定义。同理 \(\#\{j: b_j \le x\}\) 给出"最小 \(i\) 使 \(b_i > x\)"，即 `side="right"`。因此参考实现直接写
\[\text{idx} = \sum_j [\,b_j < x\,]\quad\text{或}\quad \sum_j [\,b_j \le x\,]\]
（广播成 (N, M) 布尔矩阵后沿 M 求和），完全不调用任何二分，却与原生逐位一致——这也是本题**结构上独立**的解法路径。

**闭合方向随 right 翻转**：由上式，`right=False` 时 \(x\in(b_{i-1},\,b_i]\)（右闭），`right=True` 时 \(x\in[b_{i-1},\,b_i)\)（左闭）。starter docstring 里"左闭右开段"那一句是 `right=True` 的读法；判题以计数定义为准。**判别最小式**：取 \(b=[1,3,3,7]\)、\(x=[0,3,5,9]\)，只有 \(x=3\) 恰好压在重复段上，两支分别给 1 和 3；\(x=5,9,0\) 距边界都够远，两支相同。

**重复 boundary 造出恒空桶**：若 \(b_{i-1}=b_i=c\)，则桶 \(i\) 的区间在两种口径下都是空集（右闭时 \((c,c]\)、左闭时 \([c,c)\)），`counts[i]` 恒 0。示例的重复段 \([3,3]\) 给出**下标 2 这个恒空桶，两支都是 0**。会随 right 搬家的是"恰好压在 boundary 上的值"落到哪一格：`x=3` 在 `left` 下进桶 1（\((1,3]\)），在 `right` 下进桶 3（\([3,7)\)）——`counts[1]` 因此从 1 变 0、`counts[3]` 从 1 变 2。

**counts 的长度必须是 \(M+1\)**：原生 `torch.bincount(idx)` 的长度是 \(\max(\text{idx})+1\)（本仓库实测：\(b=[1,3,7]\)、\(x=[0,2]\) 得 idx \(=[0,1]\)，裸 bincount 给长度 2 的 `[1,1]`，`minlength=4` 才给 `[1,1,0,0]`）。题面把长度固定为 \(M+1\)，实现要么显式补零，要么像参考实现那样先 `torch.zeros(m+1)` 再 `index_add_`。

**bincount 是 scatter_reduce 的特例**：`counts[i] = Σ_{idx==i} 1`（或权重之和）恰好是 `reduce="sum"`、`src ≡ 1`（或 src=weights）、`include_self=False` 的一维 `scatter_reduce`；与同批次的 `scatter-reduce-aggregate` 是"特例 vs 完全一般化"的关系。`index_add_` 是它的等价紧凑写法，也保证了未触及位保持 0。

**dtype 的三处约定**：`idx` 恒 int64（与原生 `searchsorted` 一致）；未加权 `counts` 恒 int64；加权 `counts` 的 dtype 跟随 `weights`，且题面只投浮点权重。注意原生 `torch.bincount` 对**整型** `weights` 会静默升到 **float64**（实测 `bincount([0,1,1], weights=[1,2,3])` → `tensor([1., 5.], dtype=torch.float64)`），与本题约定不同；参考实现对整型权重先转 float32，判题不覆盖该分支，照抄原生行为不会被抓到、但会偏离 docstring。

**空输入的两种退化**：\(M=0\) 时两个计数口径都是 0，于是 `idx` 全 0、`counts` 长度 1 且等于 `x.numel()`（或权重和）——"唯一的桶是整个实轴"。\(x\) 为空时 `idx` 空、`counts` 是全 0 的长度 \(M+1\) 向量，多维空（如 (0,3)）也照常返回 `idx.shape == x.shape`。这两支都不该抛错。

**错误面**：`boundaries.dim() != 1` 抛 RuntimeError（对齐原生 `searchsorted` 的报错）；`weights` 形状与 `x` 不符抛 ValueError（本题自定义的参数检查）。两类故意不同，别统一。

## 核心与易错点

1. **counts 长度写死 \(M+1\)**：用裸 `torch.bincount(idx)` 会在尾部桶全空时短一截（实测 idx 为 \([0,1]\) 时裸调用只给长度 2）；判题的期望值统一写成 `bincount(..., minlength=M+1)` 并用 `torch.equal` 比对，长度不符直接失败，另有空 `x`／空 `boundaries` 用例把两端全空的形状钉死。
2. **right 只改变"等于 boundary"的元素**：把它实现成"整体往右挪一格"是错的——远离 boundary 的元素两支相同。
3. **恒空桶与搬家的格子是两件事**：重复段 \([3,3]\) 让桶 2 在两种 right 下都恒为 0；随 right 搬家的是压在 boundary 上的取值本身（示例里 `counts[1]` 1→0、`counts[3]` 1→2）。
4. **桶下标是"个数"而不是"命中的 boundary 下标"**：\(x\) 小于所有 boundary 时是 0，大于等于最后一个时是 \(M\)，取值域 \([0,M]\) 共 \(M+1\) 个值。
5. **`x` 多维时 `idx` 保持 `x.shape`、counts 用展平**：`counts.sum()` 恒等于元素总数，与形状无关。
6. **加权 counts 的 dtype 跟随 weights**：float64 权重必须给 float64 结果，不能在 float32 里累加再升回来（判题用 `torch.allclose` 且对拍原生 `bincount(weights=...)`）。
7. **未加权 counts 必须是整型**：写成 `index_add_` 一个 float 累加器、最后 cast，会在大 numel 上丢掉精确性；判题用 `torch.equal` 断言。
8. **不要给未排序 boundaries 加排序**：题面保证升序，排序会改变"桶 i ↔ 区间"的对应关系（boundary 值与段号绑定），而且参考实现是广播比较，本身对未排序输入给的是另一个数——判题不投，别自作聪明。
9. **整型权重不是浮点权重**：docstring 说的是"浮点张量"，判题只投 float32/float64。
10. **空 `x` 与空 `boundaries` 都合法**，返回值形状仍可预期（分别是 `idx.shape == x.shape`、`counts.shape == (1,)`）。

## 与邻近题的区别

- **vs `scatter-reduce-aggregate`（同批次）**：那里的 `reduce` 有五种、任意维数、含 `include_self` 两态；本题是一维、恒 `sum`、恒不含 self 的特例，重点在**查桶**这一半。
- **vs `gather-scatter`**：那题的 index 是**给定的**位置张量，做读取/覆盖写；本题的 index 是**由数值比较算出来的**桶号，是"值 → 下标"的映射。
- **vs `sort-argsort`**：排序后取 rank 也能表达分桶，但那是 \(O(M\log M)\) 的全局操作；本题 boundaries 已升序，只需逐元素二分/比较，且不需要 rank 的稳定性概念。
- **vs `cumsum-scan`**：前缀和类题里 `searchsorted` 常作为下游出现（在 CDF 上反查），本题不提供 cumsum，也不假设 boundaries 是累积和。
- **vs `moe-switch-aux-loss`**：那题只做"整数下标的计数"（桶号已是输入），本题多整个查桶步骤与 `right` 口径。
- **vs `grid-sample` / `upsample-bilinear`**：那些也做"坐标 → 下标"，但用仿射坐标变换与双线性权重，不涉及有序边界与闭合方向。

## 对拍口径与复杂度

- **主 oracle**：`torch.searchsorted(b, x, right=...)` + `torch.bincount(idx, minlength=M+1)`（加权时 `bincount(idx, weights=w, minlength=M+1)`），覆盖 5 组随机 (M=7, N=40) × 两种 right × 加权/未加权、多维 `x`（(5,8)）、int64 一组、float64 一组；下标与整型计数用 `torch.equal`，浮点计数 \(10^{-6}\)。
- **第二 oracle**：`_oracle_scan` 用纯 Python 的 `while bl[i] < v: i += 1` **线性扫描**求桶号，再用 `list.count` / 逐桶 `sum` 求计数——与解法"广播比较求和 + index_add_"和原生"二分"都不同，专门抓 off-by-one（`<` 写成 `<=`）与尾部漏桶；其用例包含重复 boundary（`[2,2,2,5,5]`）、空 boundaries（float 与 int 各一组）、(3,4) 二维 `x` 加权。
- 判别性用例：示例的两支对照（`idx_left/idx_right`、`counts_left/counts_right` 全部实测回填）、重复 boundary 的显式下标断言、端点恰好相等时两支相差 1、越界两侧（\(-5,0,100\)）、空 `x`（含 2-D 空与加权空）、空 boundaries（含加权与双空）、加权与未加权的 dtype/求和恒等式、float64。
- 复杂度：广播比较解法为 \(O(N\cdot M)\) 时间、\(O(N\cdot M)\) 临时布尔矩阵；原生二分是 \(O(N\log M)\) 但常数更小。计数一步 \(O(N)\)（`index_add_` 或 `bincount`），输出 \(M+1\)。题面不限制实现路径，但判题规模（\(N\le 40\)、\(M\le 9\)）下两种都远快于容差要求。
