# 详解：布局账——shape/stride/offset 与 view-or-copy 判定

## 出处

教程侧是官方 *View operations*（`docs/source/beginner/basics/views_tutorial.py`）与 `torch.Tensor` 页里 `reshape`/`view`/`expand`/`is_contiguous` 的口径说明：它们只给出「一定视图 / 可能视图 / 一定复制」三类清单和 `view` 的失败条件，具体步长怎么填、`reshape` 何时退化成复制都得看实现。

实现侧（pytorch/pytorch v2.8.0，`aten/src/ATen/`）逐条对应本题七条规则：

- `native/TensorShape.cpp:3884-3911` `inferUnsqueezeGeometry` → 规则 2 的 `size[d] * stride[d]`（追加到末尾时取 1）。
- `native/TensorShape.cpp:3823-3860` `inferSqueezeGeometry` → 规则 3：只做分量删除，从不重算留下的步长。
- `native/TensorShape.cpp:1342-1370` `expand`（调用 `inferExpandGeometry_dimvector`）→ 规则 5 的右对齐逐维判定与新前置维步长。
- `native/TensorShape.cpp:1669-1695` `narrow` 直接 `return at::slice(self, dim, start, start + length, 1)`，`:2232` `select_symint` 与 `:3033-3089` `slice` → 规则 4 其实是同一段 offset 推进代码（归一化、夹紧、`end` 缺省 `INT64_MAX`、长度 `(len + step - 1) / step` 向上取整、步长 `*= step`）。
- `TensorUtils.cpp:330-410` `at::detail::computeStride_impl` → 规则 6 的「连续子空间切段 + 目标维精确吃满」，这就是 `view()` 报 `at least one dimension spans across two contiguous subspaces` 的那个函数。
- `native/TensorShape.cpp:1997-2018` `reshape_symint` → 拿到 `computeStride` 的 `nullopt` 时改成 `clone(Contiguous)` 再 view，即本题的 `copy`；`:4118-4150` `flatten` → 三条分支（0 维、`start == end` 原样返回、其余走 reshape）。
- C 连续判定与 `contiguous()` 的「已连续就返回 self」按 `c10::TensorImpl::is_contiguous` 语义实测复现（规则 7）。

## 核心与易错点

1. **`view` 与 `reshape` 同源但口径不同**：`reshape` 用 `computeStride` 判段，判不出来就自己复制一份，永不报错；`view` 判不出来直接抛异常。本题只考 `reshape`/`flatten`，因此每一步都有确定答案。
2. **正确心智模型是「切段 + 吃满」，不是「压平再拆」**：`(3,4,2)/(4,1,12)` 的段是 `[2]`（基步长 12）和 `[4,3]`（因为 `stride[0] = 4` 恰等于内段累乘 4 × 基步长 1，两段合成一条，基步长 1、段长 12）。目标 `(12,2)` 的 2 吃满第一段得步长 12、12 吃满第二段得步长 1 → view `(1,12)`；目标 `(3,8)` 的 8 既吃不满 2 也吃不满 12 → copy；目标 `(24,)` 想把两段合成一个 24 也不允许 → copy。用「先把长度乘起来再拆」的直觉三处都会算错。
3. **「看起来只差一步」的转置/展开案例**：`(4,3)/(1,4)`（转置来的矩阵）`reshape(12)` 与 `reshape(2,6)` 都是 **copy**（行主序实际读到 0,4,8,1,5,9…，没有单步长能读满，两个 6 也吃不满任何一段），但只拆外层的 `reshape(2,2,3)` 是 view `(2,1,4)`；`expand` 出的 stride-0 维同理必 copy（`(3,4)/(0,1)` 的 `reshape(12)`、`reshape(1,12)`），而 `(4,2,3)/(1,12,4)` 的 `reshape(4,6)` 不 copy。示例链 1、2、4、7 就是这几对——链 7 里 `reshape(12)` 复制之后，紧接着的 `reshape(2,6)` 立刻又变回 view，因为新存储就是标准行主序。
4. **长度 1 的维步长是「自由」的，但 torch 填得死死的**：`unsqueeze` 用右邻那一段的跨度，`expand` 新增的前置 1 用 `size[i+1] * stride[i+1]`（邻居若是 stride-0 维就跟着变 0），复制后的标准连续步长遇到长度 ≤ 1 的维不推进累乘。只比 shape 不比 strides 的题面会漏掉这一整层。
5. **`squeeze` 只删分量、绝不重算**：`(2,1,3)/(9,9,1)` 删掉维 1 得到的是 `(2,3)/(9,1)`（不是 `(3,1)`），因此 `is_contiguous()` 为假；而指定的维长度不是 1 时整步是空操作 —— `(2,1,3)/(3,3,1)` 上 `squeeze(0)` 原样返回。示例链 3 的 `unsqueeze` 两次再 `squeeze(None)` 正好把插进去的两维删干净，回到 `(3,4)/(4,1)`。
6. **空切片仍推进 offset，且步长照样乘上 step**：`(6,8)/(8,1)` 上 `slice(1, 5, 1, 2)` 得到 `(6,0)/(8,2)` offset=5，`is_contiguous()` 为 `True`（总长度为 0 恒连续）。`narrow(0, 6, 0)` 同理 offset=48。
7. **总长度为 0 的分支**：`computeStride_impl` 在 `numel == 0` 时永不返回 `nullopt`，所以 `reshape` 恒为 view —— 与旧形状相同照抄旧步长，否则给标准连续步长（`(2,0,4)/(8,4,1)` 的 `reshape(3,-1)` 给出 `(3,0)/(1,1)`）。0 维基张量的 `reshape` 走 `oldshape.empty()` 分支，步长全 1。
8. **`contiguous()` 不是「永远是复制」**：已连续时返回自身，`offset` 保留（`(2,3)/(3,1)/4` 之后仍是 4），不连续时才新分配并把 offset 归 0。判 `copy` 时 offset 归零、步长换成标准连续步长，这两件事同时发生。
9. **`flatten(a, a)` 原样返回 `self`**：不经过 `computeStride`，所以哪怕该维落在两段交界处也不会复制；0 维张量则先 `reshape({1})`。示例链 5 连着两次 `flatten(0,1)`：第一次把 `(2,2,4)/(12,8,1)` 的前两维合并，`stride[0] = 12 ≠ 2 × 8` 说明它们不同段 → copy 成 `(4,4)/(4,1)`；第二次输入只剩两维且已连续 → view。

## 定义域外的真实行为（刻意排除）

- 操作集里没有 `view`/`_reshape_alias`/`as_strided`/`movedim`/`unfold`/`diagonal`：前两者与 `reshape` 同源但会抛异常，作为题目操作不 total；`as_strided` 直接写元数据等于把答案放进输入；后三者是带窗口语义的复杂几何变换，超出「布局账」范围。
- `Tensor.slice` 这个方法在 Python 侧并不存在（实测 `AttributeError`），torch 的切片走的是基础索引 `t[(slice(None),)*d + (slice(s,e,st),)]`，测试 oracle 用的就是这条路；题面把它收成一条 `["slice", d, start, end, step]`，`end` 一律显式给值，省略 `end`（`INT64_MAX`）的分支不考。
- `memory_format`（`channels_last` 等）与 `torch.Tensor.stride(dim)` 的单维查询不在范围内；本题只讨论 C 连续。
- 长度为 0 的维上的步长在不同后端/版本间确实可能不同（`computeStride_impl` 的注释里就写了 "stride is arbitrary in the numel() == 0 case"），本题按 torch 2.14.1 的同一式子要求，属于「实现即规范」的口径。

## 对拍口径

solution 是纯 Python 的元数据推演，不调用任何 torch 接口。`test_cases.py` 拿 torch 当 oracle：

- 基元数据用 `torch.arange(n).as_strided(shape, strides, offset)` 复现，`n = offset + Σ(size-1)*stride + 1`（`max(n, 1)`），因此任意（含重叠、stride-0、越不出界的）元数据都能落成真实张量。
- `kind` 的判据是 `untyped_storage().data_ptr()` 是否改变。这条判据在本操作集内不会假阴：`numel == 0` 时 `computeStride` 永不返回 `nullopt`、`contiguous()` 在已连续时直接返回自身，两者都不产生新存储。
- 随机链：7 个固定种子 × 120 条链，基 ndim ≤ 4、链长 ≤ 8，与被测约束同域，约 3800 个 step 逐 step 比对 `shape/strides/offset/contiguous/kind` 五项全等；随机源是 `random.Random(seed)`，无全局状态污染。生成器保证每步都能被 torch 接受（拒收就换下一候选），因此不需要复现异常文本。
- 题面 8 条示例链由参考实现与 torch 双向实测得出，另有 28 条专项断言链逐条回灌 torch 复核（`check_offset_arithmetic` 等四组专项）。

## 复杂度

每步 O(ndim)，单条链 O(链长 × ndim)；全量对拍在本机 1 秒内完成。
