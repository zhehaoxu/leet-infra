# 870 实现笔记

## 官方逐字执行环境

`official870.py` 从 `filtering.py` 用 `ast.get_source_segment` 抽出`_add_noise_shaping` / `_apply_probability_distribution` / `dither` 三个函数体，`exec` 进同一共享 namespace，官方函数体**一字未改**。注入真实 `torch`（`set_default_dtype(float64)`）、`math` / `warnings` / `typing` / `F` / `Tensor`。唯一替身是 `torch.randint` 的 RNG spy：按调用顺序消费显式注入序列，记录`high` / `size` / `value`，并对 `high<=0` 复刻官方 RuntimeError 消息。

## 三层独立 oracle

探针用闭式推导 + `Fraction` 精确有理数重算 Bartlett 与量化格点；两层对拍在 822 个配置上对官方 `comp()` 与参考解 `solve()` 做类型敏感、float 逐位（含 `-0.0`）深比较；三层 oracle **不共享参考解算法**：`round` 走 `decimal.Decimal ROUND_HALF_EVEN`、Bartlett 走 `Fraction(2*min(i,n-i),n)`、噪声整形走显式 `[0]+err[:-1]` 列表移位、GPDF 别名走 dict 可变单元状态机，另用精确有理数验证每个输出都落在 `k/32768` 格点。

## 变异与等价留证

`mut870.py` 首轮 172 项（13 个锚点错误 + 22 个存活）。锚点错误用显式出现序号与多行块替换修正；22 个存活中 **10 个是判题盲区**，两轮硬化捕获：`_clean` 的nan/inf/列表分支、`round` 的 inf guard、三处默认参数、嵌套空、census 常量浅拷贝。另发现 `ba-half`（`n//2` 改 `(n-1)//2`）在 `n=98` / `196` 有 1 ulp 差异，是**第 11 个盲区而非真等价**，补大 n 常量后捕获。

最终 **171 项，剔除 11 个真等价，160 项有效全 CAUGHT**。11 个真等价由 `equiv870.py` 在 1214 配置 x 9 mode = 617507 次完整 `solve()` 输出精确比较（含异常类型与消息、float 逐位）证明零差异：dict 键顺序、`_clean` 在 solve 中从不作用于 dict 的死分支、两道空输入 guard、三角窗的 `x+0.0` 与 `1-|x-1| == 2-x` 恒等式、未变异分支的 rows/grid 等价、注释改动、同形约束下的shape 标签交换、以及未知密度本就落到 TPDF。**不删除任何变异**。

## 工程教训

- RNG spy 必须按**调用顺序**消费序列；按固定下标取值会在 `channel_size=0` / `time_size=0` 时错位。
- `torch.round` 是半偶舍入且保留 `-0.0`；`x < 0` 判不出负零，须用 `copysign`。
- PyTorch 窗函数的左右半常用不同浮点表达式，逐位对拍必须照抄分段，且大 n 才能暴露 `n//2` 类边界差异。
- 0 维张量视图上的 `+=` 会改写输入；一旦同一函数后续还要用 original，污染会跨步骤传播。
- 非有限值必须归一成稳定字符串，否则 fixtures 不是合法 Python 字面量。
