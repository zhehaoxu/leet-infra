# 详解：scatter_reduce 聚合写入的 CPU 语义

## 出处

- 直接对标的算子：`torch.Tensor.scatter_reduce_(dim, index, src, reduce, include_self=True)`（PyTorch 1.12 起进入稳定 API，reduce 取 `sum|prod|mean|amax|amin`）。本题**只对标 CPU 实现的实际数值行为**，不写文档式转述。
- Welford 增量均值：Welford (1962) / Chan-Lovell-Martin 的"可合并统计量"家族；CPU kernel 里 `mean` 走 `m += (v - m)/n` 的串行更新，而不是先求和再除。
- 应用背景：MoE 的 unpermute（把专家输出按路由计数聚合回 token 序）、`segment_reduce` 的散写形态、把变长序列统计量写回定长表。这些用法都要求"同一目标多个贡献"时的**确定性顺序**，正是本题的难点。

## 完整推导与下标约定

**下标与线性地址**：设 input 的形状为 $(S_0,\dots,S_{n-1})$，行主序步长 $\sigma_j=\prod_{k>j}S_k$。index 的扁平序号 $t$ 反解出坐标 $c=(\ell_0,\dots,\ell_{n-1})$（沿 $n-1$ 到 $0$ 做 `//`、`%` 即得），写入位置
\[m(t)=\sum_{j\ne d}\ell_j\sigma_j+\text{index}[t]\cdot\sigma_d .\]
**dim 维的坐标被 index 值替换、其余维沿用 index 自身的坐标**——这就是"index 与 src 按坐标配对、src 可以更大"的实现依据：多余 src 元素没有对应的 index 坐标，自然不参与。

**为什么贡献顺序是"dim 维坐标升序"**：$t$ 按行主序递增，$c_d$（dim 维坐标）在同一 pos 的贡献集合里就是递增的——因为同一 pos 意味着**其余坐标相同**且 index 值相同，只有 $c_d$ 在变，而行主序正是先变最右维。所以"行主序到达"与"按 dim 坐标升序"是同一件事的两种说法；参考实现用 `argsort(pos, stable=True)` 分组，稳定排序保证组内次序就是行主序。

**mean 有两处独立可观察的差异**。其一是**计数**：include_self=True 时初值 $m=\text{input 原值}$ 且 $n=1$，所以原值同时进分子与分母。判题第一组 base=0、贡献 $[1,2,3,4]$ → 原生 **2.0**，而"只对 scatter 值求均值"给 2.5；第二组 base=0、贡献 $[10^8,1,1,1]$ → 原生 **\(2\times10^7\)**，同一错误写法给 \(2.5\times10^7\)。其二是**串行舍入**：$m\leftarrow m+(v-m)/n$ 逐步舍入，与"先求和再除"在数量级悬殊时结果差异巨大。实测 base=0、贡献 $[10^8,1,1,-10^8]$：串行路径 $5{\times}10^7\to3.3333{\times}10^7\to2.5{\times}10^7\to0$（最后一步在 float32 下 $2.5\times10^7-2.5\times10^7$ 恰好相消）得 **0.0**，与原生逐位一致；而批量求和/计数给 **0.4**（精确算术下两者都应是 0.4）。也就是说：**光把公式写成 Welford 还不够，必须按行主序串行、在 input 的 dtype 下逐步舍入**——参考实现因此用标量张量逐个更新而不用向量化 `sum/count`。

**include_self=False 的种子语义**：被触及的 pos **以首个 scatter 值为初值**再并入其余贡献（mean 的 $n$ 也从 1 起算）。于是"是否并入原值"只影响**被触及**的位置，未触及的位置两种设置下都是原值。可判别的三处：prod（$5,7$ 与 $6,4$ 的对照——input 值 5/7 完全不进结果）、sum（$10{+}1{+}2$ 与 $1{+}2$）、mean（base $10^8$、贡献 $[1,1,1,1]$：inc=True 得 \(2\times10^7\)，inc=False 得 **1.0**，批量公式给 \(2.5\times10^7\)）。

**amax/amin 的"原值竞争"**：inc=True 时 $m$ 与 input 原值取极值，inc=False 时只在 scatter 值里取；示例里 amax 的 inc=True 输出恰好等于 input（所有 scatter 值都更小），而 amin 两支相同（首贡献已更小）——两行合起来把"竞争集"这个概念钉死，只写 `torch.maximum(out, src)` 的实现无法同时满足两支。

**空 index 与零元素**：`index.numel()==0` 时直接返回 input 的**副本**（`torch.equal` 断言逐位相同）。注意"副本"而非原对象——判题另有"不得修改 input"的快照断言，所以必须 clone。

**错误面为何刻意分裂**：`reduce` 非法抛 **ValueError**（Python 层的取值域错误），而 index 越界、维数不匹配、dtype 不匹配抛 **RuntimeError**、dim 越界抛 **IndexError**（与原生张量方法一致）。这是题面写死的约定，别为了"统一"把原生那几个改成 ValueError。

## 核心与易错点

1. **mean 的原值要占一个名额**：写成"只对 scatter 值求均值"在 base=0 的两组用例上分别差 0.5 与 \(5\times10^6\)；写成"批量求和后在 float64 里除"则会在数量级悬殊处与原生分道（见上文的 $[10^8,1,1,-10^8]$ 实测）。
2. **include_self=False 的种子是"首个贡献"**，不是 0（sum/prod 下 0 会把结果打成 0）也不是"跳过该位置"。
3. **未触及位置保持 input 原值**，两种 include_self 都一样——`mean/inc=False` 时给 0 是经典错误。
4. **贡献顺序是行主序**（等价于 dim 维坐标升序）：并行归约（如 `index_add_` + 事后除计数）在浮点下对不上。
5. **src 大于 index 时取同坐标前缀**，不是取前 `index.numel()` 个扁平元素（两者在 index 非连续/形状不等同时并不等价）。
6. **dim 维允许 index 比 input 宽**，其余维必须 `<=`；把约束写成"所有维都要 <= input"会误杀合法输入。
7. **dim 负索引**要先规范化再判范围（$[-n,n)$ 之外抛 IndexError）。
8. **dtype 一致性检查**：src 与 input dtype 不同抛 RuntimeError；index 必须是整型。
9. **不修改 input**：全程在 clone 上写；`out[p] = ...` 的 `p` 是**扁平**位置，写回后要 `reshape(input.shape)`。
10. **越界检查含负值**：负 index 在原生也是 RuntimeError（Python 的负索引语义**不适用**于 index 张量），必须在写入前统一判 $[0,\text{input.size}(d))$。

## 与邻近题的区别

- **vs `gather-scatter`**：那题写的是基础 `scatter_`（**覆盖**写，重复坐标时"最后到达者胜"、无聚合）与 `gather_`（读取），没有 reduce 参数、没有 include_self；`scatter_add_` 也只是 `reduce="sum"` 且恒含 self 的特例。本题的全部难点都在"重复坐标怎么合并"。
- **vs `distributed-reduce-scatter`**：那是**跨 rank** 的集合通信——各卡贡献一份同形状分片，聚合后分片散回；本题是单卡张量内部的"按 index 聚合写入"，没有通信、且目标位置可重复。二者名字相近，机制完全不同。
- **vs `searchsorted-bincount`（同批次）**：`bincount` 恰好是 `reduce="sum"`、src 恒为 1、include_self=False 的一维特例；本题是它在任意维数、五种聚合、两态 include_self 下的一般化。
- **vs `tensor-shape-indexing`**：那题考形状推断与索引语义（读侧），本题考带聚合的写侧地址计算。

## 对拍口径与复杂度

- **主 oracle 就是原生** `Tensor.scatter_reduce_`：6 seed × 5 reduce × 2 include_self × 3 组 dim{0,1,-1} 数据，外加 3-D（index 在 dim 维比 input 窄）与"src 比 index 大"两支；float32 \(10^{-5}\)、float64 \(10^{-9}\)。用原生做 oracle 之所以成立，是因为本题目标就是**逐点复刻**这条 CPU 语义（含舍入），而不是另立口径。
- **独立 python 循环 oracle** `_oracle_loop` 另覆盖 sum/amax/mean × 2 include_self × dim{0,1}：用 `itertools.product` 逐坐标走、用 dict 存 `(acc, n)` 状态，与解法"算扁平位置 + 稳定排序分组"的路径完全不同，用来抓 pos 计算错误（例如把 dim 维坐标忘了替换）。
- 判别性用例：Welford 两组极端（含 batch 公式反例）、prod 的首贡献语义、mean 的计数与未触及保持、全覆盖单贡献时五种 reduce 输出彼此 `torch.equal`（每位置只有一个贡献，聚合退化为赋值——这条性质同时校验了 include_self=False 的种子）、零值 input 下 prod 的两支、空 index、非法 reduce/越界 index/dim 越界三类错误。
- 示例的 10 个单元格全部实测回填，且大多可纸面复核（见题面示例的分步算式）。
- 复杂度：$T$ = index 元素数。扁平位置计算 $O(T)$，稳定排序 $O(T\log T)$，分组串行 fold $O(T)$；内存 $O(T)$ 加一个 input 副本。原生实现用原子操作/分段并行，本题为了**复现串行舍入**必须付出排序与串行 fold 的代价——这也是"复刻语义"类题目的通用代价模式。
