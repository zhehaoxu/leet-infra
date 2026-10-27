# scatter_reduce 聚合写入（sum/prod/mean/amax/amin + include_self）

## 题目描述

`scatter_` 是"覆盖写"，`scatter_add_` 只会累加，而真实的聚合式写入——把多个贡献按 sum/prod/mean/amax/amin 合并进同一个目标位置、并决定是否并入原值——是 `Tensor.scatter_reduce_` 的语义。它在稀疏聚合、token 路由（MoE 的 unpermute）、把变长段的统计量写回定长张量时天天用到，而它有两个**不看源码就想不到的行为**：mean 走增量式（Welford）更新而非"总和除总数"，include_self=False 时被触及位置以**首个 scatter 值**为初值。本题要求逐点复刻这条 CPU 语义：给定 input、src、index、dim、reduce、include_self，返回写入后的新张量。CPU、全程确定性、无随机数调用、不修改 input。

## 规则

- 完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。
- **写入位置**：对 index 的每个坐标 $c=(i_0,\dots,i_{k-1})$，src 在同一坐标 $c$ 处的值 $v$ 写到 input 中"第 dim 维坐标换成 `index[c]`、其余坐标不变"的位置 pos。index 里可以出现重复坐标，同一 pos 会收到多个贡献。
- **到达顺序**：贡献按 **index 的行主序**依次并入（同一 pos 的贡献等价于按其 dim 维坐标升序），聚合是**串行**fold，不是并行归约——这决定了浮点舍入的可复现性。
- **include_self=True**：pos 以 input 原值为初值，再逐个并入：sum 累加、prod 累乘、amax/amin 逐个取极大/极小、mean 用
  \[n\leftarrow n+1,\qquad m\leftarrow m+\frac{v-m}{n}\quad(\text{初值 }m=\text{input 原值},\ n=1),\]
  即 **Welford 型增量更新**。
- **include_self=False**：pos **不并入** input 原值，从**首个 scatter 贡献**起算（sum/prod 从首个值开始累加/累乘；mean 的初值即首个贡献、$n=1$，之后沿用同一增量式；amax/amin 只在 scatter 值里取最值）。
- **未被任何 index 触及的位置**：两种 include_self 下都**恒等于 input 原值**（包括 mean + include_self=False，此时也不是 0）。
- mean 有两处必须照抄原生的行为，且都可观察：
  - **计数**：include_self=True 时 input 原值既是初值也占一个名额。base=0、贡献 $[1,2,3,4]$ → **2.0**；只对 scatter 值求均值会得 2.5。base=0、贡献 $[10^8,1,1,1]$ → **\(2\times10^7\)**，同一错误写法给 \(2.5\times10^7\)。
  - **串行舍入**：每一步都在 input 的 dtype 下更新，数量级悬殊时与"先求和再除"差异巨大——base=0、贡献 $[10^8,1,1,-10^8]$ 下增量式得 **0.0**，批量公式给 **0.4**。（判题数据量级温和，主要靠上一条与原生对拍来区分；这一条的实操含义是：累加必须在 input 的 dtype 下逐步做，不要先在 float64 里批量算完再转回。）
- 返回与 input **同形状、同 dtype** 的新张量；`solve` 不得就地修改 input。

## 示例

input \(=[10,20,30,40]\)，src \(=[1,2,3,4,5]\)，index \(=[0,1,1,3,3]\)（dim=0）。贡献：pos0←1；pos1←2,3；pos2 无贡献；pos3←4,5。

| reduce | include_self=True | include_self=False |
|---|---|---|
| sum | \([11,25,30,49]\) | \([1,5,30,9]\) |
| prod | \([10,120,30,800]\) | \([1,6,30,20]\) |
| mean | \([5.5,\,8.333333,\,30,\,16.333334]\) | \([1,\,2.5,\,30,\,4.5]\) |
| amax | \([10,20,30,40]\) | \([1,3,30,5]\) |
| amin | \([1,2,30,4]\) | \([1,2,30,4]\) |

- sum：$10{+}1$、$20{+}2{+}3$、$40{+}4{+}5$；include_self=False 时不带原值，故为 $1$、$5$、$9$。
- prod：$10{\times}1$、$20{\times}2{\times}3$、$40{\times}4{\times}5$；include_self=False 时 $2{\times}3{=}6$、$4{\times}5{=}20$。
- mean（inc=True）：pos1 由 20 起步 → $20+\frac{2-20}{2}=11$ → $11+\frac{3-11}{3}=8.3333$；pos3 由 40 起步 → $22$ → $16.3333$。mean（inc=False）：pos1 以首个贡献 2 为初值 → $2+\frac{3-2}{2}=2.5$。
- **amax 两支的对照**：inc=True 时 input 原值也参与取大，而本例每个 scatter 值都小于所在位置的原值（$1<10$、$2,3<20$、$4,5<40$），输出因此**等于 input 本身**；inc=False 时只在 scatter 值里取大 → \([1,3,30,5]\)。
- **amin 两支相同**是因为每处首个 scatter 值都小于原值，"是否并入原值"不再影响结果。
- 未触及的 pos2 在全部 10 个单元格里都是 **30**——这正是"include_self=False 也不清零"的证据。

## 约束

- reduce ∈ {`"sum"`, `"prod"`, `"mean"`, `"amax"`, `"amin"`}，**非法取值抛 ValueError**（这一处刻意与原生不同）；其余错误面与原生一致：index 出现负值或 $\ge$ input.shape[dim] → RuntimeError；维数不一致、形状不满足逐维 `<=` 约束、index 为浮点 dtype、src 与 input dtype 不同 → RuntimeError；dim 不在 $[-n, n)$ → IndexError
- 形状约束：input/src/index **同维数**；每一维 `index.shape[d] <= src.shape[d]`；除 dim 外每一维 `index.shape[d] <= input.shape[d]`；dim 维本身允许 index 比 input 宽（重复写同一位置正是本题重点）
- src 比 index 大时按**同坐标配对**取用前缀，多余的 src 元素不参与
- dim 支持负数索引；input 为空张量或 index 元素数为 0 时返回 input 的副本
- 对拍容差：float32 \(10^{-5}\)、float64 \(10^{-9}\)（含 mean 的增量式路径）；"未被触及"与"全覆盖单贡献"两类位置用 `torch.equal` 精确比对
