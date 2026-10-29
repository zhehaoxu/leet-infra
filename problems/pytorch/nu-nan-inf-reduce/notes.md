# 详解：跳过式归约的三条实现路线与各自的坑

## 出处

`torch.nanmean`/`nanmax` 是公开 API，但工程里手写这一类的场景很具体：给稀疏多卡聚合写一个 all-reduce 的 fallback（未落盘的 rank 用 NaN 占位）、给 causal/文档边界掩码写 attention 的分母（被掩掉的位置不能进 softmax 的分母，也不能把整行变成 NaN）、数据清洗脚本里统计「有效样本的均值」。面试常拿它做两个判断题：`nanmean` 的分母到底是什么、以及 NaN 与 inf 混在一起时谁是老大。

## 三条路线

1. **哨兵替换**：把 NaN 换成该归约的单位元（`sum` 换 0、`max` 换 -inf、`min` 换 +inf），再用原生归约。这条路最短，但 `mean` 必须另外用 `x.isnan().sum(dim)` 拿到真实分母，而全 NaN 段会把哨兵漏出来——`max` 会返回 -inf 而不是 NaN，所以要再用「有效个数 == 0」的掩码把结果改写成 NaN。
2. **先算 mask 再 `where`**：`valid = ~x.isnan()`，值取 `torch.where(valid, x, fill)`，分母取 `valid.sum(dim)`。与第 1 条等价，只是把 mask 显式留着，便于 `keepdim` 与 0-D 输出对齐形状。
3. **纯 Python 分段**：`movedim` 把归约轴搬到末位、`reshape(-1, n)` 切成段，逐段用内置 `sum/max/min` 归约。参考解用张量写法，oracle 就用这条，二者结构完全不同——这也是本题判题能同时抓住「分母用段长」和「inf 被当 NaN」两类错误的原因。

## 核心与易错点

- **`amax` 不接受 `initial=`**：本仓库固定的 torch 版本里 `Tensor.amax` 没有 `initial` 关键字，想用「-inf 起步」的写法会直接 TypeError；`torch.max`（两参版本）行为又不同。跨版本可移植的做法是自己做哨兵替换 + 掩码改写。
- **`amax` 会传播 NaN**：`torch.tensor([nan, 1.0]).amax()` 返回 NaN（与 Python 内建 `max` 依赖比较顺序不同，张量归约是有固定传播规则的）。所以「先 amax 再把 NaN 挑出来」比「指望 amax 忽略 NaN」可靠得多。
- **被归约轴长度为 0**：`amax` 在空段上抛 RuntimeError，必须显式给 `reduced_len == 0` 分支——本题规定 `sum` 给 0.0、其余给 NaN，也就是把空段视作 count == 0。
- **`inf` 的减法**：判题辅助函数里写 `abs(a - b) <= tol` 时，`inf` 对 `inf` 会算出 `inf - inf = nan` 从而永远失败。比较之前必须加 `a == b` 的短路，这是所有含 inf 的数值题的公共坑。
- **NaN 的比较语义**：Python list 里 `nan == nan` 为假，所以逐元素断言要 NaN 感知（两边同为 NaN 才算过），不能直接 `assert got.tolist() == want`。
- **`mean` 的分母**：`[nan, 1.0, nan]` 的 `nanmean` 是 1.0，用段长 3 会得到 0.333。这条在测试里是独立的一条断言。

## 与邻近题的区别

`reduce-axis` 只考轴语义（哪些轴被消掉、keepdim 的形状、广播回填），完全没有 NaN/inf 分支；`flash-attention-tile` 与 `nu-online-softmax-rescale` 用 -inf 做掩码，因此它们只关心 max 与 exp 求和、不需要"跳过"语义。本题的独有考点是把 NaN、±inf、空段三种情况在四种 mode 下的笛卡尔积定义清楚。

## 对拍口径与复杂度

oracle 是纯 Python 的分段过滤归约，良态随机数据（35% NaN 注入，形状 1-D/2-D/3-D、全部合法 dim 与 None、keepdim 两种）要求 1e-12 相对容差且 NaN 位置逐位对应；示例数据要求逐位相等。复杂度 \(O(N)\) 一次遍历，额外一张 bool mask（\(N\) 位）与一个计数张量。
