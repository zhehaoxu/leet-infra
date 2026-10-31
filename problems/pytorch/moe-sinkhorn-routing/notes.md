# 参考做法：交替标度、eps 的位置与双随机性的崩解

## 实现

1. `q = torch.softmax(logits / tau, dim=-1)`，一次成一个新的 float64 张量，天然不碰输入。
2. 每轮两条语句：`q = q / (q.sum(dim=0, keepdim=True) + eps)`（列归一化），紧接着 `q = q / (q.sum(dim=1, keepdim=True) + eps)`（行归一化）。两条都要新建张量，不要用 `q /=` 配合原地语义去改上一步的结果，也别把输入张量本身当工作区。
3. 循环满 `iters` 次；`iters=0` 时什么都不做。
4. `square=True` 就 `q = q * q`，然后再做一次行归一化（除数仍是 `sum + eps`）。平方必须在循环之后，否则锐化会被后续轮数重新抹平。
5. `col_load = q.sum(dim=0)`，取的是**返回矩阵**的列和，也就是平方之后的那一版。

三个常见错误：

- **交替方向写反**。先行后列也能把列和拉到 1，但极限不同：`T=1`、`iters=1` 时「先列后行」给出全 $0.25$ 的行（行约束精确、列约束只是逼近），「先行后列」给出全 $1$ 的行（列约束恰好满足、行和变成 $E$）。判题用单 token 用例专门区分这两种写法。
- **eps 加错位置**。`(q + eps) / (sum + 2*eps)` 会把一列全零抬成每行 $0.5$（$\varepsilon/(2\varepsilon)$），凭空给零负载专家分走质量；`q / (sum + eps)` 才保持 0。
- **拿 `clamp_min` 掩盖 0/0**。列和为 0 时 `clamp_min(1e-30)` 会给出一个「看起来归一化过」的矩阵，而契约要求如实的 NaN。

不用 `allclose(col_sum, 1)` 提前停是刻意的：提前停让轮数依赖浮点比较的顺序，同一份输入换归约顺序就可能少跑一轮，判题就没法给出逐位一致的期望值。固定轮数把这件事变成纯算术。

## 正确性检验

7 个 `check_*` 覆盖：手算示例（`iters=0`、`iters=1`、`square`）、288 组随机对拍（seed × iters × tau × square × eps 全网格）、双随机逼近的性质测（列误差单调下降、行和恒为 1、行/列置换等变、行内加常数不变、eps 进除数的系统性偏差）、死列的 NaN 与 eps 语义、退化形状（`T=1`、`E=1`、全零 logits、16 轮、非连续输入）、非法输入、最大形状。

oracle 是纯 Python 列表实现：逐列 `math.fsum`、逐行 `math.fsum`、标量除法自己按 IEEE 语义给出 `0/0 -> NaN`、`正/0 -> inf`，完全不借用 torch 的 `softmax`、`sum` 或广播，因此不会和参考解共享同一个归约顺序。

变异测试：把交替方向换成先行后列、把 eps 挪到被除数、平方后不再归一化、把平方挪到循环之前、少跑一轮、用 `clamp_min` 掩盖 NaN、把 `col_load` 换成行和——七个 mutant 全部被这套用例抓到。

## 范围与来源

Sinkhorn–Knopp（1952）定理说：元素全正、行列数相等的矩阵，可以靠交替的行标度与列标度收敛到双随机矩阵。本题把这一迭代当作 MoE 的路由均衡算子来考，并故意把定理的前提拆掉，看实现怎么处理失效的情形——`eps=0` 加一列全零就是「不存在有限标度使列和为 1」的直接体现，此时矩阵根本没有解。

在本仓库判题环境的 transformers 5.x 源码里能找到两处同形的实现：

- `models/superglue/modeling_superglue.py` 的 `log_sinkhorn_iterations`：`for _ in range(num_iterations)` 里交替更新行、列标度，写在对数域里防下溢，轮数固定、不做容差提前停。
- `models/deepseek_v4/modeling_deepseek_v4.py` 里多路残差（mHC）的 comb 权重：`comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)`，随后 `hc_sinkhorn_iters - 1` 轮行/列归一化，目的是把 comb 约束成双随机、防止残差混合放大范数。这属于 hyper-connections 而不是 MoE 路由，但它确认了「eps 加在除数上、固定轮数、先压列再压行」这一写法在真实模型里是常规做法。

Expert-Choice 路由（Shazeer 2022）用「每专家取 top-$k_c$ 个 token」的硬选择精确满足列约束，本合集已有 `moe-expert-choice-routing` 覆盖硬版本；本题考的是软版本——列约束只能逼近，所以必须回答「逼近到什么程度、什么时候崩」。

`square` 锐化步是题目自定的规则：我没有取得可靠的一手来源把它对应到某个具体论文的实现，因此不做归属声明。`iters` 的上界 16、`eps` 默认为 0、`col_load` 取返回矩阵列和，都是判题口径，不是任何框架的默认配置。
