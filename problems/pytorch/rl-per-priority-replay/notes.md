# 详解：优先经验回放的采样与 IS 权重

## 出处
Schaul, Quan, Antonoglou, Silver, *Prioritized Experience Replay*（arXiv:1511.05952）§3.1：比例采样 \(P(i)=p_i^\alpha/\sum_j p_j^\alpha\)；为纠正采样偏差，梯度乘 \(w_i=(N P(i))^{-\beta}\) 并按批内 \(\max_j w_j\) 归一（论文脚注与算法 1 里的 "divide by \(\max_j (N P(j))^{-\beta}\)"）；\(\beta\) 从初值线性退火到 1。工程实现见 OpenAI baselines `segment_tree.py::SampleProb/compute_weight`、rlkit `PrioritizedReplayBuffer.sample_n`。

## 索引约定（本题最容易写偏的一条）
把采样概率看成的 \([0,1)\) 的**区间划分**：第 \(i\) 条样本占 \([F_{i-1}, F_i)\)，其中 \(F_i=\sum_{j\le i}P(j)\)。落点取「前缀和**严格大于** \(u\)」的最小 \(i\)，等价于 `np.searchsorted(cdf, u, side="right")` / `torch.searchsorted(cdf, u, right=True)`。
- 写成 `side="left"`（即 \(F_i\ge u\) 就停）会把**边界点归进右端点闭合的那一格**：本例 \(P=[0,0.625,0,0.375]\) 的划分是「下标 1 占 \([0,0.625)\)、下标 3 占 \([0.625,1.0)\)」，\(u=0.625\) 按半开区间属于下标 3，而 `side="left"` 会返回 1。
- 浮点下 \(F_{N-1}\) 可能算成 `0.9999999999999999`，\(u\) 贴近 1 时会越界，必须 `clamp(max=N-1)`。

## IS 权重的三个口径
1. **归一是按批内 max，不是全池 max**。全池 max 不小于批内 max，用它归一会把整批权重整体压低、且随池子状态漂移，破坏题面「批内最大权重恒为 1」这条可验证性质。
2. **指数符号**：主流代码用 \((N P(i))^{-\beta}\)（正比于 \(1/P(i)\)，低概率样本拿到高权重，这才叫修正）；论文正文的 \(\bigl(\tfrac1N\tfrac1{P(i)}\bigr)^{\beta}\) 与之同值，但很多二手资料写成 \((N P(i))^{\beta}\)，符号反了。本题固定用 \((NP)^{-\beta}\)。
3. \(\beta=0\) 时 \(N P\) 的 0 次幂恒 1，权重退化为全 1；\(\alpha=0\) 时 \(P\equiv 1/N\)、\(N P\equiv 1\)，同样全 1——**两条独立通路都通向 \(w\equiv1\)**，这是最省事的自检。

## 退化与数值坑
- **全零优先级 + \(\alpha>0\)**：\(\sum p^\alpha=0\)，直接除会得到 NaN。题面规定退化成均匀分布，实现里要有这条显式分支。
- **\(0^0\)**：\(\alpha=0\) 时 PyTorch 与 Python 的 `pow` 都给 1，于是含零优先级也能均匀退化，不必特判——但别顺手加 `p.clamp(min=eps)`，那会把「\(p_i=0\) 永不被采」的正确行为改掉。
- **累加精度**：\(N=10^5\) 时用 float32 做 `cumsum`，尾部格子的概率误差可积累到 \(10^{-3}\) 量级，足以让边界附近的 \(u\) 选错格。全程 float64 累加、最后再降 float32 输出。
- **零概率格**：\(P(i)=0\) 的前缀和是平的，\(u\) 不可能严格落在它上面，因此正确实现里永远采不到它——这也可以反过来当作「我的落点逻辑写对了」的探针。

## 与其他题族的分工
- `to-*`（`to-gae`/`to-ppo-clipped-surrogate`/`to-grpo-advantage`…）负责**公式**：TD 误差、优势、损失。本题的 `priorities` 就是它们算完 \(|\delta|\) 之后的产物，作为入参喂进来，不再重推。
- `sampling-*`（top-k/top-p/温度）操作的是**词表上的 token 分布**；本题操作的是**样本级的索引与权重**，两者只在「把分布变成离散选择」这一点上形似。
- `rl-async-rollout-staleness` 管「样本什么时候到、有多旧」，本题管「样本被选中的概率与补偿权重」，同属 RL 数据流但维度正交。

## 复杂度
\(O(N + K\log N)\)：一次前缀和 + 每格二分；oracle 用 \(O(NK)\) 的线性扫描对拍，结构不同而结果必同。
