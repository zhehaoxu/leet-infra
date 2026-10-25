# 详解：pass@k 无偏估计量

## 出处
Chen, Tworek, Jun, Yuan, Pinto, Kaplan et al., *Evaluating Large Language Models Trained on Code*（arXiv:2107.03374）式 (1)：

\[ \text{pass@}k \;:=\; \mathbb{E}_{C\sim \text{Unif}(\mathcal C_k)}\Big[\,1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\,\Big] \]

其中 \(\mathcal C_k\) 是 \(n\) 个样本的所有 \(k\)-子集。论文同时指出：直接代 \(p=c/n\) 进 \(1-(1-p)^k\) 是**有偏**的，且偏差在 \(n\) 不大（论文用 \(n=200\)）时不可忽略；官方实现见 `human_eval` 包（`compute_pass_at_k`），`EvalPlus`、`BigCodeBH`、OpenCompass 都沿用这一估计量。

## 无偏性推导
设 \(X\) 为"随机取出的 \(k\) 个样本里通过的个数"，则 \(X\) 服从超几何分布：
\[ \Pr[X = j] = \frac{\binom{c}{j}\binom{n-c}{k-j}}{\binom{n}{k}}, \qquad j = \max(0, k-(n-c)) \dots \min(c, k) \]
于是
\[ \Pr[X \ge 1] = 1 - \Pr[X = 0] = 1 - \frac{\binom{c}{0}\binom{n-c}{k}}{\binom{n}{k}} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \]
即"一次都没中的子集"只有 \(\binom{n-c}{k}\) 个。若改用 \(1-(1-c/n)^k\)，等于把超几何换成二项（有放回、独立），相当于假设每个样本以概率 \(c/n\) 独立通过：
\[ \frac{\binom{n-c}{k}}{\binom{n}{k}} = \prod_{i=0}^{k-1}\frac{n-c-i}{n-i}, \qquad \frac{1}{\binom{n}{k}} = \prod_{i=0}^{k-1}\frac{n-i}{n} \]
即朴素版把「无放回」当成了「有放回」。两者的大小关系：
\[ \frac{\binom{n-c}{k}}{\binom{n}{k}} = \prod_{i=0}^{k-1}\frac{n-c-i}{n-i} \;\le\; \prod_{i=0}^{k-1}\frac{n-c}{n} = \Big(1-\frac{c}{n}\Big)^{k} \]
因为对 \(0<a\le b\) 有 \((a-i)/(b-i)\le a/b\)——分子分母同减 \(i\) 会把一个小于 1 的正分数拉得更低。左边更小时 \(1-\) 左边就更大，所以
\[ \text{pass@}k \;=\; 1-\frac{\binom{n-c}{k}}{\binom{n}{k}} \;\ge\; 1-\Big(1-\frac{c}{n}\Big)^{k}, \]
**朴素版系统性低估**，且 \(k\) 越大低估越厉害。数值核对 \(n=10, c=2, k=3\)：正确值 \(1-\tfrac{8}{10}\cdot\tfrac{7}{9}\cdot\tfrac{6}{8} = 1-\tfrac{7}{15} = 0.5333\)，朴素值 \(1-0.8^3 = 0.488\)。\(k=1\) 时两者都是 \(c/n\)——这就是无偏性的锚点。

## 核心与易错点
1. **pass@1 就是通过率 \(c/n\)**：把它写成"这批候选里有没有至少一个通过"（0/1 判定）只在 \(k=n\) 时才等价；\(k>1\) 时那是另一个量，也是本题最常见的错法。
2. **精确整数比值**：\(\binom{n-c}{k}\) 与 \(\binom{n}{k}\) 都是整数，Python 的 `/` 对两个大整数会走真除法并给出正确舍入的结果；但**先取 `float(comb)` 再相除**在 \(n\) 上百时会丢有效位（`comb(200,10) ≈ 2.2e16` 已越过 float64 的 2^53 精确整数区间）。同理 `math.gamma`/阶乘相除更糟。
3. **\(\binom{a}{b}\) 的 \(b>a\) 约定**：\(k > n-c\) 时分子为 0，pass@k = 1。这既是"取不完那么多错样本"的组合事实，也是判题里最容易漏的分支——写成负数或抛 `ValueError` 都错（`math.comb` 本身对该情形返回 0，正好可用）。
4. **不要跨题平均**：pass@k 是**逐题**估计量，多题聚合要"先按题算 pass@k 再对题取平均"（代码评测的标准 macro），本题只负责单题的数值。这与 `eval-token-accuracy` 的 micro/macro 讨论是一对镜像：**单题内部的平均对象是 \(k\)-子集，跨题的对象是题目**。
5. **单调性**：pass@k 对 \(k\) 不减、且 pass@\(n\) = \([c>0]\)。若实现出现非单调，多半是分子分母的 \(n-c\) 与 \(n\) 写反。
6. **\(c\) 是"唯一通过数"而不是"首次通过位置"**：HumanEval 里同一条测试可能对多个候选都通过；\(c\) 必须是计数，不是比率。
7. 数值上 \(\text{pass@}k \in [0,1]\)：\(c=0\) 时分子分母相等 ⇒ 0，绝不能因为浮点顺序得到 \(-10^{-16}\)。

## 与邻近指标的区别
- 与 `sampling-*`/`top-k-sampling` 等题：那些是**生成侧**（怎样造出 \(n\) 个候选），本题是**评测侧**（如何把这 \(n\) 个候选的二值结果汇总成指标）。
- 与 `eval-ndcg-at-k`：都带 "at k"，但 nDCG@k 里的 \(k\) 是**截断位置**（顺序有意义），pass@k 里的 \(k\) 是**子集大小**（顺序无意义、组合而非排列），二者数学对象不同。

## 对拍口径
参考解两步 `math.comb` 后一次真除法。判题 `_oracle` 有两路：`_oracle_bruteforce` 枚举全部 \(\binom{n}{k}\) 个 \(k\)-子集（`itertools.combinations`）数"至少含一个正确样本"的比例——纯定义法、\(n \le 7\) 全量；`_oracle_product` 走连乘 \(\prod_i (n-c-i)/(n-i)\)（既不用组合数也不用枚举）。另设断言：\(c=0\)、\(k=1\)、\(k=n\)、\(c=n\)、\(n-c<k\) 五枚恒等式，\(k\) 上单调不减，\(n=5,c=1,k=2\) 的精确有理值 0.4，以及"朴素 \(1-(1-c/n)^k\) 无偏值不同且偏低"的对照。浮点比较统一 `allclose(atol=1e-5, rtol=1e-5)`（float64 张量），0/1 恒等用 `==`。

## 复杂度
`math.comb` 走整数算法，约 \(O(k)\) 次大数乘法与一次真除法；枚举对拍是 \(O(\binom{n}{k})\)，因此只在小 \(n\) 上使用。

## 实践提醒
\(n\) 与 \(k\) 的取法决定可比性：HumanEval 用 \(n=200\)、\(k \in \{1,10,100\}\)；把同一模型的 pass@1 与 pass@100 混着报是最常见的图表错误。此外 pass@k 只回答"能不能做到"，不回答"稳不稳定"，要与 `eval-perplexity-logits` 这类内在指标一起看。
