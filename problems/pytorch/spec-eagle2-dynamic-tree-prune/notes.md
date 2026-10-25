# 详解：EAGLE-2 动态草稿树重排与剪枝

## 出处
Li et al., *EAGLE-2: Finding Optimal Substructures for Context-dependent Distributions in Fast Speculative Decoding*（arXiv:2406.16858，官方实现 [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)）。论文的 Algorithm 1 分两步：先在静态树上跑一遍 draft 网络，拿到每个候选 token 的 top-probability 作为**条件接受率**；再自浅到深逐层展开候选、按「根到该节点路径上条件接受率的累乘」$V(v)=\prod_j c_j$ 重新排序，每层只保留 $\text{剩余预算}$ 个节点，其余连子树一起剪掉，最后把留下的节点展平送 target 做 tree-attention 验证。EAGLE-2 的核心论证是 Lemma 1：一次投机迭代吐出 token 的期望 $=1+\sum_{v\in\mathcal S}\prod_j c_j$，即「target 必得的那一个 + 逐节点累积接受率之和」——本题的 `E` 就是这个式子。

## 直觉：为什么静态树要重排
静态树是按 draft 网络的 **beam 形状**长出来的：每层各节点独立取自己的 top-$k$，于是「父节点本身就不可能被接受」的分支也照样占了验证名额。target 模型的验证预算（一次 tree-attention 前向能并行处理的 token 数）是硬约束，把名额分给 $V$ 大的节点才划算。EAGLE-2 的发现是：**同一个 draft 网络的第二遍前向已经顺带算出了这些 top-probability**，几乎零成本，所以「先草拟、再按累积置信度重排」是免费的午餐——这也是它相对 EAGLE-1 固定宽度树的主要增益来源。

## 公式与单调性
$V(\text{root})=1$、$V(v)=V(\text{parent}(v))\cdot acc[v]$，故沿任何根到叶路径 $V$ **单调不增**（$acc\le1$）。这条单调性有两个用处：① 剪枝是「连子树一起剪」——若 $v$ 没被保留，它的所有后代更不可能有价值；② 若改成「对全树按 $V$ 取全局 top-$m$」，得到的点集自动祖先封闭（父的 $V$ 不小于子的 $V$），无需额外修补。本题规定的是论文原式的**逐层贪心**：第 $d$ 层的候选 = 上一层被保留节点的全部孩子，按 $(-V,\text{编号})$ 排序后取 $\min(\text{剩余预算},\text{候选数})$ 个。

## 关键陷阱
1. **逐层贪心 ≠ 全局 top-k**。当某一层的候选数少于剩余预算时，贪心会把整层留下、把余量交给更深的层，于是「浅层里分数较低但仍在同一层」的节点会挤掉「深层里分数更高」的节点。例：第一层 $\{V{=}0.9,\;V{=}0.5\}$、第二层 $\{0.63,0.54\}$、预算 3 时，贪心选 $\{0.9,0.5,0.63\}$（和 2.03），全局 top-3 是 $\{0.9,0.63,0.54\}$（和 2.07）。判题按题面的逐层贪心，`check_sparse_level_greedy_not_global_topk` 就是钉死这一点的。
2. **父节点没留下，子节点再高分也不能要**：候选必须只从「上一层保留集合」展开，否则选出的点集不祖先封闭、tree attention 掩码无从构造。
3. **并列去歧义**：$V$ 相同（例如 $acc$ 全为 1，此时所有节点 $V\equiv1$）时按节点编号小的先取，否则结果不唯一；实现里统一用 $(-V, v)$ 作升序键。
4. **$E$ 里的 $1$ 不能漏**：target 那一遍前向无论有没有 draft 都会定出一个 token（EAGLE-2 Lemma 1 的「必得首 token」），漏掉 $+1$ 会让 $k=0$ 的退化输入吐出 $E=0$ 这种荒谬值。
5. 与 177 的 $F(\mathcal T)=\sum_v f(v)$ 形式相同，但**角色不同**：177 是拿这个式子做「树形 $\arg\max$」的 DP；本题的树形是给定的，只算「给定预算下保留哪些节点」，$E$ 只是被报告量而非被优化量。

## 与合集内其它题的分工
`spec-medusa-tree-verify`(175) 给定候选路径做 typical acceptance 的**接受判定**；`spec-sequoia-sample-verify`(176) 做单位置多分支的**残差收缩**；`spec-sequoia-optimal-tree`(177) 与 `spec-sequoia-depth-bounded-dp`(283) 是**离线树形 DP**；`spec-specinfer-token-tree-verify`(285) 做**在线逐层接受与回退**。本题处在「离线/在线之间的树重排」这一格：不掷随机、不改分布、不做接受，只决定「验证名额给谁」。

## 对拍口径
参考解用「当前层保留集合 frontier」迭代展开，$V$ 靠拓扑序一次遍历（`V[v]=V[parent[v]]*acc[v]`）递推，候选用 `sort` 切片。判题 `_oracle` 走三条不同路子：① $V$ 与所在层数由**逐节点沿 parent 链上溯**、再自顶向下逐元素乘 `acc` 得到（不复用 DP）；② 把全部节点按层数**分桶**，自浅到深扫桶、按「父是否已入选」过滤候选（不维护 frontier）；③ 取前若干名用 `heapq.nlargest` 而非排序切片，期望值用 `math.fsum` 另序求和。此外用闭式与不变量交叉验证：链状树退化成 $acc$ 的前缀积、$acc$ 全 $1$ 时 $E=1+|sel|$、$k=0/gmax=0$ 时 `sel` 空且 $E=1$、以及每次随机用例都断言祖先封闭与 $|sel|\le\min(k,M-1)$、层数 $\le gmax$。张量比较用 `torch.equal`（索引），期望值用 `allclose(atol=1e-5, rtol=1e-5)`。

## 复杂度
$O(M\log M)$（每层候选排序的总量受 $M$ 界定）；空间 $O(M)$。$M\le200$ 时毫秒级。
