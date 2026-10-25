# EAGLE-2：草稿树的动态重排与剪枝

## 题目描述

EAGLE 系列把「猜下一个 token」变成 draft 网络的特征级自回归，并一次性草拟出一棵**静态草稿树**去并行验证。**EAGLE-2**（Li et al., 2024, [arXiv:2406.16858](https://arxiv.org/abs/2406.16858)）观察到：静态树是按 draft 网络每层 top-$k$ **形状**定的，与各节点**真实被接受的可能性**无关，于是白白把验证预算花在深而没用的分支上。EAGLE-2 的做法是两遍：先草拟静态树，再用 draft 网络的第二遍前向拿到每个节点的 top-probability（条件接受率），沿根到节点的路径累乘得到该节点的**累积置信分数** $V(v)$，最后**动态重排/剪枝**——把有限的验证预算按分数逐层重新分配，只把选中的节点展平送进 target 做一次 tree-attention 验证。

本题只做「重排/剪枝并输出实际验证点集」这一步（草拟、tree attention、真实前向都不在题面里）：给定一棵静态草稿树的父子结构与逐节点条件接受率，按 EAGLE-2 的逐层贪心求出送入 target 的节点集合，并给出这一轮迭代的期望吐词。

## 规则

入参：
- `parent`：形状 $(M,)$ 的 `long`，节点 0 是已确认 token（根，`parent[0] = -1`），其余满足 `parent[i] < i`（拓扑序）。
- `acc`：形状 $(M,)$ 的 float，`acc[i]` = 在 $i$ 的父节点已被接受的前提下，节点 $i$ 的 token 被 target 接受的估计概率（EAGLE-2 第二遍前向的 top-probability）；`acc[0]` 不使用。
- `k`：验证预算 = 本轮送入 target 的**草稿节点数**上限（根的槽位免费，不占预算）。
- `gmax`：最多草拟的层数（树高上限）。

累积置信分数：$V(\text{root})=1$，
$$ V(v)=\prod_{u\in \text{root}\to v \text{ 路径},\,u\neq \text{root}} acc[u], $$
沿任意根到叶的路径单调不增。

动态选点（逐层贪心）：令 $\mathcal F=\{\text{root}\}$、$\mathcal S=\varnothing$、$\text{used}=0$、$\text{level}=0$。当 $\text{used}<k$、$\text{level}<gmax$、$\mathcal F\ne\varnothing$ 时反复：
1. 候选 $\mathcal C$ = $\mathcal F$ 中每个节点的全部孩子（按节点编号升序收集）；若 $\mathcal C=\varnothing$ 则停止。
2. 把 $\mathcal C$ 按 $(-V(v),\,v)$ 升序排序，取前 $\min(k-\text{used},\,|\mathcal C|)$ 个并入 $\mathcal S$，令 $\text{used}$ 相应增加。
3. $\mathcal F \leftarrow$ 本批新节点，$\text{level}\leftarrow \text{level}+1$。

返回 `(sel, E)`：
- `sel`：一维 `long` 张量，$\mathcal S$ 的节点索引**升序**排列（不含根）。
- `E`：float，本轮期望吐出的 token 数 $=1+\sum_{v\in\mathcal S}V(v)$（$1$ 是 target 自己必得的那个 token）。

预算是**逐层填满**的：某层候选数少于剩余预算时本层全留、余量交给下一层。这与「对所有节点取全局 top-$k$」并不等价——稀疏层上贪心会把预算摊给整层。

## 示例

```
parent = [-1, 0, 0, 1, 1, 2, 4]      # 静态草稿树（7 个节点，0 是根）
acc    = [ -, 0.75, 0.5, 0.875, 0.625, 0.5, 0.25]

V(1)=0.75  V(2)=0.5  V(3)=0.65625  V(4)=0.46875  V(5)=0.25  V(6)=0.1171875

k = 4, gmax = 3
  level 1: 候选 {1,2} 全留            used=2
  level 2: 候选 {3,4,5} 按 V 排序 -> 取 {3,4}   used=4  -> 预算用尽
  sel = [1, 2, 3, 4]   E = 1 + (0.75+0.5+0.65625+0.46875) = 3.375

k = 1, gmax = 3
  level 1: 候选 {1,2} 取 V 最大的 {1}  used=1  -> 预算用尽
  sel = [1]   E = 1.75

k = 4, gmax = 1（只许草拟一层）
  sel = [1, 2]   E = 2.25

k = 0（不投机，只跑 target）
  sel = []   E = 1.0
```

## 约束

- CPU float32；$2\le M\le 200$（$M=1$ 的退化输入也合法），$0\le acc[i]\le1$，$0\le k\le 64$，$0\le gmax\le 16$。返回 `(long 一维张量, float)`：`sel` 用 `torch.equal` 比较，`E` 用 `torch.allclose(atol=1e-5, rtol=1e-5)` 比较。
- `sel` 对祖先封闭（除根以外的父节点必在集合内），长度 $\le \min(k, M-1)$，且升序无重复。
- **退化性质**：$k=0$ 或 $gmax=0$ 时 `sel` 为空、`E=1.0`；`acc` 全 $1$ 时 $V\equiv1$，每层候选按编号先后被填满，$E=1+|sel|$；链状树（每个节点恰一个孩子）时 `sel` 就是从根往下的前 $\min(k, gmax, \text{链长})$ 个节点。
- 与 `spec-medusa-tree-verify`（175，给定候选路径只做 typical acceptance）、`spec-sequoia-optimal-tree`（177，无深度上限的 $\arg\max$ 树形 DP）、`spec-sequoia-sample-verify`（176，单位置多分支残差收缩）不同：本题**不做接受判定、不改分布**，只按给定的条件接受率做「验证预算该花在哪些节点上」的重排与剪枝。
