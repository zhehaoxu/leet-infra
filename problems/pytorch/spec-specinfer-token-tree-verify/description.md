# SpecInfer：token-tree 的逐层并行验证

## 题目描述

**SpecInfer**（Miao et al., 2023, [arXiv:2305.09781](https://arxiv.org/abs/2305.09781)，实现随 FlexFlow 发布：[flexflow/FlexFlow](https://github.com/flexflow/FlexFlow) 的 SpecInfer/inference 分支，即 flexflow-ai/SpecInfer）把推测解码的草稿从「一条序列」推广成「一棵 token tree」：draft 侧（小模型、多个 head、检索或 n-gram 都行）在同一步里给出多条分支，target 侧用 **tree attention** 把整棵树的所有分支节点放进一次并行前向，一次就算出树上每个位置的概率。验证沿**已接受前缀逐层**推进：第 $d$ 层只在第 $d-1$ 层活下来的节点的孩子里做接受判定，某一层全军覆没就停止扩张；最后取**最长存活路径**作为本轮被接受的 token，并在路径末端补一个 token（被拒位置用残差分布重采、树耗尽时用 target 分布给 bonus），使输出分布严格等于 target 分布。

本题给定一棵已经组装好的 token tree、树上每个位置的 draft 分布与 target 分布，以及逐节点的显式随机数，要求模拟这次 tree-attention 验证并输出被接受的 token 序列与存活路径的节点编号。

## 规则

入参：
- `parent`：形状 $(M,)$ 的 `long`，节点 0 是已确认 token（根，`parent[0] = -1$），其余满足 `parent[i] < i`（拓扑序）。
- `token`：形状 $(M,)$ 的 `long`，每个草稿节点提出的 token id（根的不用）。
- `qdist`：形状 $(M, V)$ 的 float，draft 在**该节点位置**上抽样用的分布，$q_v=$ `qdist[v][token[v]]`。
- `T`：形状 $(M, V)$ 的 float，target 经 tree attention 一次前向后**输出在节点 $v$** 上的分布，即 $v$ 的**孩子位置**上的分布：孩子 $c$ 的目标概率 $p_c=$ `T[parent[c]][token[c]]`。
- `u`：形状 $(M,)$ 的 float，逐节点接受硬币（`u[0]` 不用），$u\in[0,1)$。
- `u_final`：float，收尾 token 的均匀数，$\in[0,1)$。

逐层验证（记 $\mathcal A$ 为已接受节点集）：$\mathcal A=\{0\}$，$\mathcal F=\{0\}$；重复——
1. 候选 $\mathcal C$ = $\mathcal F$ 中各节点的全部孩子（按编号升序）；若 $\mathcal C=\varnothing$ 则树已耗尽，停止。
2. 对 $v\in\mathcal C$（父为 $\text{par}$）：$\text{ratio}=\dfrac{\texttt{T[par][token[v]]}}{\texttt{qdist[v][token[v]]}}$（分母为 0 时视 ratio 为 $+\infty$），当 $u_v<\min(1,\text{ratio})$ 时接受 $v$。
3. 若本层一个都没被接受，停止；否则把存活者并入 $\mathcal A$，令 $\mathcal F$ 为这批存活者。

取 $\mathcal A$ 中**最深**的节点 $v^\ast$（并列时取 token 序列字典序最小者，仍并列取编号最小者），根到 $v^\ast$ 的路径即存活路径。收尾 token：
- 若 $v^\ast$ 有被拒的孩子（本轮在某层覆没留下的那批），取编号最小的被拒孩子 $c$，用残差分布 $\mathrm{norm}(\max(\texttt{T}[v^\ast]-\texttt{qdist}[c],\,0))$（逐元素先取非负、再除以元素和）采样；
- 否则（$v^\ast$ 是叶子、树已耗尽）直接用 $\texttt{T}[v^\ast]$ 采样 bonus。

采样一律是 CDF 反查：返回**最小**的 $j$ 使 $\sum_{i\le j}\pi_i>$ `u`。

返回 `(out, nodes)` 两个一维 `long` 张量：`out` = 存活路径 token + 收尾 token（长度 $=$ `len(nodes)+1`，恒 $\ge1$）；`nodes` = 存活路径的节点索引（自根向 $v^\ast$ 升序，不含根、不含收尾 token）。

## 示例

```
parent = [-1, 0, 0, 1, 1, 3]         token = [0, 1, 2, 0, 3, 1]   (V=4)

T[0]=[0.1,0.6,0.2,0.1]  T[1]=[0.5,0.1,0.3,0.1]  T[2]=[0.2,0.2,0.3,0.3]
T[3]=[0.1,0.1,0.6,0.2]  T[4]=[0.25,0.25,0.25,0.25]  T[5]=[0.4,0.2,0.2,0.2]
q[1]=[0.1,0.3,0.3,0.3]  q[2]=[0.2,0.2,0.6,0.0]  q[3]=[0.4,0.2,0.2,0.2]
q[4]=[0.1,0.1,0.1,0.7]  q[5]=[0.3,0.3,0.3,0.1]
u = [-, 0.5, 0.9, 0.3, 0.5, 0.7]

layer 1: 候选 {1,2}
  v=1: p=T[0][1]=0.6, q=q[1][1]=0.3, ratio=2   -> 0.5<1      接受
  v=2: p=T[0][2]=0.2, q=q[2][2]=0.6, ratio=1/3 -> 0.9>=1/3   拒绝
layer 2: 候选 {3,4}（节点 2 的子树整体作废）
  v=3: p=T[1][0]=0.5, q=q[3][0]=0.4, ratio=1.25 -> 0.3<1     接受
  v=4: p=T[1][3]=0.1, q=q[4][3]=0.7, ratio=1/7  -> 0.5>=1/7  拒绝
layer 3: 候选 {5}
  v=5: p=T[3][1]=0.1, q=q[5][1]=0.3, ratio=1/3  -> 0.7>=1/3  拒绝 -> 本层全拒，停止
最深存活节点 v*=3，存活路径 = 节点 [1,3]，token = [1,0]
v*=3 有被拒孩子 5 -> 残差 = norm(max(T[3]-q[5],0)) = norm([0,0,0.5,0.1]) = [0,0,0.75,0.25]
u_final=0.8 -> CDF [0,0,0.75,1.0] 第一个 >0.8 的下标 = 3
out = [1, 0, 3]   nodes = [1, 3]
（同一输入 u_final=0.5 时 out = [1, 0, 2]）

链树全接受（bonus）：
parent=[-1,0,1]  token=[0,2,0]  u=[-,0.4,0.5]  u_final=0.3
T=[[0.1,0.1,0.7,0.1],[0.25,0.25,0.25,0.25],[0.05,0.15,0.6,0.2]]
q=[-, [0.2,0.2,0.3,0.3], [0.4,0.2,0.2,0.2]]
ratio1=0.7/0.3>1 接受；ratio2=0.25/0.4=0.625, 0.5<0.625 接受；第 3 层无候选 -> 树耗尽
v*=2 无孩子 -> bonus 从 T[2]：CDF [0.05,0.2,0.8,1.0] 第一个 >0.3 的下标 = 2
out = [2, 0, 2]   nodes = [1, 2]
```

## 约束

- CPU float32；$1\le M\le 200$（$M=1$ 表示树上没有任何草稿节点），$V\ge2$，`token` 取值在 $[0,V)$，`T`/`qdist` 每行和为 1，`u`、`u_final` 在 $[0,1)$。两个返回值都是 1 维 `long`，用 `torch.equal` 比较。
- `T` 按**父节点**索引、`qdist` 按**节点本身**索引，这是 tree-attention 的自然口径：$v$ 的输出是「$v$ 之后那个位置」的分布。
- 逐节点判定用严格 $u_v<\min(1,\text{ratio})$；**若某节点被拒，则必有 $\sum_a\max(T[v^\ast][a]-\texttt{qdist}[c][a],0)>0$**（拒绝蕴含 $p<q$，两分布不同，残差质量即全变差距离为正），故残差归一化不会遇到零除。
- 与 `inf-spec-decoding-rejection`（155，单条线性 draft 的 Leviathan/Chen 拒绝采样）相比，本题把 draft 换成**树**、把「首个拒绝处重采样」换成「最长存活路径 + 该路径末端的残差/bonus」；与 `spec-medusa-tree-verify`（175，只看 $P\ge\varepsilon$ 的 typical acceptance、无残差修正）相比，本题做的是**带接受率的拒绝采样**；与 `spec-sequoia-sample-verify`（176，单位置多分支反复收缩同一位置的残差）相比，本题的残差只在路径末端取用一次。
