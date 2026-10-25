# SpecInfer: Level-wise Parallel Verification of a Token Tree

## Problem

**SpecInfer** (Miao et al., 2023, [arXiv:2305.09781](https://arxiv.org/abs/2305.09781); the implementation ships with FlexFlow: [flexflow/FlexFlow](https://github.com/flexflow/FlexFlow), the SpecInfer/inference branch, i.e. flexflow-ai/SpecInfer) generalizes the draft of speculative decoding from "one sequence" to "a token tree": the drafting side (a small model, several heads, retrieval or n-grams) proposes several branches in the same step, and the target side uses **tree attention** to put every branch node of the tree into a single parallel forward, which produces the probability of every position on the tree at once. Verification then advances **level by level along the accepted prefix**: level $d$ only tests children of the nodes that survived level $d-1$, and expansion stops as soon as a level is wiped out entirely; the **longest surviving path** becomes the tokens accepted this round and one more token is appended at the end of that path (resampled from the residual distribution if the path ends in a rejection, drawn from the target distribution as a bonus if the tree was exhausted), so that the emitted distribution is exactly the target distribution.

Given an already assembled token tree, the draft and target distributions at every node of the tree, and explicit random numbers per node, simulate this tree-attention verification and output the accepted token sequence together with the node indices of the surviving path.

## Rules

Inputs:
- `parent`: `long` tensor of shape $(M,)$; node 0 is the confirmed token (the root, `parent[0] = -1`), the others satisfy `parent[i] < i` (topological order).
- `token`: `long` tensor of shape $(M,)$, the token id proposed by each draft node (the root's is unused).
- `qdist`: float tensor of shape $(M, V)$, the distribution the drafter sampled **at that node's position** from, $q_v=$ `qdist[v][token[v]]`.
- `T`: float tensor of shape $(M, V)$, the target distribution **output at node $v$** by the tree-attention forward, i.e. the distribution at the position of $v$'s **children**: a child $c$ has target probability $p_c=$ `T[parent[c]][token[c]]`.
- `u`: float tensor of shape $(M,)$, the per-node acceptance coins (`u[0]` unused), $u\in[0,1)$.
- `u_final`: float, the coin used for the closing token, in $[0,1)$.

Level-wise verification ($\mathcal A$ = set of accepted nodes): start with $\mathcal A=\{0\}$, $\mathcal F=\{0\}$ and repeat —
1. Candidates $\mathcal C$ = all children of the nodes in $\mathcal F$ (in increasing node index); if $\mathcal C=\varnothing$ the tree is exhausted, stop.
2. For $v\in\mathcal C$ with parent $\text{par}$: $\text{ratio}=\dfrac{\texttt{T[par][token[v]]}}{\texttt{qdist[v][token[v]]}}$ (if the denominator is 0 treat the ratio as $+\infty$); accept $v$ when $u_v<\min(1,\text{ratio})$.
3. If nothing at this level is accepted, stop; otherwise merge the survivors into $\mathcal A$ and set $\mathcal F$ to them.

Take the **deepest** node $v^\ast$ of $\mathcal A$ (ties: lexicographically smallest token sequence, then smallest node index); the root$\to v^\ast$ path is the surviving path. Closing token:
- if $v^\ast$ has a rejected child (the batch that wiped out that level), take the rejected child $c$ with the smallest index and sample from the residual $\mathrm{norm}(\max(\texttt{T}[v^\ast]-\texttt{qdist}[c],\,0))$ (element-wise non-negative part, then divide by its sum);
- otherwise ($v^\ast$ is a leaf and the tree was exhausted) sample directly from $\texttt{T}[v^\ast]$ as the bonus.

Sampling is always CDF inversion: return the **smallest** $j$ with $\sum_{i\le j}\pi_i>$ `u`.

Return `(out, nodes)`, both 1-D `long` tensors: `out` = the surviving path's tokens plus the closing token (length $=$ `len(nodes)+1`, always $\ge1$); `nodes` = the node indices of the surviving path (root-exclusive, in the order root$\to v^\ast$, without the closing token).

## Examples

```
parent = [-1, 0, 0, 1, 1, 3]         token = [0, 1, 2, 0, 3, 1]   (V=4)

T[0]=[0.1,0.6,0.2,0.1]  T[1]=[0.5,0.1,0.3,0.1]  T[2]=[0.2,0.2,0.3,0.3]
T[3]=[0.1,0.1,0.6,0.2]  T[4]=[0.25,0.25,0.25,0.25]  T[5]=[0.4,0.2,0.2,0.2]
q[1]=[0.1,0.3,0.3,0.3]  q[2]=[0.2,0.2,0.6,0.0]  q[3]=[0.4,0.2,0.2,0.2]
q[4]=[0.1,0.1,0.1,0.7]  q[5]=[0.3,0.3,0.3,0.1]
u = [-, 0.5, 0.9, 0.3, 0.5, 0.7]

layer 1: candidates {1,2}
  v=1: p=T[0][1]=0.6, q=q[1][1]=0.3, ratio=2   -> 0.5<1      accept
  v=2: p=T[0][2]=0.2, q=q[2][2]=0.6, ratio=1/3 -> 0.9>=1/3   reject
layer 2: candidates {3,4} (node 2's whole subtree becomes void)
  v=3: p=T[1][0]=0.5, q=q[3][0]=0.4, ratio=1.25 -> 0.3<1     accept
  v=4: p=T[1][3]=0.1, q=q[4][3]=0.7, ratio=1/7  -> 0.5>=1/7  reject
layer 3: candidate {5}
  v=5: p=T[3][1]=0.1, q=q[5][1]=0.3, ratio=1/3  -> 0.7>=1/3  reject -> level wiped out, stop
deepest surviving node v*=3, surviving path = nodes [1,3], tokens = [1,0]
v*=3 has the rejected child 5 -> residual = norm(max(T[3]-q[5],0)) = norm([0,0,0.5,0.1]) = [0,0,0.75,0.25]
u_final=0.8 -> first index of CDF [0,0,0.75,1.0] exceeding 0.8 = 3
out = [1, 0, 3]   nodes = [1, 3]
(same input, u_final=0.5 gives out = [1, 0, 2])

chain fully accepted (bonus):
parent=[-1,0,1]  token=[0,2,0]  u=[-,0.4,0.5]  u_final=0.3
T=[[0.1,0.1,0.7,0.1],[0.25,0.25,0.25,0.25],[0.05,0.15,0.6,0.2]]
q=[-, [0.2,0.2,0.3,0.3], [0.4,0.2,0.2,0.2]]
ratio1=0.7/0.3>1 accepted; ratio2=0.25/0.4=0.625, 0.5<0.625 accepted; no candidates at layer 3 -> tree exhausted
v*=2 has no child -> bonus from T[2]: first index of CDF [0.05,0.2,0.8,1.0] exceeding 0.3 = 2
out = [2, 0, 2]   nodes = [1, 2]
```

## Constraints

- CPU float32; $1\le M\le 200$ ($M=1$ means the tree carries no draft node at all), $V\ge2$, `token` values lie in $[0,V)$, every row of `T`/`qdist` sums to 1, and `u`, `u_final` lie in $[0,1)$. Both return values are 1-D `long` tensors compared with `torch.equal`.
- `T` is indexed by the **parent** while `qdist` is indexed by the node itself — that is the natural tree-attention convention: the output at $v$ is the distribution of the position *after* $v$.
- Every per-node test uses the strict $u_v<\min(1,\text{ratio})$. **A rejected node always leaves a positive residual mass**: rejection implies $p<q$, hence the two distributions differ and $\sum_a\max(T[v^\ast][a]-\texttt{qdist}[c][a],0)$, which equals twice the total-variation distance, is strictly positive, so the normalization never divides by zero.
- Compared with `inf-spec-decoding-rejection` (155: Leviathan/Chen rejection sampling on a single linear draft), this problem turns the draft into a **tree** and replaces "resample at the first rejection" by "longest surviving path + residual/bonus at its end"; compared with `spec-medusa-tree-verify` (175: typical acceptance, only $P\ge\varepsilon$, no residual correction), this problem does **ratio-based rejection sampling**; compared with `spec-sequoia-sample-verify` (176: several branches at one position with the residual repeatedly shrunk), the residual here is used exactly once, at the end of the path.
