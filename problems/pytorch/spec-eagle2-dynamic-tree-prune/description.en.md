# EAGLE-2: Dynamic Re-ranking and Pruning of the Draft Tree

## Problem

The EAGLE family turns "guess the next token" into feature-level autoregression by the draft network and drafts a **static draft tree** that is verified in parallel. **EAGLE-2** (Li et al., 2024, [arXiv:2406.16858](https://arxiv.org/abs/2406.16858)) observes that a static tree is shaped by the draft network's per-level top-$k$ **beam**, independently of how likely each node actually is to be accepted, so verification budget is wasted on deep, useless branches. EAGLE-2 therefore works in two passes: draft the static tree, then use a second forward of the draft network to obtain each node's top-probability (a conditional acceptance rate), multiply along the root$\to$node path to get the node's **cumulative confidence** $V(v)$, and finally **dynamically re-rank and prune** — redistributing the scarce verification budget level by level by score, and flattening only the surviving nodes into one tree-attention call of the target model.

This problem implements exactly that re-ranking/pruning step (drafting, tree attention and real forwards are out of scope): given the parent/child structure of a static draft tree together with per-node conditional acceptance rates, compute the node set that is actually sent to the target model under EAGLE-2's level-wise greedy rule, and report the iteration's expected token yield.

## Rules

Inputs:
- `parent`: `long` tensor of shape $(M,)$; node 0 is the already-confirmed token (the root, `parent[0] = -1`), the others satisfy `parent[i] < i` (topological order).
- `acc`: float tensor of shape $(M,)$; `acc[i]` = estimated probability that node $i$'s token is accepted by the target, conditioned on $i$'s parent having been accepted (the second pass's top-probability). `acc[0]` is unused.
- `k`: verification budget = maximum number of **draft nodes** sent to the target this round (the root's slot is free and does not consume budget).
- `gmax`: maximum number of drafted levels (tree height cap).

Cumulative confidence: $V(\text{root})=1$ and
$$ V(v)=\prod_{u\in \text{path from root to }v,\,u\neq \text{root}} acc[u], $$
which is non-increasing along every root-to-leaf path.

Dynamic selection (level-wise greedy): start with $\mathcal F=\{\text{root}\}$, $\mathcal S=\varnothing$, $\text{used}=0$, $\text{level}=0$. While $\text{used}<k$, $\text{level}<gmax$ and $\mathcal F\ne\varnothing$:
1. Candidates $\mathcal C$ = all children of the nodes in $\mathcal F$ (collected in increasing node index); stop if $\mathcal C=\varnothing$.
2. Sort $\mathcal C$ ascending by $(-V(v),\,v)$ and take the first $\min(k-\text{used},\,|\mathcal C|)$ of them into $\mathcal S$, increasing $\text{used}$ accordingly.
3. Set $\mathcal F \leftarrow$ this batch of new nodes and $\text{level}\leftarrow \text{level}+1$.

Return `(sel, E)`:
- `sel`: 1-D `long` tensor with the indices of $\mathcal S$ in **ascending** order (the root is not included).
- `E`: float, the expected number of tokens emitted this iteration $=1+\sum_{v\in\mathcal S}V(v)$, where the $1$ is the token the target model always produces itself.

The budget is filled **level by level**: when a level offers fewer candidates than the remaining budget, all of them are kept and the leftover carries to the next level. This is *not* the same as taking a global top-$k$ over all nodes — on a sparse level the greedy rule spends budget across the whole level.

## Examples

```
parent = [-1, 0, 0, 1, 1, 2, 4]      # static draft tree (7 nodes, 0 is the root)
acc    = [ -, 0.75, 0.5, 0.875, 0.625, 0.5, 0.25]

V(1)=0.75  V(2)=0.5  V(3)=0.65625  V(4)=0.46875  V(5)=0.25  V(6)=0.1171875

k = 4, gmax = 3
  level 1: keep both {1,2}           used=2
  level 2: candidates {3,4,5} sorted by V -> take {3,4}   used=4  -> budget exhausted
  sel = [1, 2, 3, 4]   E = 1 + (0.75+0.5+0.65625+0.46875) = 3.375

k = 1, gmax = 3
  level 1: from {1,2} take the largest V, i.e. {1}   used=1  -> budget exhausted
  sel = [1]   E = 1.75

k = 4, gmax = 1 (only one level may be drafted)
  sel = [1, 2]   E = 2.25

k = 0 (no speculation, target only)
  sel = []   E = 1.0
```

## Constraints

- CPU float32; $2\le M\le 200$ (the degenerate $M=1$ input is also legal), $0\le acc[i]\le1$, $0\le k\le 64$, $0\le gmax\le 16$. The return is `(1-D long tensor, float)`: `sel` is compared with `torch.equal`, `E` with `torch.allclose(atol=1e-5, rtol=1e-5)`.
- `sel` is ancestor-closed (every selected node's parent, unless it is the root, is also selected), has length $\le \min(k, M-1)$, and is sorted without duplicates.
- **Degenerate properties**: with $k=0$ or $gmax=0$, `sel` is empty and `E=1.0`; with all `acc` equal to $1$, $V\equiv1$, each level is filled in index order and $E=1+|sel|$; for a chain (every node has exactly one child) `sel` is the first $\min(k, gmax, \text{chain length})$ nodes below the root.
- Unlike `spec-medusa-tree-verify` (175: typical acceptance over given candidate paths), `spec-sequoia-optimal-tree` (177: shape-search DP with no depth cap) and `spec-sequoia-sample-verify` (176: per-position multi-branch residual shrink), this problem **does not test acceptance or modify any distribution** — it only decides, from given conditional acceptance rates, where the verification budget should be spent.
