# Speculative Decoding: Candidate Tree Construction and Tree Attention Mask

## Problem

Methods such as Medusa and EAGLE obtain candidate tokens for several future positions in a single forward pass: head \(h\) provides the top-\(k\) tokens for draft position \(h\). Combining the heads gives a candidate tree, where a node at depth \(d\) is a path of length \(d\) whose cumulative logprob is the sum of the per-position logprobs.

What actually runs on the GPU is a **tree-shaped forward pass**: all candidate nodes are flattened into one sequence and computed at once. That requires three things to be delivered together:

1. **The node set is prefix-closed.** A child cannot be in the tree while its parent is missing—otherwise the mask points at a nonexistent ancestor.
2. **Each node sees only the confirmed prefix plus its own ancestor chain.** Sibling nodes must be mutually invisible, otherwise parallel branches contaminate each other's attention.
3. **Nodes at the same depth share a position_id.** This lets branches reuse the KV of the same position (EAGLE's key trick) and keeps RoPE relative positions consistent with real decoding.

The tree cannot be fully materialized (\(K^H\) paths explode quickly), so this problem uses **best-first expansion by cumulative logprob**: a max-heap pops the currently most likely path, which consumes one node of the budget, then pushes its children. Because candidate logprobs are all \(\le 0\), a child is never better than its parent, so the pop order equals "all paths sorted globally by score, take the first N"—and prefix closure comes for free.

## Task

Implement `solve(cand_tokens, cand_lps, budget, prefix_len)`:

- Input: `cand_tokens` (`(H,K)` int64), `cand_lps` (`(H,K)` float32, all \(\le 0\)); `budget` is the node limit \(N\ge 1\); `prefix_len` is the confirmed prefix length \(\ge 1\). Row \(h\) holds the candidates for draft position \(h\).
- Ordering key: cumulative logprob descending, ties broken by the **lexicographically smaller path** (comparing candidate indices position by position).
- Outputs:
  - `parent` (`(N,)` int64): parent index, `-1` for children of the root (a parent index is always smaller than its child's);
  - `node_tokens` (`(N,)` int64): the token of that node;
  - `depth` (`(N,)` int64): from 1 to \(H\);
  - `positions` (`(N,)` int64): `prefix_len + depth - 1`;
  - `mask` (`(N, prefix_len+N)` bool): the first `prefix_len` columns (confirmed prefix) are all `True`; among the last `N` columns only itself and its ancestors are `True`.
- Raise `ValueError` on any positive logprob, mismatched shapes, `budget < 1`, `prefix_len < 1`, or non-2-D input.

## Example

```
cand_tokens = [[7, 9], [3, 8]]
cand_lps    = [[-0.1, -0.5], [-0.2, -3.0]]
budget = 4, prefix_len = 3

Path pool (score = sum of the per-position logprobs):
  (0)  -> 7  : -0.1
  (1)  -> 9  : -0.5
  (0,0)-> 7,3: -0.3
  (0,1)-> 7,8: -3.1   pushed only after (0,0) is popped
  (1,0)-> 9,3: -0.7
  (1,1)-> 9,8: -3.5

First 4 pops by score: (0) -0.1, (0,0) -0.3, (1) -0.5, (1,0) -0.7

parent     = [-1, 0, -1, 2]
node_tokens= [ 7, 3,  9, 3]
depth      = [ 1, 2,  1, 2]
positions  = [ 3, 4,  3, 4]

mask (7 columns = 3 prefix + 4 nodes):
node0 (7)        [1 1 1 | 1 0 0 0]
node1 (7,3)      [1 1 1 | 1 1 0 0]
node2 (9)        [1 1 1 | 0 0 1 0]
node3 (9,3)      [1 1 1 | 0 0 1 1]
```

Nodes 1 and 3 share depth and thus `position` 4, yet node 1 sees only node 0 and node 3 sees only node 2—that is exactly tree attention with multiple branches at one position.

## Constraints

- **Push timing determines prefix closure**: a child is queued only after its parent is popped. Queueing every path up front and sorting may yield the same node set, but `parent` would then point at nodes that were never emitted.
- **The tie-break key is the whole path**, not the rank inside one head: with equal scores, `(0,1)` beats `(1,0)`.
- **Prefix columns and node columns must not be mixed**: the prefix holds already-confirmed real tokens visible to every draft node; node columns contain only "self + ancestors".
- **`positions` and `mask` are independent**: sharing a `position` across a depth does not imply mutual visibility.
- Because logprobs are non-positive, best-first pop order equals global order; that monotonicity is the correctness premise, so positive logprobs must be rejected rather than silently accepted.
