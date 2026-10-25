# Solution Notes · Medusa Candidate-Tree Verification

## Spreading the head outputs into a tree

Each of Medusa's $K$ heads predicts the top-$c$ candidates for the future position $k$. Stacking them by "depth = position offset" yields a candidate tree: the root is the current position, and a node at depth $d$ represents "one concrete continuation of the first $d$ speculative tokens". The node set is exactly **all prefixes of all candidate paths**—so there is no need to build a tree explicitly; just split every path into `c[:1], c[:2], ..., c` and drop them into a `set` for deduplication.

## Typical acceptance propagates along the parent chain

The Medusa paper uses **typical acceptance** ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192) §3.2) rather than probability-ratio rejection sampling: a candidate token $x$ at position $i$ is accepted as long as $p_i(x)\ge\varepsilon$, with no draft distribution $q$ and no residual resampling. The key tree property is: **a path of depth $d$ is accepted if and only if its entire parent chain (every prefix level) is accepted**—because the token at layer $i$ is only actually verified by the target model when the preceding $i-1$ tokens hold. It therefore suffices to check `P[i-1][path[i-1]] >= eps` per layer and take `all` over the whole path.

## Take the longest, break ties lexicographically

Accepted nodes may come from several different branches; verification computes them all in parallel at once and finally emits the **longest** one (the most tokens per step). If two share the same depth, the **lexicographically smallest** sequence guarantees a unique, checkable result. With selected length $D$, the bonus token is sampled from `P[D]` (the position immediately after the $D$ emitted tokens)—exactly the extra token that tree attention earns over a linear draft.

## Differences from other speculative-decoding problems

| Method | Structure | Acceptance rule |
|---|---|---|
| Leviathan/Chen rejection sampling (155) | Single sequence | $\min(1,p/q)$ + residual resampling |
| Typical acceptance (this family, 178) | Single sequence | $p\ge\varepsilon$ |
| **Medusa (this problem)** | **Candidate tree** | Per-layer $p\ge\varepsilon$, take the longest |
| Sequoia (176/177) | General token tree | Multi-branch residual / positional acceptance-rate DP |

## Edge-case self-checks

- `check_example`: the depth-3 branch `[1,2,0]` beats the two depth-2 branches, bonus=`CDF(P[3],0.4)=1`.
- `check_tie_break_lexicographic`: `[1,2]` and `[1,3]` are both accepted at the same depth → take the lexicographically smaller one.
- `check_only_root`: raising eps so that no candidate is accepted → emit only the bonus (`D=0`, sampled from `P[0]`).
- `check_matches_oracle`: random candidate trees + random target distributions, checked against an independent pure-Python prefix-enumeration implementation.
