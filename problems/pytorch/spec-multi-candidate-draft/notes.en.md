# In Depth: Per-Position Multi-Candidate Batched Verification with Residual Sampling

## Source

- Single-candidate textbook criterion: Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (arXiv:2211.17192) and Chen et al. (arXiv:2302.01318) — one draft token verified per position, closing from \(\mathrm{norm}((p-q)_+)\) on rejection. The repo's `inf-spec-decoding-rejection` implements exactly this criterion.
- Multi-candidate origin: Miao et al., *SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference* (arXiv:2305.09781) — each level of the token tree holds parallel sibling nodes, equivalent to a "C candidates per position, first come first served" verification view. The repo's `spec-specinfer-token-tree-verify` and `spec-medusa-tree-verify` verify the whole tree; this problem flattens the tree into a **fixed-width \(C\)** per-position candidate batch — the minimal generalization between the two.

## Full derivation and index convention

**Index convention**: \(P\) has \(g+1\) rows (row \(j<g\) corresponds to position \(j\), row \(g\) is the bonus row); \(Q\) has only \(g\) rows; `cand` and `u` are \((g,C)\); `picks` always has length \(g\) (one entry per position), and `out` has length = number of accepted positions + 1.

**Why "first come first served" is self-consistent — the monotone-support lemma**: candidate \(i\) being rejected implies \(\mathrm{ratio}<1\), i.e. \(p[x_i]<q[x_i]\) (when the ratio reaches 1, \(u\le1\) always accepts, so rejection is impossible; the same holds for the must-accept case \(q[x_i]=0,p[x_i]>0\)). The residual update \((p-q)_+\) therefore zeroes its mass. A stronger invariant: once a token has zero mass in the residual, every later \((p'-q)_+\) update gives it \(0\) again (since \((0-q)_+=0\)). So **within a position the residual support only shrinks**, a rejected token never revives, and a repeated occurrence in `cand` can never be accepted afterwards — this is exactly why the first-come-first-served semantics is unambiguous.

**Benefit of multiple candidates (intuition reference, not a pointwise identity)**: let the single-candidate acceptance rate against the original \((P[j],Q[j])\) be \(\alpha=\sum_v\min(p_v,q_v)=1-\sum_v(p_v-q_v)_+\). If the \(C\) candidates were each verified independently against the original distributions, the probability of "accepting at least one" would be \(1-(1-\alpha)^C\). The sequential residual reuse of this problem is not pointwise equal to that expression (a rejected candidate moves residual mass and shifts later denominators), but the direction agrees: the more candidates and the closer they sit to the target support, the higher the chance the position yields a token; repeated candidates add almost nothing (one rejection retires the id permanently).

**Why the two update steps must not be swapped**: on rejection, first \(p\leftarrow\mathrm{norm}(\max(p-q,0))\), then (optionally) \(q[x]\leftarrow0,q\leftarrow\mathrm{norm}(q)\). The residual must be computed with the **pre-removal** \(q\) — the starter docstring states `no_replace` "only affects the denominators of later candidates within the same position, never the update of \(p\)". Removing \(q\) first would change the \((p-q)_+\) criterion and directly contradict the discriminative test case.

**The "strictly greater" CDF reverse lookup**: here `cdf(d,w)` returns the smallest \(k\) whose prefix sum is **strictly greater** than \(w\) — deliberately opposite to `inf-spec-decoding-rejection`, which uses "\(\ge u\)". The difference is visible on ties: on a uniform row with \(w=0.5\), "\(\ge\)" gives \(k=0\) while "strictly greater" gives \(k=1\). This is one of the discriminating points between the two problems. Fallback: if every prefix sum is \(\le w\) (floating-point tail error), return \(V-1\).

**Trial counting and truncation**: every attempted candidate counts one trial (rejected ones included); once position \(j\) exhausts, all candidates of positions \(j+1..g-1\) produce neither tokens nor trials. Hence \(0\le\text{trials}\le g\cdot C\), with equality \(=g\cdot C\) when everything is accepted.

**Empty inputs**: with \(g=0\) no verification happens at all (`Q` has shape \((0,V)\)) and it degenerates to plain decoding; \(C=0\) (with \(g>0\)) is equivalent to position 0 exhausting on the spot — `picks=[-1]`, `out` of length 1 (the residual is \(P[0]\) untouched, since no rejection ever happened), `trials=0`.

## Key points and common mistakes

1. **Order of the two updates**: residual first, \(q\)-removal second; swapping them is caught by the `no_replace` discriminative case.
2. **Normalization fallbacks**: if the residual sum is 0 keep \(p\) unchanged; if the sum is 0 after removing from \(q\), do not renormalize. Neither fallback may be omitted.
3. **Candidates with \(q[x]=0\)**: \(p[x]>0\) records ratio 1 (must accept); \(p[x]=0\) records ratio 0 (only \(u=0\) accepts). Never divide by zero.
4. **Acceptance on equality**: \(u=\text{ratio}\) accepts; a test case steps exactly on this line.
5. **Local state never crosses positions**: all modifications to \(p,q\) are discarded when moving to \(j+1\); the next position starts fresh from \(P[j+1],Q[j+1]\).
6. **The bonus row is used only when everything is accepted**: if any position exhausted, \(P[g]\) plays no part; the closing token comes from the current residual instead.
7. **Repeated candidates**: repeated occurrences of an un-rejected id count independent trials; one rejection retires it permanently to zero.
8. **Length semantics**: exhaustion happened ⇔ \(\text{len(out)}=\text{len(picks)}\) with the last `picks` entry −1; no exhaustion ⇔ \(\text{len(out)}=\text{len(picks)}+1\).

## Differences from neighboring problems

- **inf-spec-decoding-rejection**: the textbook criterion with \(C=1\); this problem generalizes it to \(C\ge1\). The two CDF lookups run in opposite directions (prefix sum \(\ge u\) there, strictly greater here), and the \(C=1\) discriminative case verifies the pointwise collapse.
- **spec-medusa-tree-verify / spec-specinfer-token-tree-verify**: those verify an entire token tree (tree attention/masks, best-path selection); this problem has no tree structure — a fixed-width batch of \(C\) side-by-side candidates per position, first come first served, i.e. the "per-level flattened" simplification of tree verification.
- **spec-typical-acceptance**: the acceptance rule is replaced by a typical-sampling-style criterion, unrelated to the \(p/q\) ratio.

## Cross-check convention and complexity

- The independent recompute uses a structurally different pure-python implementation (explicit prefix-sum loops + a reverse CDF lookup); `out`/`picks` are compared with `torch.equal` and `trials` with integer equality; 260 random parameter cases cover \(g\in\{0..4\}\), \(C\in\{0..4\}\), repeated candidates, zero entries in \(q\), \(u\) landing exactly on ratios, and other corners.
- The three discriminative criteria: \(C=1\) collapsing back to textbook rejection sampling (mutually corroborating `inf-spec-decoding-rejection`), acceptance on \(u\) exactly at the ratio, and `no_replace` touching only same-position later denominators.
- Complexity: time \(O(g\cdot C\cdot V)\) (one \(O(V)\) residual update plus one \(O(V)\) lookup per candidate), extra space \(O(V)\). With \(V\) the vocabulary size and \(g\cdot C\ll V\) in real systems, the bottleneck stays the same as single-candidate speculative decoding: one \((g+1)\)-position forward pass of the target.
