# In Depth: SpecInfer Token-Tree Parallel Verification

## Origin
Miao et al., *SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification* (arXiv:2305.09781, implementation shipped with FlexFlow: [flexflow/FlexFlow](https://github.com/flexflow/FlexFlow), the SpecInfer/inference branch, i.e. flexflow-ai/SpecInfer). The paper rests on two pillars: (1) the **token tree** — the drafter may be a small model, several heads, a retrieval store or n-grams, and what it produces is a tree rather than a sequence; (2) **tree attention** — all nodes of the tree are packed into one sequence fed to the target, but the attention mask lets every node attend only to its own ancestor chain, so a single forward computes the conditional probability of every position on the tree. Advancing verification **level by level** along the accepted prefix, voiding rejected branches, and finally taking the longest surviving path is the flow described in the paper's verification section; resampling one token from the residual $\mathrm{norm}(\max(P-Q,0))$ at a rejection is the standard distribution-preserving trick of Leviathan/Chen (arXiv:2211.17192 / 2302.01318).

## The indexing convention is the easiest thing to get wrong
Rows of `T` are indexed by the **parent** while rows of `qdist` are indexed by the node itself; this is not a typo but the semantics of tree attention: the logits the target produces at node $v$ describe the distribution of the position *after* $v$, i.e. the position where $v$'s children live. Hence
$$ p_c=\texttt{T}[\text{parent}[c]][\texttt{token}[c]],\qquad q_c=\texttt{qdist}[c][\texttt{token}[c]]. $$
Indexing `T` by the node itself (writing `T[c][token[c]]`) is mistake number one of this problem: it reads a row that should not exist for the first level and shifts every decision in the tree. On the bright side, the closing token also lands on $\texttt{T}[v^\ast]$: that row is the distribution of the position after $v^\ast$, which is exactly the token we still owe — a bonus when everything was accepted, the P-side of the residual when the path ends in a rejection.

## Derivation: why the output is still exactly the target distribution
For a single sequence (problem 155) correctness comes from per-position rejection sampling plus resampling from the residual at the rejection: $x\sim\mathrm{norm}(\max(P-Q,0))$ is the same as the conditional resampling distribution, so the marginal at every position stays $P$. On a tree several candidate branches hang at the same position, so the decisions must be **chained along paths**: node $v$ can only be accepted if its whole ancestor chain was accepted, and the surviving set is therefore naturally a connected, root-containing subtree. Implementing this level by level (level $d$ is expanded only from the survivors of level $d-1$) gives:
1. **the subtree of a rejected node is never tested** — the probabilities were computed for free by tree attention, but tree semantics say they must not influence the output;
2. **a level with no survivor stops the expansion** — deeper positions no longer have a legal prefix;
3. the output takes the **deepest** surviving node (ties: lexicographically smallest token sequence, then smallest index), which makes the "longest surviving path" unique;
4. the closing token is drawn **once**, at the end of the path: if $v^\ast$ has a rejected child use $\mathrm{norm}(\max(\texttt T[v^\ast]-\texttt{qdist}[c],0))$ with $c$ the smallest-index rejected child, otherwise (no children) use $\texttt T[v^\ast]$ as a bonus. Since a child of $v^\ast$ must exist-and-be-rejected whenever $v^\ast$ has children (otherwise $v^\ast$ would not be deepest), "has a child" and "has a rejected child" coincide here.

One provably well-defined property comes for free: **a rejection always leaves positive residual mass**. Rejection implies $p<q$ (otherwise $\min(1,p/q)=1$ and any $u<1$ accepts), hence $P\ne Q$ and $\sum_a\max(P_a-Q_a,0)=\mathrm{TV}(P,Q)>0$, so the normalization can never divide by zero — no special case needed.

## Boundaries inside the collection
- `inf-spec-decoding-rejection` (155): one linear draft, resample at the first rejection, drop everything after it — the special case of this problem when the tree degenerates into a single chain.
- `spec-medusa-tree-verify` (175): a tree plus **typical acceptance** (only $P\ge\varepsilon$; no $Q$, no residual). Picking the deepest surviving path looks similar but the acceptance criterion is entirely different.
- `spec-sequoia-sample-verify` (176): $k$ branches at **one and the same position**, and every rejection zeroes that token in $D$ and shrinks $R$ to $\mathrm{norm}(\max(R-D,0))$ to try again at the same position — the residual is consumed repeatedly; here the residual is used exactly once, and the tree's branches sit at **different depths**.
- `spec-eagle2-dynamic-tree-prune` (284): decides only "who gets a verification slot" and performs no acceptance test at all; here the tree is given and only the acceptance test is performed.

## Pitfall checklist
1. Collect each level's candidates in increasing node index and take the **smallest-index** rejected child, otherwise the $Q$ side of the residual is ambiguous.
2. The tie-break at equal depth compares **token sequences** lexicographically, not node indices (length first, then tokens, then index).
3. The root emits no token and is never tested (`u[0]`/`qdist[0]` unused), but `T[0]` *is* used: it is the distribution row of the first level's children.
4. Sampling is CDF inversion returning the **smallest** $j$ with cumulative sum $>$ `u` (strictly greater) with $u\in[0,1)$; mixing in the `>=` convention of 155 produces off-by-one errors exactly here.
5. With `M=1` (no draft node on the tree) the surviving path is empty and the output is exactly one bonus token — do not return an empty tensor.

## Verification approach
The reference solution advances a frontier level by level and shares the accepted prefix (each node is tested once). The judge's `_oracle` takes another road: it runs an independent linear rejection-sampling pass over **every** root-to-node path (ancestor decisions recomputed, no prefix sharing, no levels), which yields the set of nodes whose path is rejection-free; the terminal is chosen by the sort key $(-\text{path length},\ \text{token sequence},\ \text{index})$, and the residual and the CDF are recomputed with explicit index loops. The two agree because "a level fully rejected" implies "no legal prefix exists deeper". Four degenerate families cross-check the result: root-only trees ($M=1$), the first level fully rejected, $\texttt{qdist}=\texttt{T}$'s parent row (ratio identically 1, everything accepted, bonus branch), and one-level star trees plus same-depth lexicographic ties; the invariant `len(out) == len(nodes)+1` is asserted on every case. Index tensors are compared with `torch.equal`.

## Complexity
The reference solution is $O(M\cdot V)$ overall (one test per node plus one residual/CDF pass at the end); the `_oracle`, which refuses to share prefixes, is $O(M^2)$-ish, still far below the 30 s judge budget for $M\le200$.
