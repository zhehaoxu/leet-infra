# In Depth: Triplet Margin Loss with Online Hard Example Mining

## Source

- Triplet margin loss: FaceNet, Schroff et al., *FaceNet: A Unified Embedding for Face Recognition and Clustering* (CVPR 2015, arXiv:1503.03832), which introduces batch_all / batch_hard / semihard mining and the "pick one hard negative, train online" scheme.
- The systematic case for batch-hard: Hermans, Baeza-Yates, Leibe, *In Defense of the Triplet Loss for Person Re-Identification* (arXiv:1703.07737), showing batch_hard is equivalent in expectation to random triplets but far lower in variance.
- `swap` is a PyTorch extension flag with no paper counterpart: this problem does **not** take the docstring wording on faith. The formula was fixed by measuring `F.triplet_margin_loss` triplet by triplet (p∈{1,2} × 16 random triplets × both swap states + one discriminating triplet), yielding \(\operatorname{relu}(\text{margin}+d_{ap}-\min(d_{an},d_{pn}))\).
- Nothing here involves softmax contrastive losses (`mm-clip-infonce` and friends): those are "one positive, many negatives, softmax over logits"; this is an explicit triplet + hinge.

## Full derivation and index convention

**Index convention**: anchor \(a\), positive \(p_a\) and negative \(n_a\) are all **indices inside the same batch**; \(d_{ap}=D_{a,p_a}\), \(d_{an}=D_{a,n_a}\), \(d_{pn}=D_{p_a,n_a}\). The distance matrix \(D\in\mathbb R^{N\times N}\) is computed once and reused.

**Monotonicity of \(\ell=\operatorname{relu}(m+d_{ap}-d_{an})\) is the root of every mining strategy**: \(\ell\) is non-decreasing in \(d_{ap}\) and non-increasing in \(d_{an}\). Hence for a fixed anchor,
\[\max_{p\in\mathcal P_a,\,n\in\mathcal N_a}\ \ell(a,p,n)=\operatorname{relu}(m+\max_p d_{ap}-\min_n d_{an}),\]
so **farthest positive + nearest negative** attains the maximum simultaneously — which is exactly batch_hard: without enumerating triplets, two argmax/argmin calls give the anchor's loss supremum at \(O(N)\) cost.

**The semihard open interval is precisely "loss strictly between 0 and margin"**:
\[d_{ap}<d_{an}<d_{ap}+m\iff 0<m+d_{ap}-d_{an}<m\iff 0<\ell<m .\]
Left of the band (\(d_{an}\le d_{ap}\)) are the "extremely hard negatives" batch_hard would pick — under noisy labels these are often mislabels (in faces, an extreme pose of the same person), and pushing them away can hurt convergence; right of it (\(d_{an}\ge d_{ap}+m\)) the loss is identically 0 and contributes no gradient. So semihard takes the nearest member *inside* the band; when the band is empty, **this problem stipulates a fallback to the globally hardest negative** (FaceNet leaves this branch unspecified; it is the statement of the task and must be followed to match the judge).

**The two strategies coincide at margin=0**: the interval \((d_{ap},d_{ap})\) is always empty, so semihard takes the fallback every step and produces exactly batch_hard's per-anchor picks. This is readable straight off the formulas (and the margin=0 test leans on it).

**swap's geometry and the "takes the larger one" equivalence**: rewrite \(-\min(d_{an},d_{pn})\) as \(\max(-d_{an},-d_{pn})\) and use relu's monotonicity:
\[\operatorname{relu}(m+d_{ap}-\min(d_{an},d_{pn}))=\max\big(\underbrace{\operatorname{relu}(m+d_{ap}-d_{an})}_{\text{original triplet}},\ \underbrace{\operatorname{relu}(m+d_{ap}-d_{pn})}_{\text{triplet after swapping }a\leftrightarrow p}\big).\]
The second term is exactly the loss of the anchor/positive-swapped triplet (new anchor \(p\): \(d_{ap}'=d_{pa}=d_{ap}\), \(d_{an}'=d_{pn}\)). So swap is **always ≥ non-swap**: it demands that *both* the anchor and its positive sit far from the negative, and either one being hugged is a violation. The discriminating case \(a=(0,0),p=(3,0),n=(3,1)\): \(d_{ap}=3\), \(d_{an}=\sqrt{10}=3.1623\), \(d_{pn}=1\) → non-swap gives \(\operatorname{relu}(1+3-3.1623)=0.8377\) (nearly satisfied), swap gives \(\operatorname{relu}(1+3-1)=3\) (the positive is glued to the negative). Applying the min to the \(d_{ap}\) side instead yields 0 — a distance of 3 from the measured value, pinned by a `>1.0` assertion in the test suite.

**Ties take the smallest index — implementation**: mask non-candidates with \(-\infty\) (argmax side) or \(+\infty\) (argmin side); torch's argmax/argmin return the **first** extremal position on ties, i.e. the smallest index. The pure-Python oracle achieves the same semantics with a left-to-right scan plus strict comparison, and the judge supplies a `tie_last=True` oracle variant that computes the *wrong* tie-break on purpose — three cases (negative tie, positive tie, tie inside the semihard interval) each arrange distinct \(d_{pn}\) values so the swap branch also exposes the difference.

**Skipping and the denominator**: when the positive pool is empty (that class has only the anchor) or the negative pool is empty (all labels equal), the anchor contributes **no term and does not enter the denominator** — the mean divides by the number of *valid* anchors, not by N. The margin=0 case \(x=[[0],[10],[2]],\ y=[0,0,1]\): a0 → \(\operatorname{relu}(0+10-2)=8\), a1 → \(\operatorname{relu}(0+10-8)=2\), a2's positive pool is empty so it is skipped → mean \((8+2)/2=5\); dividing by 3 would give 3.33.

## Key points and common mistakes

1. **swap's min is on the negative side**: `min(d_an, d_pn)`, not `min(d_ap, d_pn)`; the reversed direction can output 0 instead of 3.
2. **The semihard interval is open**: neither \(d_{an}=d_{ap}\) nor \(d_{an}=d_{ap}+m\) qualifies; inside the band take the **smallest** distance (not the largest).
3. **Empty band must fall back** to the globally hardest negative (the batch_hard pick), not "skip the anchor".
4. **Farthest positive, nearest negative**: both directions mean "hardest"; flipping either gives batch_easy and the judge turns red immediately.
5. **The denominator counts valid anchors**; N=0 / all-same-class / all-distinct / N=1 all return 0.
6. **Compute the p-norm by definition**: \(\sum|\Delta|^p\) then raise to \(1/p\); this path is exact for \(p=1\), while an Euclidean-only operator silently miscomputes Manhattan distance.
7. **The anchor is never its own positive**: the positive pool excludes \(j=a\) explicitly; the negative pool excludes it through "different label", which only holds because labels are long integers here.
8. **Return a 0-dim scalar tensor**: `torch.stack(losses).mean()`, never a Python float or a 1-D tensor; with no valid anchor use `zeros(())` in the input dtype.
9. **Error surface**: any strategy other than batch_hard/semihard (including "random" and "batch_all") raises ValueError.

## Differences from neighboring problems

- **vs `embedding-margin-losses` (same batch)**: `nn.HingeEmbeddingLoss`/`nn.CosineEmbeddingLoss` are **pairwise** losses (given a pair plus a 1/−1 target) — no mining, no anchor concept; this problem is a **triplet** loss where the mining rule *is* the task.
- **vs `mm-clip-infonce` / contrastive losses in general**: InfoNCE is "1 positive, K negatives, softmax over logits" with every negative in the denominator; triplet picks a **single** negative for the hinge. Their notions of "hard" differ too: InfoNCE weights hard negatives automatically through the softmax, triplets need an explicit mining rule.
- **vs `euclidean-distance` / `pairwise-distance` problems**: those compute distances only; here the distance matrix is an intermediate, and the judged content is the mining choice and the hinge.
- **vs the autograd collection (todo #65)**: this returns a scalar loss value, not \(\partial\ell/\partial x\).

## Cross-check convention and complexity

- The oracle `_oracle_solve` is a **double Python loop**: per-anchor pos/neg lists, per-dimension `abs(u-v)**p` accumulation then `**(1/p)`, explicit comparisons to pick extrema — a completely different implementation path from the solution's tensor matrix + argmax/argmin. A `tie_last` switch lets the same oracle also produce the "wrong tie-break" control group.
- Native-library cross-checking is used at exactly one point — the **swap formula**: `F.triplet_margin_loss` over 4 configurations × 16 triplets = 64 per-triplet values, plus an end-to-end three-sample batch. Mining, skipping and the fallback have no counterpart in `F`, so this "cross-check only where it is defined" split avoids substituting the library for the task's semantics.
- Discriminating cases: semihard trigger vs fallback (same batch with `sh < bh`, and `sh == bh` when the band is empty), three tie-break probes each paired with a "wrong index must differ" assertion, margin=0 on both the non-zero and all-zero sides, four degenerate inputs, and the illegal-strategy error.
- Integer-grid random data (`randint(0,5)`) avoids spurious near-ties from floating point; two continuous `randn` sets cover the float path. Tolerance \(10^{-5}\); example constants backfilled by running the reference (hand-checkable: the three distances \(\sqrt8,\sqrt{13},\sqrt5\) produce 0.7928932309 and 0.5986121893).
- Complexity: the distance matrix costs \(O(N^2D)\) time and \(O(N^2)\) memory; mining is \(O(N^2)\) (two masked arg calls per anchor). batch_all enumerates \(O(N^3)\) triplets — reducing that to \(O(N^2)\) is precisely why these two strategies exist.
