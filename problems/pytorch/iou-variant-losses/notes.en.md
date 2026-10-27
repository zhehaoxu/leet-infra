# In Depth: Detection Box IoU Variant Losses

## Source

- GIoU: Rezatofighi et al., *Generalized Intersection over Union* (CVPR 2019, arXiv:1902.09630). The paper patches IoU's vanishing gradient for non-overlapping pairs with the "empty fraction" of the enclosing rectangle and proves GIoU ∈ \([-1,1]\).
- DIoU / CIoU: Zheng et al., *Distance-IoU Loss* (AAAI 2020, arXiv:1911.08287). DIoU replaces overlap-driven-only with "overlap + center distance"; CIoU adds the aspect-ratio term \(v\) and the trade-off coefficient \(\alpha\). This problem uses the paper's form (\(\alpha=v/((1-\text{IoU})+v)\)), not later NWD/EIoU offshoots.
- The base IoU convention is inherited from this repo's `iou-compute` (intersection w/h clamped at 0, IoU = 0 when the union is 0); all three losses are \(1-\)similarity.
- Forward pass only: computing \(\partial L/\partial(x_1,y_1,x_2,y_2)\) given an upstream gradient belongs to the autograd collection (todo #65).

## Full derivation and index convention

**Index convention**: row \(i\) is pair \(i\); \(p=(x_1^p,y_1^p,x_2^p,y_2^p)\) is the predicted box and \(t\) the target; \(w_p=x_2^p-x_1^p\), \(h_p=y_2^p-y_1^p\), \(A_p=w_ph_p\) (likewise for \(t\)); the closure's width/height are \(w_C,h_C\). Every pair is independent — there is no cross-pair reduction except the final mean.

**union ≤ C area always**: the closure spans both boxes' full projections in x and in y, so it contains both boxes entirely; hence \(A_p+A_t\) is at most "the closure's area plus the intersection counted twice", i.e. \(A_p+A_t\le C\text{ area}+\text{inter}\), which rearranges to \(C\text{ area}-\text{union}\ge0\). Equality holds when one box "squarely abuts" the other (the union tiles the closure exactly — e.g. two boxes of equal width stacked vertically), and there GIoU coincides with IoU. So the GIoU penalty is non-negative, \(\text{GIoU}\le\text{IoU}\), and for disjoint pairs (\(\text{inter}=0\)):
\[\text{GIoU}=-\frac{C\text{ area}-A_p-A_t}{C\text{ area}}\in[-1,0),\qquad \text{loss}_{giou}\in(1,2].\]
That is exactly example pair 0's 1.0794 > 1: the boxes share only a \(1\times1\) corner while the \(3\times3\) closure still has 2/9 of its area uncovered. **GIoU reaches \(-1\)** in the limit where the empty part fills the closure, which is why the interval \([-1,1]\) has that lower bound.

**The DIoU penalty is bounded**: both centers lie inside the closure, so \(|\Delta c_x|\le w_C\), \(|\Delta c_y|\le h_C\) and \(\rho^2\le w_C^2+h_C^2=c^2\); the penalty ∈ \([0,1]\) and \(\text{DIoU}\in[-1,1]\). The center term is \(\big(\tfrac{x_1^p+x_2^p}{2}-\tfrac{x_1^t+x_2^t}{2}\big)^2\); the reference writes it as \(((x_1^p+x_2^p-x_1^t-x_2^t)\cdot0.5)^2\), which is the same number with one fewer tensor.

**Two degeneration points of CIoU**:
1. \(v=0\iff\) the two boxes have the same aspect ratio (\(\arctan\) is injective, so \(w_t/h_t=w_p/h_p\) suffices; squares are only the probe's special case). Then \(\alpha v=0\) and CIoU collapses to DIoU — the test asserts loss_ciou == loss_diou pairwise for an all-square batch.
2. \(\text{IoU}=1\iff\) the boxes coincide \(\Rightarrow v=0\Rightarrow(1-\text{IoU})+v=0\). **This branch really is reached** (the test suite feeds `pred == target`); a bare `v/den` yields 0/0 = NaN, so `alpha` must be explicitly guarded to 0. Otherwise the three losses turn NaN precisely at a *perfect* prediction — the easiest way to fail this problem.

**Division of labour among the zero guards**: union, C area and \(c^2\) are strictly positive under the promised valid boxes, so their guards only matter for degenerate boxes; \((1-\text{IoU})+v\) is reachable as shown above. The `clamp_min(1e-7)` before the aspect divisions likewise guards a zero-height box: \(\arctan(\infty)=\pi/2\) is still finite, but Inf leaking into the difference would contaminate \(v\) — after the clamp, \(w/h\) is at most \(10^7\) and everything stays well behaved.

**Vectorized finite-value pattern**: the reference factors "divide only where the mask is True, else 0" into a helper that first replaces the denominator with 1 outside the mask and then uses `torch.where`. The reason is that torch computes `a/b` for **the whole tensor before** selecting, so a zero denominator produces NaN/Inf eagerly; `torch.where` selects values but does not prevent a branch from being evaluated, so the denominator must be sanitized beforehand rather than patched afterwards.

## Key points and common mistakes

1. **Guard \(\alpha\) at coincident boxes** — degeneration point 2 is the only division by zero reachable with legal input.
2. **Fixed order** `(giou, diou, ciou)`, and all three are \(1-\)similarity, not similarity; returning similarity blows the `pred == target` zero probe.
3. **\(c^2\) is the squared diagonal** \(w_C^2+h_C^2\) — not \(2(w_C+h_C)\), not the closure area; getting this wrong destroys the \(\rho^2/c^2\le1\) bound.
4. **GIoU subtracts (closure area − union), not (closure area − inter)**; DIoU/CIoU reuse the same closure's diagonal — do not fit a separate bounding circle.
5. **Use \(\rho^2\) (squared), not \(\rho\)**: the paper's ratio is squared; taking a root changes both the numbers and the dimensional consistency with \(c^2\).
6. **N=0 with mean** returns three zero scalars, not NaN (`torch.mean` of an empty tensor is NaN, so short-circuit explicitly).
7. **The closure's `clamp_min(0)`** is unreachable for legal boxes, but if a `min`/`max` is swapped (e.g. using `min` for \(C_{x_2}\)) the width goes negative and the clamp is the last line of defence.
8. **Error surface**: any reduction other than mean/none (including "sum") must raise ValueError; the return must be a 3-tuple, 0-dim for mean and (N,) for none.

## Differences from neighboring problems

- **vs `iou-compute`**: that problem produces a scalar IoU (two coordinate conventions) and no penalty term; here the closure, center distance and aspect ratio are all computed at once. The base IoU is reused verbatim, so the intersection/union part can be read side by side.
- **vs `nms`**: NMS uses IoU for **hard deletion** (keep the top score, drop boxes over the threshold) — an inference-time selection operator. IoU variant losses are a **continuous regression target** at training time; they act on different objects.
- **vs `soft-nms` (same batch)**: soft-nms uses IoU to decay *confidence* (the score side), this problem uses it as a *coordinate* regression loss. Same IoU kernel, one edits scores, the other edits boxes.
- **vs classification losses such as `bce-with-logits` / `focal-loss`**: those are classification/confidence losses; this is the localization loss. A real detection head sums both.
- **vs the autograd collection (todo #65)**: that batch computes \(\partial\text{loss}/\partial\text{box}\); this one stops at the forward value.

## Cross-check convention and complexity

- The oracle `_oracle_pair` recomputes each pair with **pure Python scalars**: builtin `max/min`, `math.atan`, `math.pi` — a completely different implementation path from the solution's tensor `torch.minimum/clamp_min/atan`. 40 fixed-seed random box pairs cover both {none, mean}; float32 tolerance \(10^{-5}\).
- `_plain_iou_loss` is a **second, independent implementation** (sharing no code with the oracle) used only for the inequalities \(\text{loss}_{giou}\ge1-\text{IoU}\), \(\text{loss}_{diou}\ge1-\text{IoU}\), \(\text{loss}_{ciou}\ge\text{loss}_{diou}\). These pin the relative ordering of the three variants and catch sign errors that pure numeric matching would miss.
- Discriminating cases: `pred == target` (all three losses 0 — kills the \(\alpha\) NaN), disjoint all-square pairs (loss_giou > 1 *and* loss_ciou == loss_diou), variant ordering on random boxes, reduction shapes and ValueError, and both N=0 branches.
- Example constants were backfilled by running the reference implementation: the six per-pair values and three means (\(\text{mean}=(\text{pair0}+\text{pair1})/2\) is checkable on paper).
- Complexity: time \(O(N)\) (a constant number of comparisons/multiplications per pair plus two `atan` calls); extra memory is a dozen or so intermediate \(O(N)\) tensors. `atan` is the only transcendental, and vectorizing it only changes constants versus a per-pair Python loop.
