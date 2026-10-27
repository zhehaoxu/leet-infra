# Detection Box IoU Variant Losses: GIoU / DIoU / CIoU

## Problem

\(1-\mathrm{IoU}\) is the plainest box-regression loss, but its gradient vanishes when the two boxes **do not overlap** (IoU stays exactly 0), and it cannot tell "the box is misplaced" from "the box has the wrong shape". Each variant patches one hole: GIoU (Rezatofighi et al., CVPR 2019, arXiv:1902.09630) subtracts the **enclosing-closure ratio** \((C\text{ area}-\text{union})/C\text{ area}\) so that non-overlapping pairs still get a direction; DIoU (Zheng et al., AAAI 2020, arXiv:1911.08287) replaces that term with the **center-distance ratio** \(\rho^2/c^2\) and converges faster; CIoU subtracts an additional \(\alpha v\) on top of DIoU so that aspect-ratio mismatch is penalized too. This problem computes all three at once and returns \((1-\mathrm{GIoU},\,1-\mathrm{DIoU},\,1-\mathrm{CIoU})\). CPU, fully deterministic, no random number calls, forward pass only (gradients w.r.t. an upstream grad are out of scope).

## Rules

- The complete signature, argument meanings and defaults live only in the starter docstring; that is the sole judging contract. Boxes use \([x_1,y_1,x_2,y_2]\); pred and target have the same shape (N,4) and are paired row by row. Inputs are guaranteed to be valid boxes (\(x_2>x_1\), \(y_2>y_1\)) — you do **not** need to validate them; N=0 is allowed; reduction accepts only `"mean"` (default) and `"none"`, anything else raises ValueError.
- **Base IoU uses exactly the `iou-compute` convention**: intersection width and height are each `clamp_min(0)`-ed, \(\text{inter}=\text{iw}\cdot\text{ih}\), \(\text{union}=A_p+A_t-\text{inter}\), \(\text{IoU}=\text{inter}/\text{union}\); when \(\text{union}=0\) (both boxes degenerate) IoU is 0.
- **Enclosing box C**: \(C_{x_1}=\min(a_{x_1},b_{x_1})\), \(C_{y_1}=\min(a_{y_1},b_{y_1})\), \(C_{x_2}=\max(a_{x_2},b_{x_2})\), \(C_{y_2}=\max(a_{y_2},b_{y_2})\), width and height `clamp_min(0)`; \(C\text{ area}=w_C h_C\) and \(c^2=w_C^2+h_C^2\) (the squared diagonal).
- **The three formulas** (each penalty term is 0 whenever its own denominator is 0, so the output is always finite):

  \[\text{GIoU}=\text{IoU}-\frac{C\text{ area}-\text{union}}{C\text{ area}},\qquad
    \text{DIoU}=\text{IoU}-\frac{\rho^2}{c^2},\qquad
    \text{CIoU}=\text{DIoU}-\alpha v\]

  where \(\rho^2\) is the **squared distance between the two box centers**, \(v=\frac{4}{\pi^2}\big(\arctan\frac{w_t}{h_t}-\arctan\frac{w_p}{h_p}\big)^2\) and \(\alpha=\dfrac{v}{(1-\text{IoU})+v}\). Before the aspect-ratio divisions, each \(w,h\) is `clamp_min(1e-7)`; when \((1-\text{IoU})+v=0\) (boxes exactly coincident) \(\alpha\) is 0.
- All losses are **\(1-\) similarity**: return a 3-tuple in the fixed order `(loss_giou, loss_diou, loss_ciou)`. `"none"` gives three (N,) tensors, `"mean"` gives three 0-dim scalars; with N=0 and `"mean"`, the empty-set convention returns three zeros.

## Example

pred \(=[[0,0,2,2],[0,0,1,2]]\), target \(=[[1,1,3,3],[0.5,0.5,2.5,2.0]]\).

**Pair 0** (two \(2\times2\) squares offset by one cell on the diagonal): \(\text{iw}=\text{ih}=1\), inter=1, \(A_p=A_t=4\), union=7 → IoU \(=1/7=0.142857\); closure \(3\times3\) → \(C\text{ area}=9\), \(c^2=18\); centers \((1,1)\) vs \((2,2)\) → \(\rho^2=2\). Both boxes are squares → \(v=0\), \(\alpha=0\), so CIoU degenerates to DIoU. Per-pair losses (EXAMPLE_NONE_*):

| loss_giou | loss_diou | loss_ciou |
|---|---|---|
| 1.0793650150 | 0.9682539701 | 0.9682539701 |

Here GIoU \(=1/7-2/9=-0.079365\), hence the loss is **>1** — exactly the situation where GIoU punishes a barely-overlapping pair harder than \(1-\)IoU does.

**Pair 1** (a \(1\times2\) tall box vs a \(2\times1.5\) flat box): inter \(=0.5\times1.5=0.75\), union \(=2+3-0.75=4.25\) → IoU \(=0.176471\); closure \(2.5\times2\) → \(C\text{ area}=5\), \(c^2=10.25\); centers \((0.5,1)\) vs \((1.5,1.25)\) → \(\rho^2=1.0625\); \(v=\frac4{\pi^2}(\arctan\frac2{1.5}-\arctan\frac12)^2=0.087123\), \(\alpha=0.087123/0.910652=0.095670\). Per-pair losses:

| loss_giou | loss_diou | loss_ciou |
|---|---|---|
| 0.9735293984 | 0.9271879196 | 0.9355232120 |

**Default `mean`** (EXAMPLE_*): loss_giou = 1.0264471769, loss_diou = 0.9477209449, loss_ciou = 0.9518885612 — the pairwise means of the two rows above.

## Constraints

- pred/target both shape (N,4), N may be 0; box convention \([x_1,y_1,x_2,y_2]\); no batch/class dimension, no coordinate scaling, no cxcywh conversion
- reduction ∈ {`"mean"`, `"none"`}; `"sum"` and anything else raise ValueError; fixed triple order; `mean` returns 0-dim scalar tensors, `none` returns (N,)
- Numerics: test every denominator before dividing (union, C area, \(c^2\), \((1-\text{IoU})+v\)) and record the corresponding penalty term as 0 when it vanishes — NaN/Inf are never allowed; aspect ratios clamp \(w,h\) to \(10^{-7}\) first
- Cross-check tolerance float32 \(10^{-5}\) (atol and rtol). Extra inequality probes: all three losses are 0 when `pred == target`, \(\text{loss}_{ciou}\ge \text{loss}_{diou}\ge 1-\text{IoU}\), loss_giou > 1 for disjoint pairs, and loss_ciou equals loss_diou pair-wise when every box is a square
