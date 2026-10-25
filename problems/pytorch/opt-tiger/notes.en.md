# Tiger Sign-of-Momentum Update Explained

## Source and notes
The reference implementation is `Tiger` from kozistr/pytorch-optimizer (`weight_decouple=True`, `maximize=False`), whose documentation describes it as "Sign based updates with a single gradient momentum buffer". The algorithm belongs to the **sign-based optimizer family**: the same lineage as Lion (arXiv:2305.14616) and SignSGD — using the sign of the update direction to normalize each step's magnitude. Tiger can be seen as the minimal version that "has only momentum and takes the sign of the momentum directly" (Lion takes the sign of an interpolation between `β1·m+(1−β1)·g` and `g`; Tiger only takes the sign of the EMA momentum).

> Note: this collection requires verifiable sources. Tiger's implementation convention follows kozistr/pytorch-optimizer (default β=0.965, decoupled wd=0.01); no independent authoritative paper was found, so we do not invent an arXiv number, and the problem's criteria are based entirely on that implementation and its update formula.

## Update formula and key points
Each parameter has only one momentum \(m\), initialized to 0:

\[ m \leftarrow \beta m + (1-\beta) g,\qquad p \leftarrow p - lr\cdot\operatorname{sign}(m) \]

- **No second moment, no bias correction**: because \(\operatorname{sign}\) is taken, the update magnitude is always \(lr\), independent of the size of \(m\) (only its sign matters). This makes it naturally immune to gradient scale, but it also means the direction is already \(\pm 1\) while the early momentum is still small.
- **The sign function at 0**: \(\operatorname{sign}(0)=0\), so that coordinate does not move on that step. Random floating-point gradients in the grader are almost never exactly 0, but the test cases guarantee that parameters with `grad is None` skip even the weight decay.
- **Decoupled weight decay precedes momentum**: `p.mul_(1 - lr*wd)`, the same convention as AdamW.

## Grading details
The grader uses an independent recomputation (`_oracle_tiger`): `m = beta*m+(1-beta)*g`, `p -= lr*torch.sign(m)`, compared under multiple shapes and multiple steps of random gradients with `allclose(atol=1e-5, rtol=1e-5)`. The sign is an element-wise, discrete criterion; floating-point noise does not flip the sign (values stay far from 0), so it aligns naturally with the solution.

## Complexity
\(O(\text{numel})\) per parameter per step, with only one momentum \(m\) as extra state, half that of Adam.
