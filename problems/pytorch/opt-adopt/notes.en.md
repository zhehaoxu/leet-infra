# In Depth: ADOPT Bias-Correction-Free + Previous-Step Second Moment + Gradient Clipping

## Source
Taniguchi, Harada, Minegishi, Oshima, Guo, Nagahara, Imaizumi, Kashiwabara, Suzuki, *ADOPT: Modified Adam Can Converge with Any β₂ with the Optimal Rate* (arXiv:2411.02853). The reference implementation is `ADOPT` from kozistr/pytorch-optimizer (`clip_lambda=step**0.25`, `maximize=False`, with `cautious/stable_adamw` disabled).

> Weight-decay convention: the library defaults to `weight_decouple=False` (adding \(\lambda p\) to the gradient). This problem uniformly uses the **decoupled** form \(p\leftarrow p(1-lr\cdot\text{wd})\), staying consistent with AdamW/Yogi and the rest of the collection and avoiding in-place rewrites of `p.grad`. With the default `weight_decay=0` the two are indistinguishable.

## Three structural changes of ADOPT relative to Adam
1. **Bias correction removed**: \(1-\beta^t\) no longer appears; the momentum \(m\) is the update direction directly and the step size is simply \(lr\). The authors prove it is precisely bias correction that destroys convergence for arbitrary \(\beta_2\).
2. **Normalization by the previous step's second moment**: at step \(t\) the denominator comes from \(v_{t-1}\) (**before the current \(g^2\) is folded in**). So the update order is: first normalize \(g\) with the old \(v\) and update \(m\) and \(p\), and only **last** do `v ← β2·v + (1−β2)·g²`.
3. **Special first step**: at \(t=1\) only \(v\leftarrow g^2\) is initialized and **no parameter update happens** (the old \(v=0\) would divide by zero, so it is skipped). This is the easiest thing to get wrong and the judge tests it explicitly.
4. **Clipping**: the normalized gradient is clamped inside \(\pm t^{1/4}\) (`clip_lambda=step**0.25`); \(t^{1/4}\) loosens slowly with the step count, suppressing anomalously large normalized steps early and becoming nearly inactive later. The denominator is additionally `clamp(min=eps)` element-wise.

## Cross-check convention
Judging uses an independent vectorized recomputation (`_oracle_adopt`): written differently (`g/de_nom`, `clamp`, `b1*m+(1-b1)*gt`, and so on) and compared with `allclose(atol=1e-5, rtol=1e-5)` over random gradients on many shapes and step counts. `check_first_step_only_inits_v` verifies specifically that the first step leaves the parameter untouched apart from weight decay; parameters with `grad is None` are skipped entirely.

## Complexity
Same as Adam: \(O(\text{numel})\) per parameter per step, with two states \(m,v\).
