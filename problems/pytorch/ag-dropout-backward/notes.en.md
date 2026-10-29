# Notes: Where the Scaling Lives Decides What the Backward Looks Like

## Origin

The dropout backward is trivial on its own, but it is the perfect vehicle for the "vanilla dropout versus inverted dropout" concept question and for the train/eval asymmetry (`model.train()` vs `model.eval()`): an inference engine that carries the training-time \(1/(1-p)\) into the eval graph produces outputs that are uniformly too large. People who have done ONNX export or inference alignment have muscle memory for this.

## Equivalence of the Two Conventions

Vanilla dropout uses \(y=x\cdot mask\) forward and \(dx=dY\cdot mask\) backward, deferring the \(1-p\) compensation to inference-time weight scaling; inverted dropout uses \(y=x\cdot mask/(1-p)\) forward, and since \(\partial y/\partial x = mask/(1-p)\) the backward **also carries a \(1/(1-p)\) factor**. Both preserve the expectation but move the scaling to a different place. PyTorch uses the latter — measured, the gradient values of `F.dropout(x, 0.5)` are \(\{0,2\}\), not \(\{0,1\}\) — so mixing the two conventions divides one time too many or too few; judging compares against `F.dropout`.

## train=False Is a Full Identity

At inference dropout is the identity map, so its backward is too: `mask` may have any shape (even all zeros) and `p` may be anything usable, yet the result must equal `dY`. This is the easiest case for an implementation to miss — putting the division outside the mode branch amplifies eval outputs by \(1/(1-p)\).

## Core Ideas and Pitfalls

- **Broadcast masks**: `dropout2d` masks have shape `(1,C,1,1)` and must be `expand`ed to the shape of `dY` rather than required to match; a non-broadcastable mask must raise ValueError instead of silently mis-aligning.
- **`p=0` is legal**: the scale becomes 1, but the training branch (using the mask) still applies.
- **`mask` dtype**: multiply through float64 explicitly so the result matches autograd bit for bit.
- **Never scale twice**: if the caller already applied the forward, the backward divides by \(1-p\) exactly once.

## How It Differs from Neighboring Problems

`ag-activation-backward` is a pure elementwise analytic derivative with no stochastic component; `ag-embedding-scatter-backward` also uses a mask, but there it is a semantic constant (padding) rather than a sampled one. The signature difficulty here is that a mode switch changes the formula itself.

## Cross-Check Basis and Complexity

The oracle runs `F.dropout(x, p, training=True)` and `torch.autograd.grad`, feeding the very mask sampled by that forward for comparison, atol=1e-9 in float64. The example uses \(p=0.5\) so the scale is the integer 2 and all four entries are checkable at a glance. Complexity is \(O(\text{numel})\): a single elementwise multiply and divide.
