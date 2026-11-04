# Notes: BEFT Bypass-Bias Fine-Tuning

## Provenance

The paper BEFT: Bias-Efficient Fine-Tuning of Language Models in Low-Data Regimes (arXiv:2509.15974, ACL 2026 main conference, Baichuan Huang / Ananth Balashankar / Amir Aminifar, Lund University + Google DeepMind, merged into HF PEFT v0.20 `tuners/beft`). Research question: BitFit showed that fine-tuning ALL biases approaches full fine-tuning in low-data regimes, but the three attention biases \(b_q/b_k/b_v\) are not equivalent — which class is worth training? Answer: directly fine-tuning \(b_v\) consistently beats \(b_q\)/\(b_k\) in low-data regimes, across encoder-only/decoder-only architectures up to 6.7B parameters (including bias-free LLMs).

The structural argument: with softmax attention \(Q=XW_q+b_q\), \(K=XW_k+b_k\), \(V=XW_v+b_v\) — adding the same \(b_k\) to every Key shifts all attention logits by a constant, which the softmax **cancels**; \(b_q\)'s effect is entangled with query-key interactions; \(b_v\) acts linearly and directly on the value path, so every bit of its gradient counts. On the implementation side, BEFT **adds** a bypass bias to the target layer (rather than training the original bias — bias-free models can use it too); this is the implementation boundary from BitFit.

## Line-by-line translation of the official implementation

Three accounts in `layer.py` (device/dtype details removed):

```python
# update_layer: bypass bias shape + zero init
weight = torch.randn((1, self.out_features))
self.beft_bias[adapter_name] = nn.Parameter(weight)
if init_weights:
    nn.init.constant_(self.beft_bias[adapter_name], 0.0)
# note: a None base bias only warns "cannot merge"; forward/training proceed

# forward: multi-adapter summed first, added to the base output once
beft_bias = None
for active_adapter in self.active_adapters:
    current_bias = self.beft_bias[active_adapter].flatten()
    beft_bias = current_bias if beft_bias is None else beft_bias + current_bias
result = self.base_layer(x)
if beft_bias is not None:
    result = (result + beft_bias).to(orig_dtype)

# merge / unmerge state machine
if base_layer.bias is None:
    raise ValueError(f"Base layer has no bias, cannot merge bias adapter ...")
if safe_merge:
    output_bias = (base_layer.bias.data + beft_bias.squeeze()).clone()
    if not torch.isfinite(output_bias).all():
        raise ValueError(f"NaNs detected ...")
    base_layer.bias.data = output_bias.to(orig_dtype)
else:
    base_layer.bias.data.add_(beft_bias.squeeze()).to(orig_dtype)
# unmerge: base_layer.bias.data - beft_bias.squeeze()
```

Three key readings:

1. **Bypass, not in-place**: beft_bias is an ADDED parameter; the original weight/bias are untouched in forward — at the zero-init starting point forward ≡ base. BitFit trains the ORIGINAL bias (the `peft-bitfit-count` parameter account); that is the implementation boundary.
2. **flatten (forward) vs squeeze (merge)**: the official forward uses `.flatten()` and merge uses `.squeeze()`; on a \((1,out)\) shape both yield \((out,)\) — different semantic origins but numerically indistinguishable, so no mutation pins them.
3. **safe_merge checks the SUM**: `(base + beft).clone()` then `isfinite(...).all()` — it catches operand NaN and also the overflow of two finite values (1e308+1e308=inf); the non-safe path does not intercept and NaN is written back as-is.

**Multi-adapter semantics**: with more than one active adapter the biases accumulate first and the base output receives the sum once; the additive structure guarantees both "unmerge one by one" and "unmerge the sum" restore the original bias (round trip to 1e-12 in float).

## Pitfalls

1. **Dropping the base bias or the bypass**: forward = base + bypass, neither can be missing (m-fwd-nobase / m-fwd-nobeft); forgetting the weight transpose is also pinned (m-fwd-wnot).
2. **Broadcast axis**: beft_bias must reshape to \((out,)\) and broadcast on the last dim; reshaping to a column \((out,1)\) broadcasts on the wrong axis (m-fwd-col).
3. **Merge direction**: merge adds and unmerge subtracts; swapping them breaks the round trip (m-merge-sub / m-unmerge-add).
4. **The bias-free guard**: merge with a None base bias must raise ValueError — silently substituting a zero vector breaks the "original bias exists" premise (m-merge-noguard); unmerge has no such guard (the official code only warns).
5. **The safe-intercept semantics**: with safe=False a NaN result is returned legally (official behavior); only safe=True raises (m-merge-nosafe). The check target is the sum, not the operands — pinned with an overflow fixture (m-merge-safeop SURVIVED until the overflow case was added).
6. **The shape account**: beft_bias is \((1, out)\), not \((out,1)\) (m-setup-shape); the parameter count is exactly out with no in-dim term (m-setup-count); the zero init is what makes training start at \(\Delta=0\) (m-setup-nozero).
7. **Multi-adapter summation**: keeping only the first adapter drops the rest (m-sum-first); subtracting breaks accumulation (m-sum-sub).
8. **Equivalent-mutation records**: `base + beft then add` vs `beft + base then add` is float-addition commutativity, an identity (m-fwd-swap SURVIVED, recorded and removed — do not pin again); flatten vs squeeze coincide on \((1,out)\) for the same reason.

## Cross-check account

Anchors dumped from the sandbox (torch float64, seed 42): in=4/out=6, x (2,3,4). Three cross-check lines — (1) forward ≡ the explicit x@Wᵀ + b + beft expansion; (2) forward with "zero bypass + merged bias" ≡ the original forward (bypass-absorption equivalence); (3) the merge→unmerge round trip ≡ the original bias (single- and multi-adapter variants). safe_merge uses two fixtures: a NaN operand and a 1e308 overflow. Of 16 mutants 15 CAUGHT plus 1 recorded equivalent.

## Complexity

The forward increment is \(O(out)\) (the bypass add), negligible against the base linear layer's \(O(BN\cdot in\cdot out)\); merge/unmerge are each \(O(out)\); trainable parameters \(\sum_{\text{value layers}} out\) — on the order of 0.01% of parameters approaching full fine-tuning (the paper's headline), the leanest problem in the peft group.
