# In Depth: KTO Prospect-Theoretic Loss

## Source
Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization* (arXiv:2402.01306). The value function is defined relative to the reference point \(z_0\): desirable takes \(\sigma(\beta(z-z_0))\), undesirable takes \(\sigma(\beta(z_0-z))\), and the loss is \(\lambda_y(1-v)\). The reference point \(z_0\approx\mathrm{KL}(\pi_\theta\|\pi_\text{ref})\) is estimated within the batch and detached in the paper; here it is given as an explicit constant input, focusing on the loss itself.

## Core idea and pitfalls
1. **The two directions are opposite**: good samples reward \(z>z_0\) (the value rises with \(z-z_0\)), bad samples penalize \(z>z_0\) (the value rises with \(z_0-z\)). Writing the direction backwards would encourage bad samples.
2. **The loss is 1−σ, not −log σ**: KTO directly minimizes the "unachieved prospect value" \(1-\sigma(\cdot)\) (the linearly weighted prospect value of prospect theory), unlike DPO's logsigmoid.
3. **λ weights**: \(\lambda_D,\lambda_U\) mitigate the imbalance in the number of good/bad samples; when equal it degenerates to equal weighting.
4. When \(z=z_\text{ref}\), \(\sigma(0)=0.5\) → the per-sample loss is exactly \(\lambda/2\), a probe for checking the direction.

## Cross-check methodology
The reference solution vectorizes `torch.where(label, λ_D*(1-sigmoid(β(z-z_ref))), λ_U*(1-sigmoid(β(z_ref-z)))).mean()`. The judging `_oracle` uses the identity \(1-\sigma(x)=\sigma(-x)\) to rewrite the two branches as `sigmoid` of a negated argument and recomputes with an element-wise Python loop (different structure, numerically equivalent), with `allclose(atol=1e-5,rtol=1e-5)`. There are also assertions for "all samples z==z_ref → loss=λ/2" and "label direction".

## Complexity
Element-wise \(O(N)\).
