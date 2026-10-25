# DoRA Magnitude-Direction Decomposed Low-Rank Adaptation Explained

## Source
Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation* (arXiv:2402.09353). Following the idea of Weight Normalization, the weight is split into "magnitude \(m\)" and "direction \(\hat V\)": \(W_\text{eff}=m\odot \hat V\), where the direction \(\hat V=V/\lVert V\rVert\) and \(V=W_0+sBA\) reuses LoRA's low-rank composition. Only \(m\) and \(A,B\) are trainable; \(W_0\) is frozen.

## Core points and pitfalls
1. **The norm is along dim=1 (output rows)**: DoRA normalizes each row of the weight (corresponding to one output neuron) separately, so \(n\) has shape \((d_\text{out},1)\). Using the wrong dimension breaks the degenerate properties.
2. **Compose before normalizing**: \(V\) already includes the LoRA increment, and the direction \(\hat V\) normalizes \(V\), not \(W_0\).
3. **Magnitude broadcasting**: \(m:(d_\text{out},)\) must be `unsqueeze(1)`-ed to \((d_\text{out},1)\) to multiply correctly onto each row along the input dimension.
4. **Degenerate checks**: with \(m=\lVert V\rVert_\text{row}\), \(W_\text{eff}=V\), so DoRA == merged LoRA forward; with \(\alpha=0\) and \(m=\lVert W_0\rVert\), it returns to the base. These two invariants are probes for whether the direction/broadcasting was written backwards.

## Grading details
Reference solution, batched: \(V=W_0+s(BA)\), \(n=\text{norm}(V,\text{dim}=1)\), \(W_\text{eff}=(V/n)\cdot m[:,None]\), \(y=xW_\text{eff}^\top\). The grader's `_oracle` explicitly loops row by row with \(n_i=\sqrt{\sum_j V_{ij}^2}\), \(W_{ij}=m_i V_{ij}/n_i\) (a different structure, numerically equivalent), `allclose(atol=1e-5,rtol=1e-5)`. Additional cases: the two degenerate invariants and 3D batches.

## Complexity
Compared with LoRA, only one extra \((d_\text{out},)\) magnitude vector; each step adds one row-wise norm and one normalization division.
