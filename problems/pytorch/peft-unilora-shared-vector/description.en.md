# PEFT: UniLoRA Shared-Parameter Vector Index Assignment

## Description

**UniLoRA** (Unified Low-Rank Adapter, arXiv:2506.00799) compresses all LoRA adapter trainable parameters into a **single global shared vector** \(\theta\in\mathbb R^D\). Each linear layer no longer owns its own \(A/B\); instead, a **fixed pseudo-random index assignment** determined by `proj_seed` extracts the \(r(d_\text{in}+d_\text{out})\) elements needed for that layer from \(\theta\), reshaping them into \(A\in\mathbb R^{r\times d_\text{in}}\) and \(B\in\mathbb R^{d_\text{out}\times r}\), then merges them with the standard LoRA scaling:

\[W' = W_0 + \frac{\alpha}{r}\,B A.\]

`layer_id` distinguishes different layers in the model: each layer's random seed is `proj_seed + layer_id`, so index assignments within the same model are non-overlapping and fully deterministic.

## Task

Given frozen base \(W_0\in\mathbb R^{d_\text{out}\times d_\text{in}}\), shared vector \(\theta\in\mathbb R^D\), rank \(r\), scaling \(\alpha\), projection seed `proj_seed`, and layer id `layer_id`, return the merged weight \(W'\) according to the rule above.

- You may assume \(r(d_\text{in}+d_\text{out}) \le D\) (tests will not violate this).
- Index assignment uses `torch.randperm(D, generator=...)`, with the generator seed set to `int(proj_seed) + int(layer_id)`.
- The first \(r\cdot d_\text{in}\) extracted elements reshape into \(A\) (row-major), and the next \(r\cdot d_\text{out}\) elements reshape into \(B\).
- Output \(W'\in\mathbb R^{d_\text{out}\times d_\text{in}}\), float32.

## Hint

- Generate the fixed permutation first, then slice, reshape, compute the low-rank product, and add back to \(W_0\).
- When \(\alpha=0\) or the selected \(\theta\) entries are all zero, \(W'\) should equal \(W_0\) exactly.
- Changing `layer_id` under the same `proj_seed` yields different \(A/B\) slices, simulating non-overlapping layer-wise assignments when a single vector is shared across the whole model.
