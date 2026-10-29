# PEFT: UniLoRA Shared-Parameter Vector Index Assignment — Algorithm Notes

## 1. Formula and Source

UniLoRA (arXiv:2506.00799) compresses all LoRA adapter trainable parameters in the model into a **single shared vector** \(\theta\in\mathbb R^D\). The per-layer low-rank update still follows the PEFT style:

\[\Delta W = \frac{\alpha}{r}\,B A,\qquad A\in\mathbb R^{r\times d_\text{in}},\ B\in\mathbb R^{d_\text{out}\times r}.\]

The difference is that \(A/B\) are not independent layer parameters; they are slices taken from \(\theta\) according to a fixed random permutation. This problem only tests single-layer index assignment and merging; in a multi-layer setting, each layer uses `proj_seed + layer_id` as its own seed, ensuring non-overlapping index blocks from the same \(\theta\).

## 2. Index Assignment Details

- This layer needs \(r(d_\text{in}+d_\text{out})\) scalars in total.
- `torch.randperm(D, generator=g)` produces a permutation of \([0,D)\); the first \(r d_\text{in}\) entries become \(A\), and the next \(r d_\text{out}\) entries become \(B\).
- `generator.manual_seed(proj_seed + layer_id)` makes it reproducible and independent of the global RNG state.
- `A.reshape(r, d_in)` and `B.reshape(d_out, r)` both use row-major reshape.

## 3. Degenerate Cases and Grading Points

- \(\alpha=0\): scaling is zero, so \(W'=W_0\).
- Selected slice all zeros: \(\Delta W=0\), so \(W'=W_0\).
- Changing `layer_id` changes the permutation, simulating non-overlapping layer-wise assignments when one vector is shared across the whole model.
- Output depends only on \(W_0,\theta,r,\alpha,\text{proj_seed},\text{layer_id}\); no injected randomness, fully deterministic on CPU float32.

## 4. Split with Existing Problems

- `peft-lora-forward`/`peft-lora-merge`: each layer owns its own \(A/B\).
- This problem: \(A/B\) are slices of a shared vector, emphasizing index assignment and parameter sharing.
