# PEFT: RandLora Frozen Random Low-Rank Bases with Trainable Coefficients — Algorithm Notes

## 1. Formula and Source

RandLora (arXiv:2502.00987) replaces trainable \(A/B\) with **frozen random low-rank bases** and learns only a set of combination coefficients \(c\in\mathbb R^r\):

\[\Delta W = \frac{\alpha}{r}\,B\,\operatorname{diag}(c)\,A.\]

Here \(B\in\mathbb R^{d_\text{out}\times r}\) and \(A\in\mathbb R^{r\times d_\text{in}}\) are standard-normal random matrices generated from a fixed seed and not updated during training. The number of trainable parameters per layer drops from standard LoRA's \(r(d_\text{in}+d_\text{out})\) to \(r\).

## 2. Deterministic Generation of Random Bases

- Use a single `torch.Generator().manual_seed(base_seed)`.
- The generation order must be fixed: first generate \(B\) with shape `(d_out, r)`, then \(A\) with shape `(r, d_in)`.
- Because the generator state advances sequentially, any change in order will make \(A/B\) differ from the expected values.

## 3. Degenerate Cases and Grading Points

- \(c=0\): \(\Delta W=0\), so \(W'=W_0\).
- \(\alpha=0\): \(\Delta W=0\), so \(W'=W_0\).
- Different `base_seed` produces different random bases and thus different \(W'\).
- Output depends only on \(W_0, c, \alpha, r, \text{base_seed}\); no runtime randomness.

## 4. Split with Existing Problems

- `peft-lora-forward`/`peft-lora-merge`: both \(A/B\) are trainable.
- This problem: \(A/B\) are frozen random and only \(c\) is trainable, emphasizing trainable parameter count and random-basis reconstruction.
