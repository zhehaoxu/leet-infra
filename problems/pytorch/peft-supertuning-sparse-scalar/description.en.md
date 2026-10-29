# PEFT: SuperTuning Sparse Scalar Support with Optional LoRA

## Description

**SuperTuning** (arXiv:2607.09287) places trainable sparse scalars at the positions of largest magnitude in the base weight \(W_0\), forming a fine-grained local correction; it can also be mixed with LoRA in the **Supra** variant (SuperTuning + LoRA):

\[\Delta W = M_k(W_0)\odot S + \mathbf 1_{r>0}\,\frac{\alpha}{r}\,BA,\]

where \(M_k(W_0)\) is a 0/1 mask selecting the top-\(k\) entries of \(|W_0|\) by magnitude (ties broken by smaller flat index), \(S\in\mathbb R^{d_\text{out}\times d_\text{in}}\) is the trainable scalar update matrix, and \(BA\) is the optional LoRA side path. When \(r=0\) only the sparse scalar term remains; when LoRA is present and \(\alpha>0\) the two terms are added.

## Task

Given \(W_0\in\mathbb R^{d_\text{out}\times d_\text{in}}\), sparse update \(S\), mask budget \(k\ge 0\), optional LoRA factors \(A\in\mathbb R^{r\times d_\text{in}},B\in\mathbb R^{d_\text{out}\times r}\), and \(\alpha,r\), return the merged weight:

\[W' = W_0 + M_k(W_0)\odot S + \mathbf 1_{r>0}\,\frac{\alpha}{r}\,BA.\]

- When \(k=0\) the mask is all zeros, leaving only the LoRA term (if \(r>0\)).
- When \(r=0\) ignore \(A,B,\alpha\) and keep only the sparse scalar term.
- Sort \(|W_0|\) in **stable descending** order; tied elements prefer the smaller flat index.
- Output \(W'\in\mathbb R^{d_\text{out}\times d_\text{in}}\), float32.

## Hint

- Flatten \(W_0\), take absolute values, stable-sort and pick the first \(k\) flat indices, then reshape back to the matrix shape to obtain \(M_k\).
- When \(\alpha=0\) or \(S=0\) the corresponding term vanishes; note that \(M_k\) only controls the sparse support of \(S\), not \(W_0\) itself.
- The naming of \(A/B\) here matches `peft-lora-forward`: \(A\in\mathbb R^{r\times d_\text{in}},B\in\mathbb R^{d_\text{out}\times r}\).
