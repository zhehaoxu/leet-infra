# PEFT: SuperTuning Sparse Scalar Support with Optional LoRA — Algorithm Notes

## 1. Formula and Source

SuperTuning (arXiv:2607.09287) is based on the observation that only a small number of elements in a large model weight need fine-grained correction, and these positions correspond to the peaks of \(|W_0|\). It therefore restricts the trainable update \(S\) to the top-\(k\) support of \(|W_0|\):

\[\Delta W = M_k(W_0)\odot S.\]

When a LoRA side path is added, the hybrid variant is called **Supra**:

\[\Delta W = M_k(W_0)\odot S + \frac{\alpha}{r}\,BA.\]

## 2. Deterministic top-\(k\) Mask

- Flatten \(|W_0|\), stable-sort in descending order (`stable=True`); tied elements enter the top-\(k\) in increasing original flat-index order.
- Obtain a 0/1 mask \(M_k\) with the same shape as \(W_0\).
- Multiply \(M_k\) with \(S\) element-wise, then add to \(W_0\) and the LoRA term.

## 3. Degenerate Cases and Grading Points

- \(k=0\): mask is all zeros, leaving only the LoRA term (if \(r>0\)).
- \(r=0\): ignore \(A,B,\alpha\), leaving only the sparse scalar term.
- \(\alpha=0\): LoRA term vanishes.
- \(S=0\): sparse scalar term vanishes.
- The tie-break test verifies that stable sorting prefers the smaller flat index.

## 4. Split with Existing Problems

- `peft-lora-forward`/`peft-lora-merge`: pure LoRA update.
- `sparsity-magnitude-prune`: drops weights by magnitude (irreversible compression).
- This problem: selects a trainable scalar support by magnitude and can mix it with LoRA; the focus is on mask construction and mixed merging.
