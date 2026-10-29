# PEFT: FourierFT Discrete Fourier Basis Coefficient Reconstruction — Algorithm Notes

## 1. Formula and Source

FourierFT (arXiv:2405.18201) belongs to the transform-domain sparse-adaptation family. It represents the weight update \(\Delta W\) as a linear combination of a small number of 2D frequency components:

\[\Delta W[i,j] = \sum_{(u,v)\in\mathcal F} c_{u,v}\,\cos\!\left(2\pi\left(\frac{u\,i}{M} + \frac{v\,j}{N}\right)\right).\]

Here \(M=d_\text{out},N=d_\text{in}\), \(\mathcal F\) is the learnable frequency set, and \(c_{u,v}\) are the trainable coefficients. The merge is:

\[W' = W_0 + \alpha\,\Delta W.\]

## 2. Implementation Details

- Build row coordinates `i = arange(M)[:, None]` and column coordinates `j = arange(N)[None, :]`, broadcast to \((M,N)\).
- For each \((u,v)\) compute the phase matrix \(\phi_{u,v}[i,j]=2\pi(u i/M + v j/N)\).
- Accumulate `coeff * cos(phase)` into \(\Delta W\).
- Multiply by \(\alpha\) and add back to \(W_0\).

## 3. Degenerate Cases and Grading Points

- \(\alpha=0\): \(\Delta W\) is zeroed, so \(W'=W_0\).
- Empty frequency list: \(\Delta W=0\), so \(W'=W_0\).
- Only \((0,0)\) frequency: \(\cos\equiv 1\), so \(\Delta W\) is a uniform constant matrix.
- Output is fully determined by \(W_0,\text{freqs},\text{coeffs},\alpha\); no randomness.

## 4. Split with Existing Problems

- `peft-loha-hadamard`: time-domain Hadamard-product low-rank update.
- This problem: frequency-domain sparse coefficient reconstruction, emphasizing the recovery of a full weight update from a small set of frequency components.
