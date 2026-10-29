# PEFT: FourierFT Discrete Fourier Basis Coefficient Reconstruction

## Description

**FourierFT** (arXiv:2405.18201) treats the weight update \(\Delta W\) as a \(d_\text{out}\times d_\text{in}\) real image and sparsely represents it with a small number of learnable **discrete Fourier cosine-basis coefficients**:

\[\Delta W[i,j] = \sum_{(u,v)\in\mathcal F} c_{u,v}\,\cos\!\left(2\pi\left(\frac{u\,i}{d_\text{out}} + \frac{v\,j}{d_\text{in}}\right)\right),\]

where \(\mathcal F\) is the set of learned frequency positions and \(c_{u,v}\) are the corresponding coefficients. In this problem the frequency positions and coefficients are given as inputs; reconstruct \(\Delta W\) and merge it into the base weight:

\[W' = W_0 + \alpha\,\Delta W.\]

Frequency \(u\) satisfies \(0\le u<d_\text{out}\) and \(v\) satisfies \(0\le v<d_\text{in}\).

## Task

Given \(W_0\in\mathbb R^{d_\text{out}\times d_\text{in}}\), frequency positions `freqs` (int64 tensor of shape \((m,2)\), each row is \((u,v)\)), corresponding coefficients `coeffs` (float32 tensor of length \(m\)), and scaling \(\alpha\), return \(W'\) according to the formula above.

- You may assume all frequencies in `freqs` are within valid ranges.
- \(\alpha=0\) degenerates to \(W_0\).
- Empty frequency list (\(m=0\)) gives \(\Delta W=0\), also degenerating to \(W_0\).
- Output \(W'\in\mathbb R^{d_\text{out}\times d_\text{in}}\), float32.

## Hint

- Use row/column coordinate grids `arange(d_out)[:, None]` and `arange(d_in)[None, :]` to broadcast the phase matrix.
- For each frequency \((u,v)\) compute `cos(2*pi*(u*i/d_out + v*j/d_in))`, multiply by the coefficient, and accumulate.
- This is a simplified real-cosine-basis version of FourierFT; the judge only cares whether the given sparse frequency coefficients correctly reconstruct \(\Delta W\).
