# Diffusion: Sinusoidal Timestep Embedding

## Problem

The denoising network of a diffusion model needs to "know which step it is at", but the timestep \(t\) is only an integer scalar, which carries too little information when fed in directly. The convention is to encode it as a series of **sines/cosines at different frequencies** (from the same lineage as Transformer positional encoding; Ho et al. DDPM appendix), so the network can perceive any \(t\) smoothly.

Given a scalar timestep \(t\) and an even dimension `dim`, let \(\text{half}=\text{dim}/2\) and the frequencies be

\[ \text{freq}_i = \frac{1}{10000^{\,i/\text{half}}},\quad i=0,1,\dots,\text{half}-1 \]

The angles are \(\theta_i = t\cdot \text{freq}_i\). The embedding concatenates "\(\sin\) in the first half, \(\cos\) in the second half":

\[ \text{emb} = \bigl[\sin\theta_0,\dots,\sin\theta_{\text{half}-1},\ \cos\theta_0,\dots,\cos\theta_{\text{half}-1}\bigr]\in\mathbb{R}^{\text{dim}} \]

## Rules

Return a float32 tensor of shape \((\text{dim},)\). \(t\) is passed as a Python scalar (int/float). Ready-made diffusion libraries are forbidden.

## Examples

\(t=7,\ \text{dim}=8\) (half=4, freq=\([1,10^{-1},10^{-2},10^{-3}]\)):

```
emb = [ 0.6569865942001343, 0.6442176103591919, 0.06994284689426422, 0.006999942008405924,
        0.7539022564888   , 0.7648422122001648, 0.9975510239601135, 0.9999755024909973 ]
```

The first 4 entries are `sin([7,0.7,0.07,0.007])`; the last 4 entries are `cos([7,0.7,0.07,0.007])`.

## Constraints

- `dim` is a positive even number; float32.
- At \(t=0\) the \(\sin\) half is all 0 and the \(\cos\) half is all 1.
- The frequency exponent is \(i/\text{half}\) (equivalent to \(2i/\text{dim}\)); the maximum period is fixed at 10000.
