# Solution Notes · SmoothQuant

## The Conflict: Activations Have Outliers, Weights Are Well-Behaved

Observations show that activation tensors in LLMs contain a few "outlier channels" (certain feature dimensions whose magnitudes exceed the other channels by 1-2 orders of magnitude), while weight distributions are relatively uniform. Directly applying INT8 per-tensor quantization to activations fills the range with the outlier channels, and the vast majority of normal channels get quantized to values near 0 — accuracy collapses. Weight quantization has no such trouble.

## Core Trick: An Equivalent Transformation Moves the Difficulty onto the Weights

Since activations are hard and weights are easy, reparameterize along the input channel with a **mathematically equivalent** transformation:

$$X W^\top = (X\,\text{diag}(s)^{-1})(W\,\text{diag}(s))^\top$$

As long as, within the same channel, $X$ is divided by $s_j$ and $W$ is multiplied by $s_j$, the product stays unchanged; choosing $s_j$ well "cuts the peaks and fills the valleys": it presses down the activation outlier channels and lifts the corresponding weight channels, leveling the quantization difficulty on both sides.

## Geometric Meaning of the Factor Formula

$$s_j = \frac{\max_i|X_{ij}|^\alpha}{\max_o|W_{oj}|^{1-\alpha}}$$

The numerator "claims" the channel's activation peak to the power $\alpha$ and the denominator cancels the weight peak to the power $1-\alpha$, exactly balancing the post-migration absmax ratio between activation and weight in that channel geometrically. $\alpha$ is the difficulty-allocation knob: commonly 0.5-0.8; too large and the weights get blown up instead, too small and the activations remain hard to quantize.

## Orthogonality to Other Quantization Problems

This problem stops at the "migration" step (it returns s and the migrated X/W) and involves no rounding. The actual quantization reuses the same symmetric/per-channel RTN machinery. The problem is cut this way to isolate and test SmoothQuant's signature **scaling factor**, which also makes judging the most stable.

## Invariance Self-Check `check_invariance`

After migration, `X_hat @ W_hat.T == X @ W.T` is the fastest assertion for catching a reversed direction for $s$ (multiplication and division mixed up). `check_alpha_zero_one` verifies the two extreme degenerations — $s=1/\max|W|$ at $\alpha=0$ and $s=\max|X|$ at $\alpha=1$ — ensuring the exponents are not written wrong.
