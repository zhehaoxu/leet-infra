# RMSNorm Explained

## Definition

A simplified LayerNorm: **no mean subtraction**, it only scales by the root mean square:

$$ \hat{x}_i=\frac{x_i}{\sqrt{\frac1d\sum_{j}x_j^2+\epsilon}},\qquad y_i=\gamma_i\hat{x}_i $$

There is no $\beta$ (the bias term of centering).

## Why use it

- One fewer mean reduction and subtraction, so it is faster and cheaper, and empirically matches or slightly beats LN.
- Mainstream large-model Transformers such as LLaMA and Gemma all replace LN with RMSNorm — understanding it is the entry point to modern LLM normalization.

## Implementation notes

- The root mean square is `rms = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)`; for `(B,...,C)` reduce along the last dimension.
- **Divide by rms only, never do `x - mean`** — that is the single structural difference from LN, and writing the mean subtraction is a bug.
- $\gamma$ (weight) has shape `(C,)` and broadcasts over the last dimension; no bias needed.
- eps is added **inside** the square root.

## Scale invariance

$y$ is insensitive to an overall rescaling of the input: under $x\to cx$ the rms is multiplied by $c$ too, so the ratio is unchanged (this is exactly why QK-Norm uses it to stabilize attention; see the related "QK-Norm" problem).

## Reference solution for the comparison

There is no built-in `F.rms_norm` (older versions), so the reference is written by hand: `x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps) * weight`. `torch.rsqrt` is slightly more stable and faster than `1/sqrt`.
