# PEFT: DoRA Magnitude-Direction Decomposed Low-Rank Adaptation

## Problem

**DoRA** (Weight-Decomposed Low-Rank Adaptation, Liu et al., arXiv:2402.09353) decomposes the weight vector into "magnitude × direction", then applies a LoRA-style low-rank update only to the **direction** part. Compared with plain LoRA, it makes the low-rank increment focus more on changing the weight direction, so with the same parameter count it often comes closer to full fine-tuning.

Given a frozen base \(W_0\) (\((d_\text{out}, d_\text{in})\)) and low-rank factors \(A:(r, d_\text{in})\), \(B:(d_\text{out}, r)\), scaling \(s=\alpha/r\), and a trainable magnitude vector \(m:(d_\text{out},)\):

1. Compose the unnormalized weight \(V = W_0 + s\,B A\) (\((d_\text{out}, d_\text{in})\)).
2. Take the L2 norm along **each output row** \(n = \lVert V\rVert_2 \;(\text{dim}=1,\ \text{keepdim})\), i.e. the direction \(\hat V = V / n\).
3. Magnitude-direction recomposition \(W_\text{eff} = m \odot \hat V\) (\(m\) broadcasts along the input dimension to every row).
4. Forward \(y = x\,W_\text{eff}^\top\).

## Rules

Input `x` shape \((\dots, d_\text{in})\). Return `y` (shape \((\dots, d_\text{out})\)). float32 throughout, ready-made PEFT libraries forbidden.

## Examples

Using the same fixed inputs as the LoRA problem: \(W_0(3\times5), A(2\times5), B(3\times2), x(2\times5)\), \(\alpha=8.0, r=2\) (\(s=4.0\)), magnitude \(m=[0.7, 0.8, 0.6]\):

```
y = [[ 0.3858,  0.9513, -0.3756],
     [ 0.3263, -0.1246, -0.0308]]
```

## Constraints

- CPU float32; `x` supports 2D and 3D (last dim is \(d_\text{in}\)).
- The row norm \(n\) is along dim=1 (one scalar per output row), and the length of \(m\) equals \(d_\text{out}\).
- **Degenerate properties**: when \(m = \lVert V\rVert_\text{row}\), \(W_\text{eff}=V\), so the DoRA forward equals the forward after merging LoRA; when \(\alpha=0\) and \(m=\lVert W_0\rVert_\text{row}\), it returns to the base forward \(x W_0^\top\).
