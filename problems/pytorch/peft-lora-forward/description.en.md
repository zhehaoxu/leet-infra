# PEFT: LoRA Low-Rank Adaptation Forward

## Problem

**LoRA** (Hu et al., arXiv:2106.09685) is the foundational parameter-efficient fine-tuning (PEFT) method: during fine-tuning, freeze the pretrained weight matrix \(W_0\in\mathbb R^{d_\text{out}\times d_\text{in}}\) and train only two low-rank "side" matrices \(A\in\mathbb R^{r\times d_\text{in}}\) (down-projection, \(r\ll\min(d)\)) and \(B\in\mathbb R^{d_\text{out}\times r}\) (up-projection), constraining the weight update to \(\Delta W = B A\). The forward pass does not modify \(W_0\); instead it adds "the original output of \(W_0\) + the output of the low-rank side path":

\[y = W_0 x + \frac{\alpha}{r}\,B A x\]

where \(\alpha\) is a scaling hyperparameter and \(\frac{\alpha}{r}\) is the standard scaling factor (calibrating the low-rank increment to the right magnitude). This problem implements the LoRA forward output for a given input batch, which is element-wise equivalent to "first merge \(\Delta W\) into \(W_0\) and then do a single matrix multiplication".

## Rules

Given: input `x` shape \((\dots, d_\text{in})\); frozen weight `W0` shape \((d_\text{out}, d_\text{in})\); `A` shape \((r, d_\text{in})\); `B` shape \((d_\text{out}, r)\); scalars `alpha` and `r` (rank). Linear layers follow PyTorch `nn.Linear`'s "row vector × weight transpose" convention, i.e. the original output \(=x W_0^\top\). Compute:

1. Scaling factor \(s = \alpha / r\).
2. Base output \(y_0 = x\,W_0^\top\).
3. Low-rank side path \(y_\Delta = s\cdot\bigl(x\,A^\top\bigr)B^\top\) (first go through \(A\) down to \(r\) dims, then through \(B\) back up to \(d_\text{out}\)).
4. Output \(y = y_0 + y_\Delta\).

Return `y` (shape \((\dots, d_\text{out})\)). float32 throughout, ready-made PEFT libraries forbidden.

## Examples

\(W_0\) (3×5), \(A\) (2×5), \(B\) (3×2), \(x\) (2×5), \(\alpha=8.0\), \(r=2\) (so \(s=4.0\)):

```
W0 = [[ 0.5,-0.2, 0.1, 0.3, 0.0],
      [ 0.2, 0.4,-0.1, 0.0, 0.6],
      [-0.3, 0.1, 0.5, 0.2,-0.4]]
A  = [[ 0.3,-0.1, 0.2, 0.0, 0.5],
      [-0.2, 0.4, 0.1,-0.3, 0.2]]
B  = [[ 0.1,-0.2],
      [ 0.3, 0.0],
      [-0.1, 0.4]]
x  = [[ 1.0, 0.5,-0.3, 0.2, 0.8],
      [ 0.1,-0.4, 0.6, 0.3,-0.2]]

y = [[0.610, 1.618, -0.804],
     [0.516, -0.212, -0.066]]
```

## Constraints

- CPU float32; `x` supports 2D \((n,d_\text{in})\) and 3D \((b,l,d_\text{in})\) (last dim as the feature dim).
- The output matches "merge the weights \(W_0+\frac{\alpha}{r}BA\) and do a single matrix multiplication" element-wise (grading details in notes.md).
- \(r\) is the shared rank dimension of \(A,B\); this formula uses the given `r` for \(\alpha/r\) and does not infer it from shapes.
