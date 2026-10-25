# PEFT: IA3 Element-wise Activation Rescaling

## Problem

**IA³** (Infused by **I**nhibiting and **A**djusting with **A**t**t**ention, Liu et al., arXiv:2205.05638) is a PEFT method more parameter-efficient than LoRA: it does not touch the weight matrices, but learns a few **element-wise scaling vectors** that multiplicatively rescale the **intermediate activations** of attention/feed-forward along a dimension (amplifying useful dimensions, suppressing irrelevant ones). The original IA³ acts separately on the self-attention keys and values and on the FFN intermediate layer; this problem abstracts it to rescaling the output of a single linear transformation.

Given the output of a linear layer followed by an IA³ vector \(\lambda\), the forward pass is:

\[y = \lambda \odot (W x)\]

\(\lambda\) broadcasts along the output dimension (the same \(\lambda\) is used for every sample in the batch and every token). Initialization is \(\lambda=\mathbf 1\) (identity, leaving the pretrained model unchanged).

## Rules

Given input `x` shape \((\dots, d_\text{in})\), weight `W` shape \((d_\text{out}, d_\text{in})\), IA³ scaling vector `lam` shape \((d_\text{out},)\):

1. Linear transformation \(h = x\,W^\top\) (`nn.Linear` convention, shape \((\dots, d_\text{out})\)).
2. Rescale along the output dimension \(y = \lambda \odot h\) (\(\lambda\) broadcasts on the last dimension).

Return `y` (shape \((\dots, d_\text{out})\)). float32 throughout, ready-made PEFT libraries forbidden.

## Examples

\(W\) (3×5) same as the LoRA problem; \(\lambda=[1.2,\ 0.8,\ 1.5]\); `x` (2×5) same as the LoRA problem:

```
y = [[0.516, 0.728, -1.020],
     [0.336, -0.256,  0.555]]
```

## Constraints

- CPU float32; `x` supports 2D and 3D (last dim treated as features).
- When \(\lambda\equiv\mathbf 1\) the result equals the plain linear \(xW^\top\).
- Element-wise equivalent to "first fold \(\lambda\) into the weight rows \((\lambda\odot W)\) and then matrix-multiply".
