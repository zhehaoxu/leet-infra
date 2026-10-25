# PEFT: LoRA Trainable Parameter Count and Memory Accounting

## Problem

Interviews often ask "how many trainable parameters does LoRA actually save, and which part is saved?". The answer: full fine-tuning updates the full-rank weight of every linear layer \(W_0\in\mathbb{R}^{d_\text{out}\times d_\text{in}}\), with \(d_\text{in}\cdot d_\text{out}\) parameters in total; LoRA freezes \(W_0\) and trains only two low-rank factors \(A\in\mathbb{R}^{r\times d_\text{in}}\) and \(B\in\mathbb{R}^{d_\text{out}\times r}\), with \(r(d_\text{in}+d_\text{out})\) in total. Given the dimensions of a set of linear layers and the rank \(r\), count the **trainable parameter count** on both sides and compute the compression ratio.

## Rules

The input `layer_dims` is a list whose elements are pairs \((d_\text{in}, d_\text{out})\); `r` is the LoRA rank (the same for all layers). Accumulate layer by layer:

- Full trainable parameters: \(\text{full} = \sum d_\text{in}\cdot d_\text{out}\)
- LoRA trainable parameters: \(\text{lora} = \sum r\cdot(d_\text{in}+d_\text{out})\)
- Compression ratio: \(\text{ratio} = \text{lora}/\text{full}\) (float, range \((0,1)\))

Return the triple `(full, lora, ratio)`, where `full` and `lora` are integers and `ratio` is a float. Count only the low-rank factors themselves, excluding the base \(W_0\), bias, optimizer state, etc.

## Examples

`layer_dims = [(1024,1024), (1024,4096), (4096,1024)]`, `r = 8`:

```
full = 1024*1024 + 1024*4096 + 4096*1024 = 9,437,184
lora = 8*(1024+1024) + 8*(1024+4096) + 8*(4096+1024) = 98,304
ratio = 98304 / 9437184 ≈ 0.010416667   (about 1.04%)
Output: (9437184, 98304, 0.010416666666666666)
```

## Constraints

- Dimensions and `r` are positive integers; `layer_dims` is non-empty.
- Parameter counts are exact integers, no approximation; `ratio` uses float division.
- Only when \(r\) is far smaller than \(\min(d_\text{in},d_\text{out})\) is `ratio` noticeably smaller than 1, which is exactly where LoRA's parameter savings come from.
