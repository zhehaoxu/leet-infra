# PEFT: LoHa Hadamard-Product Low-Rank Adaptation

## Problem

**LoHa** (Hadamard Product for Low-Rank Adaptation, from the FedPara family, Hyeon-Woo et al., arXiv:2108.04916; implemented in the PEFT library `tuners/loha`) is the "multiplicative twin" of LoRA: LoRA combines two low-rank factors into the weight **additively** (\(W' = W_0 + s\,BA\)), while LoHa multiplies two low-rank branches **element-wise (Hadamard product)**:

\[\Delta W = s\,\bigl(W_{1a}W_{1b}\bigr)\odot\bigl(W_{2a}W_{2b}\bigr),\qquad s=\frac{\alpha}{r}\]

where \(W_{1a},W_{2a}\in\mathbb R^{d_\text{out}\times r}\) and \(W_{1b},W_{2b}\in\mathbb R^{r\times d_\text{in}}\) (matching the shapes of the PEFT library's `hada_w1_a/hada_w1_b` etc. parameters), and \(\odot\) is the element-wise product. The point of the multiplicative combination: the Hadamard product of two rank-\(r\) factors can reach a **full-rank** \(d_\text{out}\times d_\text{in}\) update, while the parameter count is still only two copies of \(r(d_\text{out}+d_\text{in})\) — exactly how the FedPara paper approaches full fine-tuning at an equal parameter budget.

In the PEFT library's Linear implementation (`make_weight`), \(\Delta W\) is first rebuilt by the formula above and then merged back into the base: \(W' = W_0 + \Delta W\), so inference reduces to a single matmul.

## Task

Given the frozen base weight \(W_0\), LoHa's two factor pairs, the scaling parameter \(\alpha\), and the rank \(r\), return the **merged weight matrix** \(W' = W_0 + s\,(W_{1a}W_{1b})\odot(W_{2a}W_{2b})\) with \(s=\alpha/r\).

- Inputs \(W_0\in\mathbb R^{d_\text{out}\times d_\text{in}}\), \(W_{1a},W_{2a}\in\mathbb R^{d_\text{out}\times r}\), \(W_{1b},W_{2b}\in\mathbb R^{r\times d_\text{in}}\); \(\alpha\) is a real number and \(r\) a positive integer.
- Output \(W'\in\mathbb R^{d_\text{out}\times d_\text{in}}\).

**Convention**: do not use any ready-made PEFT/LyCORIS library; implement directly with tensor ops; results are judged under float32 semantics (relative/absolute tolerance \(10^{-5}\)).

## Hints

- First compute each branch's \(W_{ia}W_{ib}\), then multiply element-wise, and finally multiply by \(s\) and add to \(W_0\).
- Multiplication has an important degenerate case: **if any branch's factor is all zeros, \(\Delta W\) is all zeros** — the default init of PEFT/LyCORIS zeroes \(W_{2b}\) precisely so that the adapter "does nothing" at the training start.
