# LoRA Trainable Parameter Count and Memory Accounting Explained

## Source
Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (arXiv:2106.09685). The core selling point is reducing the trainable amount of each weight matrix from \(d_\text{in}\cdot d_\text{out}\) to \(r(d_\text{in}+d_\text{out})\), with \(r\) commonly 4/8/16, far smaller than the hidden dimension.

## Core points and pitfalls
1. **Full rank vs low rank**: \(W_0\)'s parameter count is the "area" \(d_\text{in}\cdot d_\text{out}\); LoRA is two "edge lengths" times \(r\): \(A:r\cdot d_\text{in}\), \(B:d_\text{out}\cdot r\), totaling \(r(d_\text{in}+d_\text{out})\).
2. **What is saved is trainable parameters**: the base weights are still needed **for the forward pass** (occupying GPU memory); LoRA reduces the parameters that must be updated / have optimizer state stored, not the total weight memory. This problem counts only the "trainable" sides.
3. **The compression ratio grows with dimensions**: the single-layer ratio is \(=r(d_\text{in}+d_\text{out})/(d_\text{in}d_\text{out})\), approaching 0 as the hidden dimension grows, so LoRA is relatively more economical on large models.
4. **Multi-layer accumulation**: different layers have different dimensions; sum layer by layer and then divide to get the global ratio, and do not average the per-layer ratios.

## Grading details
The reference solution simply accumulates in two passes: `full += din*dout`, `lora += r*(din+dout)`. The grader's `_ref` sums separately with list comprehensions (a different structure), compares `full`/`lora` with integer `==`, and `ratio` with `abs(...-...) < 1e-9`. Additional cases: a single identity layer (no saving when \(r=\min\)), the \(r=1\) lower bound, and the fact that the global ratio is the ratio of the sums rather than the per-layer mean.

## Complexity
\(O(L)\) over \(L\) layers, pure integer arithmetic, no tensors.
