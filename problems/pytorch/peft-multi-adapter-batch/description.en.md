# PEFT: Multi-Adapter In-Batch Hot-Swap Routing

## Problem

The core primitive of multi-LoRA serving (S-LoRA, OSDI 2024, arXiv:2311.03285; punica, arXiv:2310.18547): one frozen base, and within a single batch different requests attach **different adapters**, computed in one forward pass — no model copy per adapter, no per-request serialization. In-batch routing: request \(i\) uses adapter \(k_i\); the base output is shared by everyone while the low-rank increment picks each row's own factors and scale:

\[y_i=W_0x_i+s_{k_i}\,B_{k_i}A_{k_i}x_i\]

(\(A_k\in\mathbb R^{r\times d_\text{in}}\) down-projection, \(B_k\in\mathbb R^{d_\text{out}\times r}\) up-projection, \(s_k\) the adapter's scale — merging/weighting multiple adapters gives \(s\ne1\).) The systems literature calls the corresponding batched kernel SGMV (segmented gather-matrix-vector): GEMV segmented by adapter within the batch.

## Task

Implement adapter-routed batched forward.

- Inputs: \(W_0\in\mathbb R^{d_\text{out}\times d_\text{in}}\); \(A\in\mathbb R^{K\times r\times d_\text{in}}\), \(B\in\mathbb R^{K\times d_\text{out}\times r}\) (\(K\) adapters); \(\texttt{scales}\in\mathbb R^{K}\); \(\texttt{ids}\in\mathbb Z^{N}\) (per-sample adapter index, \(0\le k<K\)); \(x\in\mathbb R^{N\times d_\text{in}}\).
- Output \(y\in\mathbb R^{N\times d_\text{out}}\) (float32).

**Convention**: compute in float64 internally, output float32; no ready-made PEFT/vLLM library; only the forward composition (memory and scheduling accounts belong to the collection's serving problems). Tolerance \(10^{-5}\).

## Hints

- The base part is adapter-independent: compute \(xW_0^\top\) first, then accumulate increments grouped by id.
- With all-zero scales it must degenerate exactly to \(xW_0^\top\) (the shared-base semantics).
- With identical ids it must agree bit-for-bit with the single-adapter \(W_0x+s\,BAx\).
