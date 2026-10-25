# FLOPs Estimation and MFU In Depth

## 6ND: the empirical formula for training compute

The most famous rule in large-model training throughput estimation is **$\text{flops} \approx 6ND$**
($N$ = parameter count, $D$ = token count): each parameter does one multiply-add in the forward pass per
token (counted as 2 FLOPs) → $2N$; the gradient computation in backward is about 2× the forward → $4N$;
together $6N$ per token. This problem unpacks that rule using "matmul parameter count per layer
$P_{\text{layer}}$", which is closer to the real structure (attention/FFN separated), and additionally
covers **parameter-free** FLOPs such as $QK^\top$ and $AV$.

## Why attention's $4sh$ deserves its own term

$2P_{\text{layer}}$ only covers the "activation × weight" projection matmuls; $QK^\top$ and $AV$ are
**activation × activation** with no parameters involved, so they are easy to leave out. They add up
**linearly** in sequence length $s$ (each query scans $s$ keys), and their share grows with long contexts —
one of the reasons optimizations such as FlashAttention pay off.

## MFU vs HFU

- **MFU** (Model FLOPs Utilization): counts only the FLOPs that "must be done by definition of the model",
  divided by peak compute. Gradient checkpointing's **recomputation FLOPs are not counted in the
  numerator**, so enabling checkpointing lowers MFU (more real computation in exchange for memory).
- **HFU** (Hardware FLOPs Utilization): counts recomputation in the numerator too, measuring how "compute
  saturated" the hardware is. The gap between the two is exactly the recomputation overhead, and in
  practice both are looked at together. This problem asks for MFU.

## Judging methodology and traps

- One multiply-add = 2 FLOPs (multiply-accumulate); do not count it as 1.
- `step_ms` is in milliseconds, so converting to FLOP/s requires `/1000`; achieved is then `/1e12` to get
  TFLOP/s before comparing with `peak_tflops`.
- FLOPs of embedding / LayerNorm / activation functions / dropout are ignored in mainstream estimates
  (small share, and many are memory-bound).
- In reality an MFU of 0.4~0.6 is already excellent (the PaLM report gives roughly 0.5~0.7 for GPT-style
  models); do not expect anything close to 1.
