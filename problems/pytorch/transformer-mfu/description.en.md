# Transformer FLOPs and MFU (Compute Utilization)

## Problem

**MFU (Model FLOPs Utilization)** measures the fraction of the hardware peak that training actually uses:
divide the theoretical FLOPs a training step "should" perform by the "measured wall time" to get the
achieved rate, then divide by the chip peak to get utilization; it is the core metric for evaluating the
efficiency of a training system. Using a standard Transformer training step, this problem estimates the
forward / total FLOPs and the MFU.

## Rules

- Implement `solve(b, s, layers, hidden, ffn, peak_tflops, step_ms)`, returning
  `(forward_flops, step_flops, mfu)`.
- **Parameters per layer that take part in matmuls** (excluding LayerNorm weights and the token embedding,
  in this simplified model):
  $$ P_{\text{layer}} = \underbrace{4h^2}_{\text{attn } q,k,v,o} + \underbrace{2 h f}_{\text{FFN up/down projections}} $$
- **Forward FLOPs per token per layer** (one multiply-add counts as 2 FLOPs):
  $$ c = 2\,P_{\text{layer}} + \underbrace{4\,s\,h}_{QK^\top \text{ and } AV \text{ non-parameter FLOPs}} $$
  where $4sh$ = $2sh$ for $QK^\top$ + $2sh$ for $A V$ (after summing over all heads, each token runs an
  $h$-dimensional operation against the layer's $s$ keys/values).
- **Forward FLOPs of one step** (tokens $= b\cdot s$):
  $$ \text{forward\_flops} = b\cdot s\cdot \text{layers}\cdot c $$
- **Total step FLOPs**: backward is about 2× forward, hence
  $$ \text{step\_flops} = 3 \times \text{forward\_flops} $$
  (consistent with the familiar "training $\approx 6ND$" accounting: $2N$ forward and $4N$ backward per
  token).
- **MFU**: `peak_tflops` is the single-chip peak (TFLOP/s) and `step_ms` is the measured time of that step
  (milliseconds):
  $$ \text{achieved} = \frac{\text{step\_flops}}{\text{step\_ms}/1000}\ \text{FLOP/s},\qquad \text{mfu} = \frac{\text{achieved}/10^{12}}{\text{peak\_tflops}} $$

## Example

```
Input: b=4, s=1024, layers=6, hidden=512, ffn=2048, peak_tflops=150.0, step_ms=8.0
  P_layer = 4*512² + 2*512*2048 = 3,145,728
  c = 2*3,145,728 + 4*1024*512 = 8,388,608   (per token per layer)
  forward_flops = 4*1024*6*8,388,608 = 206,158,430,208
  step_flops    = 3*forward = 618,475,290,624
  achieved = 618475290624 / 0.008 = 7.7309e13 FLOP/s = 77.309 TFLOP/s
  mfu = 77.309 / 150.0 = 0.5154
Output: (206158430208, 618475290624, 0.5154...)
```

## Constraints

- `b, s, layers, hidden, ffn` are positive integers; `peak_tflops`, `step_ms` are positive floats
- `forward_flops`, `step_flops` are returned as integers; `mfu` is a float ratio (on the order of 0~1)
- This simplified model ignores non-matmul FLOPs such as embedding/LayerNorm/activation functions (a small
  share) and focuses on the mainstream accounting
