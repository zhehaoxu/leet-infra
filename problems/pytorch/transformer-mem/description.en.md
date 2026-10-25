# Transformer Activation Memory (with Gradient Checkpointing)

## Problem

During training, backpropagation needs the **activations** saved by the forward pass, which is one of the
main memory costs of large models (alongside parameters, gradients, and optimizer state). **Gradient
checkpointing (activation checkpointing / recomputation)** trades compute for memory by "storing only each
layer's input and recomputing the intermediate activations during backward", and is a standard technique for
training deep Transformers. Using a **simplified teaching model**, this problem estimates the activation
memory of a whole model: the byte counts with and without checkpointing.

## Rules

- Implement `solve(b, s, layers, hidden, ffn, heads, dtype_bytes=2)`, returning
  `(full_bytes, checkpoint_bytes)`.
- **Activations that each layer's forward pass must save** (the tensor list prescribed by this simplified
  model, in elements):

  | Tensor | Shape | Elements |
  |---|---|---|
  | Q, K, V projection outputs | each `(b,s,hidden)` | $3\,b\,s\,h$ |
  | Attention output projection (before the residual) | `(b,s,hidden)` | $b\,s\,h$ |
  | Inputs of the two residual/LayerNorm sites | `(b,s,hidden)` | $b\,s\,h$ |
  | Two FFN intermediates (fc1 out, act out) | `(b,s,ffn)` | $2\,b\,s\,f$ |
  | Attention probability matrix | `(b,heads,s,s)` | $b\,a\,s^2$ |

  Collecting like terms gives the full per-layer footprint:
  $$ E_{\text{full}} = b\,s\,(6h + 2f) + b\,a\,s^2 $$
- **Without checkpointing**: activations of all `layers` layers are retained:
  $$ \text{full\_bytes} = \text{layers}\cdot E_{\text{full}}\cdot \text{dtype\_bytes} $$
- **With checkpointing**: only each layer's **input** tensor $(b,s,h)$ stays resident, and when backward
  reaches a layer its full activations are recomputed (the peak additionally holds one layer's full
  activations):
  $$ \text{checkpoint\_bytes} = \big(\text{layers}\cdot b\,s\,h + E_{\text{full}}\big)\cdot \text{dtype\_bytes} $$
- Return integer byte counts.

## Example

```
Input: b=4, s=1024, layers=6, hidden=512, ffn=2048, heads=8, dtype_bytes=2(fp16)
  E_full = 4*1024*(6*512+2*2048) + 4*8*1024*1024 = 62,914,560 elements
  full_bytes      = 6 * 62,914,560 * 2 = 754,974,720   (= 720 MiB)
  checkpoint_bytes= (6*4*1024*512 + 62,914,560)*2 = 150,994,944 (= 144 MiB)
Output: (754974720, 150994944)   -- checkpointing cuts activation memory to about 1/5
```

## Constraints

- All inputs are positive integers; `dtype_bytes` is commonly 2 (fp16/bf16) or 4 (fp32)
- This is a **simplified teaching model** (it excludes parameter/gradient/optimizer-state memory and
  estimates attention via the probability matrix); it only tests your understanding of the order of
  magnitude of activations and of checkpointing's "store the input + recompute one layer", and does not aim
  to match any framework profiler exactly
- Return **integer** byte counts (use Python int, do not lose precision)
