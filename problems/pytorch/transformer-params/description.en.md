# Transformer Parameter Count

## Problem

Interviews often ask you to "estimate how many parameters a model has from its structure, without looking
at code" (where does the parameter count of GPT-2/GPT-3 come from). Following the architecture of a standard
decoder-only Transformer, this problem counts the parameters exactly from the per-layer dimensions.

## Rules

- Implement `solve(layers, hidden, ffn, vocab, tied_embedding=True)`, returning the total parameter count
  (integer).
- **Parameter count of each Transformer block** (matrices counted as `in×out`, bias included, positional
  encoding excluded):
  $$ P_{\text{layer}} = \underbrace{4h^2}_{\text{attn } q,k,v,o\ (W_{\text{proj}}: h\to h)} + \underbrace{2hf}_{\text{FFN } h\to f,\ f\to h} + \underbrace{4h}_{\text{2× LayerNorm (weight+bias, } h \text{ each)}} $$
- The **token embedding** `vocab × hidden` is counted once; the output LM head is also `vocab × hidden`:
  - `tied_embedding=True`: input/output share the same embedding table, so `vocab·h` is counted only once;
  - `tied_embedding=False`: one more `vocab·h` is added (the unshared output projection).
- Add one **final LayerNorm** at the end: $2h$ (weight+bias).
- Total:
  $$ \text{total} = \text{layers}\cdot P_{\text{layer}} + \text{vocab}\cdot h + (\text{tied?}\,0:\text{vocab}\cdot h) + 2h $$
- The number of heads `heads` does not change the parameter count (MHA's $q,k,v,o$ are always four
  $h\times h$ matrices), so this problem does not need `heads`.

## Example

```
Input: layers=12, hidden=768, ffn=3072, vocab=50257, tied_embedding=True
  P_layer = 4*768² + 2*768*3072 + 4*768 = 2,359,296 + 4,718,592 + 3,072 = 7,080,960
  layers term = 12 * 7,080,960 = 84,971,520
  embedding term = 50257 * 768 = 38,597,376
  final LN   = 1,536
  total     = 84,971,520 + 38,597,376 + 1,536 = 123,570,432   (≈123.6M)
Output: 123570432
```

## Constraints

- All inputs are positive integers; `tied_embedding` is a boolean
- This model has **no learnable positional embedding** (`max_seq × h`) and no parameter-free items such as
  dropout; add them yourself if you want them, the accounting here is fixed
- Return an **integer** parameter count (Python int)
