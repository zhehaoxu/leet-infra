# GQA / MQA Explained

## Three kinds of attention sharing the same set of KV heads

- **MHA**: $n_\text{head}$ query heads ↔ $n_\text{head}$ KV groups, the largest KV cache.
- **MQA**: $n_\text{head}$ query heads ↔ **1** KV group, the smallest KV cache but a noticeable quality drop.
- **GQA** (Mistral / Llama-2-70B / Gemma): $n_\text{head}$ query heads ↔ $g$ KV groups, $1 < g < n_\text{head}$; every $n_\text{head}/g$ **contiguous** query heads share the same KV group.

$$ h_i^{kv} = \left\lfloor \frac{i}{n_\text{head}/g} \right\rfloor,\quad i \in [0, n_\text{head}) $$

## Key pitfall: `repeat_interleave` vs `repeat`

When expanding KV along the head dimension to the number of query heads, the PyTorch semantics are:

- `kv.repeat_interleave(rep, dim=1)` → each kv head is **replicated contiguously rep times** (heads `[0,1]` belong to kv0, `[2,3]` to kv1).
- `kv.repeat(1, rep, 1, 1)` along the head dimension → cyclic concatenation (`[0]` belongs to kv0, `[1]` to kv1, `[2]` back to kv0).

GQA uses the **former**; the group-assignment test in the grader specifically perturbs the values of the two kv groups to catch wrong solutions that write "contiguous grouping" as "cyclic modulo".

## Degenerate relations to MHA / SDPA

- `num_kv_heads == num_heads` → must agree with standard multi-head attention (independent KV per head); useful as a regression check.
- `num_kv_heads == 1` → MQA, the most KV-cache-efficient (GPU memory is the main bottleneck for long-context inference).
- The scale must be $1/\sqrt{d_h}$ with $d_h = \text{last dim}/n_\text{head}$ (not the last dimension itself, and not the KV last dimension).

## Applications

Modern LLM inference services almost all use GQA: the PagedAttention cache in vLLM / TensorRT-LLM is allocated per **number of kv heads**; the smaller $g$, the less cache is used, allowing longer contexts or larger batches. Converting MHA weights to GQA is usually done by "merging / folding KV heads" (Ainslie 2023's upcasting or mean-pool).
