# KV Cache Update in Detail

## Background: the acceleration core of autoregressive inference

Generation decodes token by token. Recomputing the $K,V$ of the entire sequence at every step costs $O(t^2)$ with massive duplication. The KV Cache stores the $K,V$ of historical tokens, computes only the **new token**'s $K,V$ at each step and **appends** them to the tail of the cache, then attends over the full sequence. This trades "repeated computation" for "incremental storage" and is the main body of GPU memory management in LLM inference engines (vLLM, continuous batching).

## The operation in this level

Along the **sequence dimension** (usually `dim=-2`, the $L$ axis in shape $(B,h,L,d_k)$), concatenate the incoming `new_k` onto `cache_k`:

$$ \text{cache}_k \leftarrow \text{concat}([\text{cache}_k,\ \text{new}_k],\ \dim=L) $$

$K,V$ must be updated **in pairs, with matching dims**, and their lengths must always agree.

## Pitfalls

1. **The concatenation dimension is the sequence axis, not the feature axis**: `dim=-2` ($L$). Using `dim=-1` ($d_k$) by mistake widens the vectors and completely breaks attention.
2. **The cache is empty/None on the first step**: with no cache, return `new_k,new_v` directly (the solution here handles both kinds of "empty": `None` and `numel()==0`). Do not `cat` an empty tensor and raise a dimension error.
3. K and V may individually be present or absent (edge tests), so check each for emptiness separately.

## Advanced (perspective)

In real inference the cache is preallocated and written in place (avoiding a new allocation from `cat` at every step), with **PagedAttention** managing memory in blocks and reusing the cache. Prefill (compute the whole segment once into the cache) and decode (append step by step) are the two phases. This level is the minimal semantic core of that machinery.
