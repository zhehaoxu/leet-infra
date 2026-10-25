# Solution Notes · MoE Top-k Gating

## Why softmax first, then topk

Some implementations do "topk first, then softmax over the logits"; the final scores of the two orders are **different**:
- softmax then renorm (this implementation): the renorm denominator is just the sum of the $k$ selected probabilities, $\sum_j s_j = 1$.
- topk then softmax: softmax the $k$ raw logits directly, which is equivalent to a biased allocation within the selected set.

HF `MixtralTopKRouter.forward` uses **softmax first, then topk, then renorm**, and grading follows this path.

## tie-break

`torch.topk` does **not guarantee a stable order** for equal values (it may differ across backends / versions). The test cases avoid strictly tied logits; if ties must be handled, use "smaller expert index first":
```
order = sorted(range(E), key=lambda e: (-float(probs[t,e]), e))
```

## Meaning of renorm

After top-k, the sum of the $k$ probabilities is $<1$ (the long tail is discarded). Without renorm, using them directly as combine weights would make the MoE output **systematically smaller** than the dense baseline; renorm guarantees that each token's final weighted sum is still equivalent to "a weighted average over the selected $k$ experts".

## Difference from softmax-with-temp

A dense FFN uses softmax because "all neurons are computed"; MoE only uses the $k$ activated neurons, so the softmax merely provides **an initial score**, and the real weights are the $k$-dimensional distribution after renorm. Therefore the output of top-k gating **always sums to 1 per row**.
