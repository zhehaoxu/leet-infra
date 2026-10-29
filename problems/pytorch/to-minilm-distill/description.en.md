# MiniLM: Self-Attention and Relation Distillation

## Problem

**MiniLM** (Wang et al. 2020, *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*, arXiv:2002.10957, NeurIPS 2020) is a classic "how a small model learns from a large one" paper. Its central claim: distilling only the final logits is not enough — the **self-attention distribution is itself knowledge**, and each teacher head's pattern of "which tokens it attends to" deserves to be transferred head by head. It proposes two losses:

1. **Self-attention distribution KL**: pull the teacher's and student's attention distributions for the same head together, one term per head and per query position;
2. **Relation-representation MSE**: element-wise mean squared error between several token-pair relation vectors (e.g. CLS·CLS, CLS·token, token·token scaled-dot-product representations).

The total is the sum of the two. This problem gives the attention scores **before softmax** plus three relation tensors; recover both terms.

## Rules

Given batch size $B$, head count $H$, sequence length $L$:

1. **Attention distribution**: divide the scores by temperature $\tau$, then take a masked softmax over the **key dimension (last)**. `mask=False` keys must have probability **exactly 0** (fill with $-\infty$ then `log_softmax`, or an equivalent masking).
2. **Per-(b, head, query) KL**:
   $$\mathrm{KL}\big(p_t\,\|\,p_s\big)=\sum_j p_t(j)\big(\log p_t(j)-\log p_s(j)\big),$$
   each side normalized by its own softmax, over the **same** mask.
3. **Attention loss** = sum of KLs over all **valid queries** and all heads, divided by (number of valid queries × $H$). A `mask=False` query row enters neither the numerator nor the denominator.
4. **Relation loss** = squared differences of the three relation tensors, averaged over all $B\times3\times D$ elements.
5. **Total loss** = attention loss + relation loss (weights 1:1).

Return three 0-dimensional float64 scalars: `attn_kl`, `cls_mse`, `loss`.

## Examples

```
teacher_attn[0,0] = [[1.0, 0.0], [0.0, 1.0]]   student_attn[0,0] = [[0.0, 1.0], [1.0, 0.0]]
teacher_attn[0,1] = [[0.5, 0.5], [1.0, 0.0]]   student_attn[0,1] = [[0.0, 0.5], [0.5, 0.0]]
mask = [[True, True]]      τ = 1.0

teacher_cls = [[[1.0], [0.0], [-1.0]]]
student_cls = [[[0.0], [0.0], [ 1.0]]]

attn_kl = 0.245377      # mean of the 4 (query, head) KLs
cls_mse = 1.666667      # (1² + 0² + 2²) / 3
loss    = 1.912044
```

On both queries of head 0 the teacher's and student's attention distributions are exact mirror images, each contributing $\ln 2$; head 1 differs less, so the four-term average lands near 0.245.

## Constraints

- All three outputs are 0-dimensional float64; inputs may be float32 and are promoted internally.
- `mask` applies to both queries and keys: masked keys have probability exactly 0; masked query rows are excluded from the average.
- The KL direction is teacher first ($p_t\|p_s$), and the squared term is `(teacher - student)²`; reversing either yields different numbers.
- Invalid input raises `ValueError`: non-tensor / attention not 4-D or not square / non-float / NaN·Inf / shape mismatch / mask not bool or mismatched / mask all False / relation tensor not $(B,3,D)$ or $B$ mismatch / $D<1$ / $\tau\le0$.
- CPU, deterministic, no random number calls.
