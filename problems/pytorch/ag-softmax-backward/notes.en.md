# Notes: softmax Backward: Gradient w.r.t. logits from Output and Grad-Output

## Origin

PyTorch's `aten::_softmax_backward_data` / `_log_softmax_backward_data` (in `SoftMax.cu`, `SoftMax.cpp`) implement exactly these two branches; Goodfellow's *Deep Learning* 6.2.2 derives the softmax-plus-cross-entropy combination, and the FlashAttention paper (Dao 2022, 2205.14135) writes step 3 of its backward as "rebuild without storing P". "Derive the softmax gradient" is a staple whiteboard question.

## From the Jacobian to One Dot Product

Forward \(y_i = e^{x_i}/\sum_k e^{x_k}\), differentiated w.r.t. \(x_j\):

- \(i=j\): \(\partial y_i/\partial x_i = y_i(1-y_i)\)
- \(i\ne j\): \(\partial y_i/\partial x_j = -y_i y_j\)

Together \(\partial y_i/\partial x_j = y_i(\delta_{ij}-y_j)\). The chain rule gives

$$dx_j=\sum_i grad_i\,y_i(\delta_{ij}-y_j)=y_j\Big(grad_j-\sum_i y_i\,grad_i\Big)$$

and \(\sum_i y_i grad_i\) is precisely "grad dotted with y along the normalized axis": one scalar per row. So the whole backward is a single `sum` plus a single elementwise multiply, dropping the cost from \(O(n^2)\) to \(O(n)\).

The log_softmax branch differs: the forward is \(y_i=x_i-\log\sum_k e^{x_k}\), so \(\partial y_i/\partial x_j=\delta_{ij}-p_j\) with \(p=\exp y\), hence \(dx_j = grad_j - p_j\sum_i grad_i\) — what gets subtracted is the **total of grad** scaled by the probability, not the dot product of y and grad.

## Core Ideas and Pitfalls

- The reduction axis must follow `dim`, kept with `keepdim=True` (or reshape) so broadcasting does not hit the wrong axis; middle axes (e.g. `dim=1` on (B,C,N)) silently degrade into last-axis reductions.
- `y*(grad - (grad*y).sum(dim))` and `y*grad - y*(y*grad).sum(dim)` are equivalent; forgetting the outer \(y_j\) is the classic slip and scales the answer wrongly by one probability.
- Translation invariance is a strong self-check: a constant grad along dim forces the result to zero because softmax is unchanged by adding a constant to the logits.
- Using \(e^{y_i}\) where the formula needs \(y_i\) (or vice versa) on the log_softmax branch is the closest-looking mistake here: the two differ only by an exponential, so every entry fails the cross-check.
- Divisors such as \(1/C\) and "number of valid rows" belong to ag-cross-entropy-backward, not to this problem.

## How It Differs from Neighboring Problems

- vs ag-cross-entropy-backward: that one is the merged "(p−onehot)/n" result needing no upstream grad; this keeps an arbitrary grad.
- vs ag-attention-qkv-backward / ag-flash-attn-backward: those embed this formula into the matrix backward of \(QK^\top\) (the former stores P, the latter rebuilds it from LSE and D=rowsum(dO∘O)).
- vs the softmax / log-softmax problems in the operators collection: those are about the forward and numerical stability (subtracting the max); this one is purely backward.

## Cross-Check Basis and Complexity

Judging re-implements the forward inside the test file (`torch.softmax` / `torch.log_softmax`) and takes `torch.autograd.grad` as the oracle: structurally different from the reference solution but numerically identical, atol=1e-9 for float64 cases. There are dedicated assertions on "grad all ones ⇒ output all zeros" plus cross-validation of both `log_input` branches. Time \(O(N)\) (one reduction plus one elementwise pass), extra space one keepdim scalar.
