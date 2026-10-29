# Notes: Why the Backward Only Needs O and LSE

## Origin

The FlashAttention backward (Dao et al., 2022) is a staple follow-up for training/inference framework roles: "if you cannot say which buffer was saved, you have not read the paper". The standard phrasing is: the forward never stores \(P\), so how does the backward push \(dP\) through softmax? Producing the line \(D=\text{rowsum}(dO\circ O)\) is the pass condition.

## Rewriting the Softmax Backward with Only O and dO

A naive backward computes \(dS=P\circ(dP-\sum_k P_k dP_k)\), and that row reduction collapses: \(\sum_k P_{ij}(dO_i\cdot V_j)=dO_i\cdot\sum_j P_{ij}V_j=dO_i\cdot O_i\). In other words the needed scalar is **exactly the dot product of the forward output with dO**:

$$D_i=\sum_d dO_{id}O_{id},\qquad dS_{ij}=P_{ij}(dP_{ij}-D_i)\cdot scale$$

The backward then only recomputes \(P\) (from \(S+mask-LSE\), tile by tile, never storing the whole matrix) and yields all three gradients. The role of \(LSE\) is to make the exponential already normalized: subtracting only the row max would still require dividing by \(\sum\exp\), and that sum has already been absorbed into \(LSE\).

## Core Ideas and Pitfalls

- **`mask` must enter both \(S\) and \(LSE\)**: the forward `LSE` is a masked logsumexp, so recomputation needs \(S+mask-LSE\); applying the mask on one side only breaks row normalization and a causal case fails instantly.
- **`D` is `dO·O`, not `dO·constant`**: the test pins this down with the paired assertion "shifting \(O\) changes nothing when rows of \(dO\) sum to zero, but must change the result when \(dO\) is all ones".
- **Adding \(c\) to `LSE` alone** scales \(P\) by \(e^{-c}\) and hence all three gradients — the cheapest probe that normalization went through the right path.
- **Range safety**: \(S+mask-LSE\le 0\) by construction and masked entries give `exp(-inf)=0`; subtracting before exponentiating prevents overflow.

## How It Differs from Neighboring Problems

`ag-attention-qkv-backward` hands you \(P\), so nothing must be recomputed and no \(D\) is needed (normalization is hidden inside \(P\)); here \(P\) is withheld and only \(O,LSE\) are given, which is precisely the equivalence above. Causal scheduling and blocking memory policies are out of scope.

## Cross-Check Basis and Complexity

The oracle uses a test-local forward `softmax((Q@Kᵀ)*scale+mask) @ V` with `torch.autograd.grad`, atol=1e-9 in float64; the example has \(S=[0,0]\) and \(LSE=\log 2\) so every quantity is an exact decimal multiple of \(0.25\). Asymptotic work stays \(O(n^2d)\) (this reference is not truly tiled), but peak extra memory drops from a \(B H n^2\) copy of \(P\) to \(B H n\) for \(LSE\) and \(D\).
