# Notes: The Softmax Fold-Back and GQA Merging in Attention Backward

## Origin

"Derive the backward of scaled-dot-product attention" is one of the most frequent hand-derivations for large-model framework roles (training kernels, inference engines, operator fusion); the appendix of the FlashAttention paper is exactly this identity. GQA head merging is a branch that any LLaMA-2/3 or Qwen implementation must handle.

## Where dS = P∘(dP − Σ P∘dP) Comes From

With \(O_i=\sum_j P_{ij}V_j\) and \(P_{i\cdot}\) the softmax of row \(i\), the softmax backward (see `ag-softmax-backward`) gives

$$\frac{\partial L}{\partial s_{ij}}=P_{ij}\Big(\frac{\partial L}{\partial P_{ij}}-\sum_k P_{ik}\frac{\partial L}{\partial P_{ik}}\Big),\qquad \frac{\partial L}{\partial P_{ij}}=dO_i\cdot V_j$$

and \(\partial L/\partial s\) must additionally be multiplied by the forward `scale` (since \(s_{ij}=\text{scale}\cdot q_i\cdot k_j+\text{mask}\)) to become a gradient w.r.t. \(QK^T\). Confusing the two (dropping `scale`) makes \(dQ,dK\) off by exactly \(1/\sqrt D\) while \(dV\) stays perfect — a handy way to localize the bug.

## Folding GQA Heads

The head axis of \(dK\) and \(dV\) has length \(H_{kv}\). For each kv head \(t\), sum the contributions of its \(H_q/H_{kv}\) query heads: \(dK^{(t)}=\sum_{h:\,h//g=t} dS_h^TQ_h\). Implementations either reshape to `(B,Hkv,g,Sk,D)` and reduce over `g`, or use `index_add`. **Forgetting the fold** produces `(B,Hq,Sk,D)`, which does not even match the oracle's shape; **swapping `groups` with `Hkv`** in the reshape gives the right shape but wrong numbers — the nastiest variant.

## Core Ideas and Pitfalls

- **`rowsum(P∘dP)` is not `rowsum(dP)`**: masked rows (where `P` has zeros) diverge immediately.
- **Masked entries vanish on their own**: \(P_{ij}=0\) forces \(dS_{ij}=0\), so no extra mask is needed, and applying the complement of the mask would instead clear legitimate entries.
- **`dV` never touches softmax**: \(dV=\sum_h P_h^T dO_h\), independent of `scale`.
- **Broadcast discipline**: with `Hkv` heads, select `h // groups` before multiplying with `dO` rather than relying on broadcasting.

## How It Differs from Neighboring Problems

`ag-flash-attn-backward` receives `O, LSE, dO` instead of `P` and must recompute \(P=e^{s-LSE}\); here `P` is handed over, so the difficulty sits on the softmax fold-back and head merging. Mask-construction exercises never differentiate attention at all.

## Cross-Check Basis and Complexity

The oracle builds `P = softmax(Q @ K_head^T * scale)`, `O = P @ V_head` under autograd and takes the three gradients of `loss = (O*dO).sum()`, atol=1e-9 in float64. The example is the scalar case of one head and one query so all four values are hand-computable. Complexity is \(O(BH_qS_qS_kD)\): two batched products plus one row reduction.
