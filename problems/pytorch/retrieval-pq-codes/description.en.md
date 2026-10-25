# Retrieval: Product Quantization Encoding

## Problem

Product Quantization (PQ, Jégou et al., TPAMI 2011) **decomposes** the \(D\)-dimensional space into the **Cartesian product** of \(m\) subspaces of dimension \(d=D/m\), running vector quantization independently in each subspace with its own codebook of \(k_{sub}\) centers. A vector is thereby compressed into \(m\) "sub-codes" (about \(\log_2 k_{sub}\) bits each), far smaller than the original \(D\)-dimensional float. This is the cornerstone of reranking and compression in Faiss `IndexPQ` / IVF-PQ. This problem implements the deterministic **encoding** step of PQ.

## Rules

Implement `solve(x, codebook)`. The input `x` (float32, shape \((n, D)\)) and `codebook` (float32, shape \((m, k_{sub}, d)\)) holds \(m\) sub-codebooks (\(k_{sub}\) centers of dimension \(d\) each), with \(D = m\cdot d\). For each vector \(x_i\), split it along the feature dim into \(m\) segments \(x_i[j\!\cdot\!d : (j\!+\!1)\!\cdot\!d]\); the sub-code of segment \(j\) is the index of the center with the smallest squared Euclidean distance to that sub-vector in the **\(j\)-th codebook**. Return an int64 tensor of shape \((n, m)\). Ties go to the smallest center index.

## Examples

\(m=2, k_{sub}=2, d=2\) (hence \(D=4\)):

```
codebook[0]=[[0,0],[10,10]]   # the two centers of segment 0 (first 2 dims)
codebook[1]=[[0,0],[0,5]]     # the two centers of segment 1 (last 2 dims)
x = [[0,0, 0,0],             # seg0=[0,0]->0, seg1=[0,0]->0
     [3,3, 0,6]]             # seg0=[3,3]->0(dist 18<98), seg1=[0,6]->1(dist 1<36)
out = [[0,0],[0,1]]   shape = (2, 2)
```

## Constraints

- CPU float32; \(D = m\cdot d\) must divide exactly.
- The subspaces are **independent**: segment \(j\) is only compared with the \(j\)-th codebook, never mixing codebooks across segments.
- The squared Euclidean distance is enough; ties go to the smallest sub-code index.
