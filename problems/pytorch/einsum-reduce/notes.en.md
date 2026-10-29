# Explained: Einsum Spec Evaluator (Label String to Multiply-Reduce)

## Origin

Einstein notation reached array libraries as `np.einsum` (added in 2011) and PyTorch exposes the same idea as `torch.einsum`. Interview questions usually go the easy direction ("hand-write `QK^T` for batches without einsum"); this problem reverses it — you get the label string and must resolve axis alignment and the contraction set yourself, which simultaneously covers matmul, transpose, outer product and per-axis sums.

## Full Derivation and Index Conventions

Split the spec into input labels $L_1,\dots,L_n$ and output label $M$. Define

- the contraction set $\mathcal{C} = \bigcup_k L_k \setminus M$ (ordered by first appearance);
- the output letter list $M$ (guaranteed duplicate-free and present in the inputs).

Use the unified axis order `order = C ++ M`. For operand $k$ the letters it owns inside
that order are $\text{kept}_k = [c \in \text{order} \mid c \in L_k]$ with original dim indices
`perm_k = [L_k.index(c) for c in kept_k]`. Since order covers every letter of $L_k$,
`perm_k` is exactly a permutation of $\{0,\dots,\dim(T_k)-1\}$, so

$$T'_k = \text{permute}(T_k;\text{perm}_k)\ \text{reshape}\ \text{shape}[i] = \begin{cases} \text{axis length of that letter in } L_k, & c \in L_k\\ 1, & c \notin L_k \end{cases}$$

After that, an element-wise product lets broadcasting do all three jobs at once
(outer product / dot product / batch alignment):

$$P[\text{order}] = \prod_k T'_k[\text{order}], \qquad \text{out}[M] = \sum_{\mathcal{C}} P$$

Because order puts the contracted letters **first** and the output letters in $M$'s order
afterwards, summing away the leading $|\mathcal{C}|$ axes leaves axes already in the order of
$M$ — **no final permute is needed**, which is the most commonly over-written step here.

For `"ij,jk->ik"`: $M=\{i,k\}$, $\mathcal{C}=\{j\}$, order $=[j,i,k]$; $X$ has kept $=[i,j]$ →
perm $=[1,0]$ → shape $(2,3,1)$, $Y$ has kept $=[j,k]$ → perm $=[0,1]$ → shape $(2,1,3)$;
the product is $(2,3,3)$ and summing axis 0 gives $(3,3)$ ✓.

## Core Ideas and Pitfalls

1. **Output axis order**: `"ij->ji"` versus `"ij->ij"` differ only in the tail of order; an
   implementation that "sorts letters alphabetically before reducing" turns the transpose
   into an identity.
2. **Missing letters become length 1, they are not deleted**: in `"i,j->ij"` the first operand
   has no $j$ axis, so it must reshape to $(3,1)$ and the second to $(1,4)$, broadcasting to
   $(3,4)$. `expand` works too, but `reshape` on the permuted non-contiguous view implicitly
   copies while keeping the semantics correct.
3. **Fully contracted output is 0-dimensional**: `"i,i->"` must give `out.dim() == 0`;
   returning a Python float fails the shape assertions.
4. **No diagonal/trace**: `"ii->i"` would need `torch.diagonal`, which the statement places
   outside the subset and requires to raise ValueError — implementations that treat it as
   legal input inevitably fail here.
5. **Precision**: multiply-accumulate in float64 and cast back to float32; against
   `torch.einsum`'s float32 accumulation with $k \le 6$ the difference is far below 1e-5.

## Difference from Neighbouring Problems

- `batched-gemm`: only the broadcast semantics of `(...,n,m)@(...,m,p)` with implicit axes;
  here axes are named by the string and may be permuted and contracted freely.
- `concat-stack-split`: pure shape manipulation, no multiply-add.
- `scaled-dot-product-attention`, `grouped-query-attention`: einsum is a *means* there,
  while it is the *object* here.

## Cross-Check Basis and Complexity

- The judge oracle is literally `torch.einsum` (pointwise equal inside the restricted subset),
  plus structural assertions (shape, dtype, 0-dim) and ValueError probes over seven illegal specs.
- Cost: the product tensor has $\prod_{c\in\text{order}}|c|$ entries, more memory-hungry than
  contracting pairwise, but the judge uses at most 6 operands of length ≤ 6, so each case runs
  in milliseconds on CPU.
