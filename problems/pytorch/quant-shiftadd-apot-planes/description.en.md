# Quantization: ShiftAddLLM Bit-Plane Scales as Additive Powers of Two

## Problem

Multiplication is the expensive operator. ShiftAddLLM (Park et al., arXiv:2406.05981, NeurIPS 2024) takes the "multiplication-less reparameterization" route: the weight matrix is first decomposed, BCQ style, into **bit planes** - one sign matrix per plane times one scalar per plane - and then each scalar is compressed into a **short signed sum of powers of two** (ApOT, additive powers of two). Together, one matrix product turns into "select planes by sign, shift, add".

This exercise covers the deterministic part of that chain: flatten and group -> greedy bit planes -> ApOT-decompose each plane's scale -> reconstruct with the decomposed scales. The upstream 15-round CG/BST refinement, the CUDA kernels and the `q_bias`/`scale` branches are out of scope. The difficulty is not the maths, it is the four conventions in the upstream code that are easy to get wrong: `torch.round` is **half-to-even**, not floor; `log(0) = -inf` combined with `sign(0) = 0` leaves a zero term at 0; `view([-1, group_size])` groups the **row-major flattening**, so a group can straddle rows; and reconstruction uses the ApOT scale, not the raw one.

## Rules

Let the weight be $W\in\mathbb{R}^{R\times C}$ and let $n$ be the number of groups. Everything is float64 internally and returned in the input dtype.

1. **Grouping**: $g=C$ when `group_size <= 0` (including `-1`), otherwise $g=\texttt{group\_size}$. Flatten $W$ in **row-major** order into $R\cdot C$ elements and cut it into $n=R\cdot C/g$ groups of $g$ elements. $g$ must divide $R\cdot C$; when $g$ does not divide $C$ a group straddles a row boundary, which is the upstream behaviour and not an error.
2. **Greedy bit planes** (`greedy_mean_torch` with the weight mask `wf` all ones): $r_0=W_{\text{group}}$, and for $i=0..B-1$ with $B=\texttt{num\_bits}$
   $$b_i=\operatorname{sign}(r_i),\qquad \alpha_i=\frac{\sum_j |r_{i,j}|}{g},\qquad r_{i+1}=r_i-b_i\,\alpha_i .$$
   `planes[:, :, i] = b_i` and `alpha[:, i] = α_i`. $b_i\in\{+1,-1,0\}$; once an element reaches 0, $\operatorname{sign}(0)=0$ keeps every later plane at 0 for that element.
3. **ApOT decomposition** (`additive_power_of_2`, applied to each $\alpha$ independently, $S=\texttt{apot\_nums}$):
   $$t_0=2^{\operatorname{round}(\log_2|\alpha|)}\cdot\operatorname{sign}(\alpha),\qquad
     t_k=2^{\operatorname{round}(\log_2|d_k|)}\cdot\operatorname{sign}(d_k),\quad d_k=\alpha-\sum_{m<k}t_m .$$
   Here $\operatorname{round}$ is `torch.round` (nearest integer, ties to even), not floor: $\log_2 3=1.585\to 2$, $\log_2 0.4=-1.322\to -1$. When $\alpha=0$ or the residual is exactly 0, $\log 0=-\infty$, $2^{-\infty}=0$ and $\operatorname{sign}(0)=0$, so $t_k=0$. `alpha_apot` $=\sum_{k<S}t_k$ and `apot_terms` stores $t_0..t_{S-1}$ in submission order.
4. **Reconstruction**: $\hat W_{g,j}=\sum_i \texttt{planes}[g,j,i]\cdot\texttt{alpha\_apot}[g,i]$ (the upstream `einsum('ijl,il->ij')`), folded back to $(R,C)$ in row-major order; $\texttt{mse}=\operatorname{mean}\bigl((W-\hat W)^2\bigr)$ over all $R\cdot C$ elements.

Return a dict with exactly 7 keys:

- `group_shape` tuple[int]: $(n, g)$;
- `planes` float $(n, g, B)$: entries are only $-1/0/+1$, axis 2 is the bit-plane index;
- `alpha` float $(n, B)$: the raw greedy bit-plane scales;
- `alpha_apot` float $(n, B)$: the scales after ApOT decomposition;
- `apot_terms` float $(n, B, S)$: the individual powers of two for each scale, in submission order;
- `recon` float $(R, C)$: the weight reconstructed from `alpha_apot`;
- `mse` float 0-dim: MSE of $W$ against `recon`.

## Task

- Input: `weight`, a float32 or float64 $(R,C)$ tensor (a non-contiguous tensor is read in row-major order); `num_bits` and `apot_nums` are ints $\ge 1$; `group_size` is an int and degenerates to one group per row when $\le 0$.
- Output: the 7-key dict above; float tolerance `1e-5` for float32 and `1e-9` for float64; `group_shape` compared as a Python tuple.
- Raise `ValueError` for: `weight` not 2-D / zero elements / dtype other than float32 or float64, `num_bits < 1`, `apot_nums < 1`, any of the three integer arguments not an int (a bool counts as illegal), `group_size > 0` that does not divide $R\cdot C$.

## Examples

Example 1: a $2\times 4$ weight, `group_size=-1` (one group per row), $B=2$, $S=2$. Row 0, element 0 is $1.0$: $\alpha_0=\operatorname{mean}(|W|)=2.5$ with $b_0=+1$, and the residual $1.0-2.5=-1.5$ gives $b_1=-1$; $2.5$ splits into $2+0.5$ and $1.5$ into $2-0.5$, so $\hat w=1\cdot 2.5+(-1)\cdot 1.5=1.0$. The last element of row 1 is 0 and both planes stay at 0 for it.

```
weight     = [[1.0, 1.0, -3.0, 5.0], [2.0, -1.0, 0.0, 1.0]]
num_bits   = 2
group_size = -1
apot_nums  = 2

group_shape   = (2, 4)
planes[0][0]  = [1, -1]
planes[0][1]  = [1, -1]
planes[0][2]  = [-1, -1]
planes[0][3]  = [1, 1]
planes[1][0]  = [1, 1]
planes[1][1]  = [-1, 0]
planes[1][2]  = [0, 0]
planes[1][3]  = [1, 0]
alpha         = [[2.50000000, 1.50000000], [1.00000000, 0.25000000]]
alpha_apot    = [[2.50000000, 1.50000000], [1.00000000, 0.25000000]]
apot_terms    = [[[2.00000000, 0.50000000], [2.00000000, -0.50000000]],
                 [[1.00000000, 0.00000000], [0.25000000, 0.00000000]]]
recon         = [[1.00000000, 1.00000000, -4.00000000, 4.00000000],
                 [1.25000000, -1.00000000, 0.00000000, 1.00000000]]
mse           = 0.32031250
```

Example 2: a $2\times 3$ weight with `group_size=2`. $g=2$ does not divide $C=3$; the flattening is $[1,2,4,8,16,32]$ and the three groups are $[1,2]$, $[4,8]$, $[16,32]$ - the second group straddles the two rows. $\alpha=[1.5,6,24]$ and the nearest powers are $2,8,32$ ($6\to 2^{\operatorname{round}(2.585)}=8$, $24\to 2^{\operatorname{round}(4.585)}=32$), so columns 2 and 3 of the same row end up with different scales.

```
weight     = [[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]]
num_bits   = 1
group_size = 2
apot_nums  = 1

group_shape   = (3, 2)
planes[0][0]  = [1]
planes[0][1]  = [1]
planes[1][0]  = [1]
planes[1][1]  = [1]
planes[2][0]  = [1]
planes[2][1]  = [1]
alpha         = [[1.50000000], [6.00000000], [24.00000000]]
alpha_apot    = [[2.00000000], [8.00000000], [32.00000000]]
apot_terms    = [[[2.00000000]], [[8.00000000]], [[32.00000000]]]
recon         = [[2.00000000, 2.00000000, 8.00000000],
                 [8.00000000, 32.00000000, 32.00000000]]
mse           = 45.50000000
```

## Notes

- Axis 1 of `planes` is the position **inside the flattened group**, not the row index; `planes[k]` matches row $k$ of $W$ only when $g$ divides $C$.
- Term $k$ of the ApOT decomposition consumes the residual "scale minus the terms already submitted", not the original scale; each term may be negative, so $t_0$ may exceed $\alpha$.
- `alpha` and `alpha_apot` are different quantities and `recon`/`mse` use only the latter. With $S=1$ the ApOT scale is just the nearest power of $\alpha$; raising $S$ only pulls `alpha_apot` closer to `alpha` and never changes `planes`.
- Rounding is "nearest integer", not floor: for $\alpha=3$ the first term is $4$ (floor would give $2$), and for $\alpha=0.4$ it is $0.5$ (floor would give $0.25$). A tie is only possible at $\alpha=2^{k+1/2}$, which is irrational, so binary inputs essentially never hit one and judging does not depend on the tie rule.
- Judging looks only at the returned dict; the implementation strategy is free.
