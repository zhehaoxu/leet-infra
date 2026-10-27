# PEFT: LoKr Kronecker-Product Weight Merge — In Depth

## 1. Formula and the PEFT convention

LoKr comes from Yeh et al., arXiv:2108.07383 (LyCORIS `lokr`), implemented in PEFT at `tuners/lokr/layer.py`. Linear form:

- `factorization(in_dim)`/`factorization(out_dim)` split the dimensions into multipliers (see the LoKr Kronecker Factorization problem), giving the shape quadruple \((a,b)\times(c,d)\);
- full form `lokr_w1: (a,b)`, `lokr_w2: (c,d)` (either factor may itself be low-ranked: `w1 = w1_a @ w1_b`);
- `get_delta_weight` calls `make_kron(w1, w2, self.scaling[adapter_name])`, and the last line of `make_kron` is `torch.kron(w1, w2) * scale`, with `scale = alpha / r` (`update_layer`, same as LoRA/LoHa).

Merge: \(W' = W_0 + s\,(W_1\otimes W_2)\).

## 2. Block layout of the Kronecker product (judging core)

\[(W_1\otimes W_2)[\,i\,c+u,\; j\,d+v\,] = W_1[i,j]\cdot W_2[u,v]\]

Memory aid: each element of \(W_1\) "inflates" into a \(W_2\)-shaped small block; blocks are ordered along rows by \(W_1\)'s row order and within a block by \(W_2\)'s row order (likewise for columns). **The two factors are asymmetric**: outer structure × inner texture, hence \(\operatorname{kron}(W_1,W_2)\neq\operatorname{kron}(W_2,W_1)\) (even the shapes can differ). `check_matches_oracle` uses the non-square factors \((2,3)\otimes(3,2)\) to test the layout direction — a flipped row/column block order fails to match immediately.

## 3. Parameter ledger (an interview favorite)

- \(\Delta W\) element count: \((a\,c)(b\,d)\); factor parameter count: \(a\,b+c\,d\). Square case \(W_1,W_2\in\mathbb R^{k\times k}\): \(2k^2\) parameters express a \((k^2)^2=k^4\) matrix — at \(k=16\), 512 parameters vs 65 536 elements, a 128× compression.
- Compared with LoHa: LoHa is "full-rank but element-wise gated", LoKr is "structured sparsity (Kronecker rank 1)"; both need only \(O(d)\)-scale parameters (LoKr can go even lower).
- LyCORIS-style init (`reset_adapter_parameters_lycoris_way`) zeroes \(W_2\): \(\Delta W=0\) at the start, corresponding to `check_zero_w2_degenerates_to_base`.
- In the full form, \(r\) only carries scaling semantics (\(s=\alpha/r\)); \(r\) decides the side-path width only when a factor is low-ranked (`use_w1/use_w2=False`).

## 4. Division of labor within the collection

- `peft-lokr-factorization`: the upstream of this problem — how the dimensions split into \((a,b),(c,d)\).
- `peft-loha-hadamard`: the other multiplicative route (Hadamard vs Kronecker).
- `peft-lora-merge`: additive merge \(W_0+sBA\).
