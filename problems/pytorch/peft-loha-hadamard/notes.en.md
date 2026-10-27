# PEFT: LoHa Hadamard-Product Low-Rank Adaptation — In Depth

## 1. Formula and provenance

LoHa comes from the FedPara family (Hyeon-Woo et al., *FedPara: Low-Rank Hadamard Product for Language Models*, arXiv:2108.04916), implemented in the PEFT library at `tuners/loha/layer.py`. Parameter names and shapes of the Linear layer (`create_adapter_parameters`):

- `hada_w1_a`: \((d_\text{out}, r)\), `hada_w1_b`: \((r, d_\text{in})\)
- `hada_w2_a`: \((d_\text{out}, r)\), `hada_w2_b`: \((r, d_\text{in})\)

`make_weight` (`HadaWeight.forward`) has a one-line core:

```python
diff_weight = ((w1a @ w1b) * (w2a @ w2b)) * scale      # scale = alpha / r
```

i.e. \(\Delta W = s\,(W_{1a}W_{1b})\odot(W_{2a}W_{2b})\), with the same scaling as LoRA \(s=\alpha/r\) (`update_layer` sets `self.scaling[adapter_name] = alpha / r`). Merging (`get_delta_weight` then `weight += delta`) is exactly this problem's \(W' = W_0 + \Delta W\).

## 2. Why multiplication can be "full-rank"

LoRA's \(\Delta W = s\,BA\) always satisfies \(\operatorname{rank}(\Delta W)\le r\). The Hadamard product has a famous rank property (Matousek & Stivner):

\[\operatorname{rank}(A\odot B)\ \le\ \operatorname{rank}(A)\cdot\operatorname{rank}(B)\]

Multiplying two rank-\(r\) branches turns the rank upper bound into \(r^2\), and it is attainable in general position — that is, expressing a full-rank \(d_\text{out}\times d_\text{in}\) update with \(2r(d_\text{out}+d_\text{in})\) parameters. FedPara uses exactly this to keep the parameter budget at LoRA's level while reaching full-rank expressivity. The intuition behind the counterexample assertion (`check_hadamard_not_sum`): multiplication has an "element-wise gating" effect, and position-wise correlation is beyond what addition can produce.

## 3. Initialization and degeneracies (judging focus)

- PEFT/LyCORIS default init (`reset_adapter_parameters`) zeroes `w2_b` and uses Kaiming for the rest — at the start \(\Delta W=0\) and the adapter "does nothing" (the original comment: *It is enough to initialize only single matrix with zeros to make adapter do nothing*). `check_zero_branch_degenerates_to_base` checks exactly this.
- \(\alpha=0\) obviously gives \(\Delta W=0\); with \(r=1\) both branches are single outer products, \(\Delta W = s\,(u_1v_1^\top)\odot(u_2v_2^\top) = s\,(u_1\odot u_2)(v_1\odot v_2)^\top\), degenerating to a rank-1 LoRA — at rank 1, multiplication and addition are of the same order.
- Conv-layer LoHa also has a t-factor rebuild (`make_weight_cp`); this problem only covers the Linear form.

## 4. Division of labor within the collection

- `peft-lora-forward`/`peft-lora-merge`: additive combination \(s\,BA\), rank \(\le r\).
- `peft-dora`: magnitude-direction decomposition on top of LoRA.
- This problem: multiplicative combination, \(\Delta W\) can be full rank; note the shape convention follows the PEFT source \((d_\text{out},r)\times(r,d_\text{in})\), opposite in naming to the LoRA problem's \(A\in(r,d_\text{in}),B\in(d_\text{out},r)\) but isomorphic in essence.
