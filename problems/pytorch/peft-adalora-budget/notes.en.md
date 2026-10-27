# PEFT: AdaLoRA Budgeted Rank Reallocation — Algorithm Notes

## Formula and provenance

AdaLoRA (Zhang et al., ICLR 2023, arXiv:2303.10512) observes that spreading a fixed parameter budget \(\text{budget}=r\cdot(d_\text{out}+d_\text{in}+1)\) uniformly over all layers is suboptimal — different layers and different singular directions vary a lot in importance. It writes each layer's incremental update as an SVD triplet:

\[\Delta W = \bigl(B\,(A\odot E)\bigr)\cdot\frac{s}{\rho+\epsilon},\qquad A\in\mathbb R^{r\times d_\text{in}},\ E\in\mathbb R^{r\times 1},\ B\in\mathbb R^{d_\text{out}\times r}\]

where \(A,B\) are the singular vectors (an orthogonality approximation), \(E\) the singular values, \(\rho\) the current rank, and \(s\) the scale (PEFT `get_delta_weight`: `transpose(lora_B @ (lora_A * lora_E), fan_in_fan_out) * scaling / (ranknum + 1e-5)`). The triplet uses \(r(d_\text{out}+d_\text{in}+1)\) parameters — \(r\) more than plain LoRA — in exchange for **reallocating rank during training**.

This problem only covers the scoring and truncation of `RankAllocator.update_and_allocate`, not the forward pass.

## Two-step EMA in the sensitivity score (source-code convention)

The PEFT `RankAllocator` maintains two smoothed quantities per parameter (source variables `exp_avg_ipt`/`exp_avg_unc`):

- **Importance**: \(\text{ipt}=|p\cdot g|\) (element-wise parameter-gradient product, absolute value).
- **Uncertainty**: an EMA over the deviation \(|\text{ipt}-\bar m|\) between ipt and its EMA mean.

Between two `update_and_allocate` calls exactly two fresh gradient steps accumulate, so the score is produced by pushing two steps from the zero state:

\[\bar m^{(1)}=(1-\beta_1)\,\text{ipt}_1,\qquad u^{(1)}=(1-\beta_2)\,|\text{ipt}_1-\bar m^{(1)}|\]
\[\bar m^{(2)}=\beta_1\bar m^{(1)}+(1-\beta_1)\,\text{ipt}_2,\qquad u^{(2)}=\beta_2 u^{(1)}+(1-\beta_2)\,|\text{ipt}_2-\bar m^{(2)}|\]

Note \(\bar m^{(1)}=(1-\beta_1)\text{ipt}_1\ne\text{ipt}_1\) (unless \(\beta_1=0\)), hence \(u^{(1)}=(1-\beta_2)\,\beta_1\,\text{ipt}_1\) is generally nonzero — "no history on step one" does not mean "no uncertainty on step one". The final element score is \(\bar m^{(2)}\cdot u^{(2)}\): a parameter counts as important only if its mean is large **and** it fluctuates; a large-but-steady parameter gets discounted.

## Triplet composition and global truncation (judged points)

The three member tensors of singular triplet \(i\) each have element scores; the composition **averages over each tensor's owning dimension**:

- \(E\) already holds one scalar per row: \(s^E_i=\text{element}(E)_i\);
- row \(i\) of \(A\) (its \(d_\text{in}\) parameters) belongs to triplet \(i\): \(s^A_i=\text{mean}_j\,\text{element}(A)_{ij}\) (source `mean(dim=1)`);
- column \(i\) of \(B\) (its \(d_\text{out}\) parameters) belongs to triplet \(i\): \(s^B_i=\text{mean}_w\,\text{element}(B)_{wi}\) (source `mean(dim=0)`).

\(\text{score}_i=s^E_i+s^A_i+s^B_i\) is a **plain sum** (not an average). The truncation is global: \(k=r-\text{budget}\), threshold \(=\) the \(k\)-th smallest score (`torch.kthvalue(scores, k)`), and the mask condition is `scores <= threshold` — which means:

- **Ties delete more**: two triplets with strictly equal scores at the threshold are both zeroed, so the kept count drops below budget. `check_ties_delete_more` pins this edge (native PEFT behavior, no dedup).
- **The mask only hits \(E\)**: \(A,B\) stay intact. A killed direction may later "revive" through further training — the paper's ablations show this soft deletion is more stable than hard parameter removal.
- When \(k<1\) (budget \(\ge r\)) `kthvalue` is undefined and PEFT simply skips masking; the `budget=r` branch of `check_budget_boundary` covers it.

## Division of labor within the collection

- `peft-lora-forward`/`peft-lora-merge`: static LoRA forward and merge with a fixed rank.
- `peft-lora-memory`: also "sensitivity scoring", but with the **inter-layer** static parameter-count convention (each layer's score only decides whether the whole layer is kept); this problem is the **intra-layer** dynamic rank reallocation (each step may kill part of the singular directions within the same layer).
- This problem: AdaLoRA's budget schedule (cubic decay from the initial budget to the target) is reduced to a single given budget; the two gradient steps are provided directly, so no real backpropagation is needed.

## Verification approach

- `check_example`: \(r=2,d_\text{in}=d_\text{out}=1\), both gradient steps all ones, \(\beta=0.5\) → hand-computed scores \([6.5625, 10.5]\); with budget=1 \(E_0\) is zeroed.
- `check_matches_oracle`: random inputs vs an independent scalar-by-scalar recomputation (float64 internally, float32 out, tolerance \(10^{-5}\)).
- The threshold and tie behavior is pinned by the fixed counter-example in `check_ties_delete_more`.
