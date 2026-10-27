# PEFT: AdaLoRA Budgeted Rank Reallocation

## Problem

**AdaLoRA** (Zhang et al., ICLR 2023, *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*; implemented in the PEFT library `tuners/adalora`) re-parameterizes the additive LoRA update into SVD-form triplets: each layer maintains right singular vectors \(A\in\mathbb R^{r\times d_\text{in}}\), singular values \(E\in\mathbb R^{r\times 1}\), and left singular vectors \(B\in\mathbb R^{d_\text{out}\times r}\). Its real novelty is **dynamically reallocating the rank budget during training**: every few steps it scores the \(r\) singular triplets with a "parameter sensitivity" measure and zeroes the singular values of the lowest-scored ones — the zeroed triplets stop contributing updates, which is equivalent to moving budget from "unimportant" singular directions to "important" ones.

The sensitivity scoring (PEFT `RankAllocator` convention) computes, for each scalar parameter \(p\), an EMA smoothing over two gradient steps \(g_1,g_2\) with \(\text{ipt}=|p\cdot g|\):

\[\bar m^{(1)}=(1-\beta_1)\,\text{ipt}_1,\qquad u^{(1)}=(1-\beta_2)\,\bigl|\text{ipt}_1-\bar m^{(1)}\bigr|\]
\[\bar m^{(2)}=\beta_1\bar m^{(1)}+(1-\beta_1)\,\text{ipt}_2,\qquad u^{(2)}=\beta_2 u^{(1)}+(1-\beta_2)\,\bigl|\text{ipt}_2-\bar m^{(2)}\bigr|\]
\[\text{element}(p)=\bar m^{(2)}\cdot u^{(2)}\]

Both steps start from the zero state (the first step has no history). The element scores of the three tensors are then aggregated **by dimension-wise averaging** into one score per singular triplet:

\[s^E_i=\text{element}(E)_i,\qquad s^A_i=\frac{1}{d_\text{in}}\sum_{j}\text{element}(A)_{ij},\qquad s^B_i=\frac{1}{d_\text{out}}\sum_{w}\text{element}(B)_{wi}\]

\[\text{score}_i=s^E_i+s^A_i+s^B_i\]

The budget truncation takes the **global \(k\)-th smallest** of the \(r\) scores as the threshold (\(k=r-\text{budget}\), compared globally across triplets), and zeroes \(E_i\gets 0\) for **all** triplets with \(\text{score}_i\le\) threshold — ties delete extra triplets, which is the native PEFT behavior; when \(k<1\) (budget \(\ge r\)) no masking happens at all. Note the mask only lands on \(E\): \(A,B\) stay intact, and a killed direction is merely "weight-zeroed".

## Task

Implement one round of AdaLoRA budget reallocation: given the current triplet parameters and two gradient steps, the EMA coefficients, and the budget, return each triplet's sensitivity score and the masked \(E\).

- Inputs \(E\in\mathbb R^{r\times 1}\), \(A\in\mathbb R^{r\times d_\text{in}}\), \(B\in\mathbb R^{d_\text{out}\times r}\); gradients \(g^{(1)}_E,g^{(2)}_E\) shaped like \(E\), \(g^{(1)}_A,g^{(2)}_A\) like \(A\), \(g^{(1)}_B,g^{(2)}_B\) like \(B\); \(\beta_1,\beta_2\in(0,1)\); \(\text{budget}\) an integer (\(0\le\text{budget}\le r\)).
- Output \((\text{scores},\ \text{masked\_E})\): \(\text{scores}\in\mathbb R^{r}\), \(\text{masked\_E}\in\mathbb R^{r\times 1}\) (both float32).

**Convention**: compute in float64 internally, output float32; the threshold test uses \(\le\) (ties included); do not use any ready-made PEFT library. Tolerance \(10^{-5}\).

## Hints

- The two EMA steps start from the zero state: the first step has no history terms, so do not apply the second-step recurrence to it.
- \(s^A\) averages over the **rows** of \(A\) (the \(d_\text{in}\) parameters of each triplet) and \(s^B\) over the **columns** of \(B\) — do not swap the directions.
- The global \(k\)-th smallest can be obtained with `torch.kthvalue(scores, k)`, where \(k=r-\text{budget}\).
- Tied scores make the deleted count exceed \(r-\text{budget}\) — keep the native PEFT semantics, do not deduplicate.
