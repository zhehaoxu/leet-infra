# PEFT: DARE Drop-and-Rescale with TIES Merging

## Problem

After fine-tuning multiple homologous models, **merging them into one multi-task model** is high-frequency inference-side engineering. The two most common merging operators both take "task vectors" as input (\(\delta=\theta_\text{ft}-\theta_0\); this problem hands you the \(\delta\) matrices directly). Implement strictly by the following conventions.

**Step one: DARE drop-and-rescale** (Drop And REscale, Yu et al., arXiv:2311.03099, "Language Models are Super Mario"). Task vectors have heavily cancelling, redundant parameters; DARE drops at random with keep probability \(1-p\) and rescales the survivors to preserve the expectation:

\[\delta' = \frac{m\odot\delta}{1-p},\qquad m\in\{0,1\}^{N}\ (m=1\ \text{means keep})\]

This problem **does not flip coins**: the drop masks \(m_1,m_2\) are given as explicit inputs (for determinism). The two tasks' DARE outputs are averaged:

\[\tau_\text{DARE}=\tfrac12\bigl(\delta_1'+\delta_2'\bigr)\]

**Step two: TIES merging** (TrIm, Elect Sign & disjoint merge, Yadav et al., arXiv:2306.01708), applied to \(\delta_1',\delta_2'\) (i.e. the full DARE-TIES pipeline), in three steps:

1. **TRIM**: flatten each vector and keep the `keep_n` largest entries by \(|\delta'|\), zeroing the rest; **ties in magnitude go to the smaller index**.
2. **ELECT SIGN**: vote per position, \(\gamma=\operatorname{sign}\bigl(\operatorname{sign}(\delta_1')+\operatorname{sign}(\delta_2')\bigr)\); when the two vectors disagree in sign (votes cancel) or are both zero, \(\gamma=0\).
3. **DISJOINT MERGE**: per position, only vectors whose **sign agrees with \(\gamma\)** enter the mean, and **the denominator is the "number of agreeing tasks" (1 or 2), not the total task count**:

\[\tau_\text{TIES}[j]=\frac{\sum_{t:\,\operatorname{sign}(\delta_t'[j])=\gamma_j\neq0}\delta_t'[j]}{\bigl|\{t:\operatorname{sign}(\delta_t'[j])=\gamma_j\neq0\}\bigr|}\qquad(\gamma_j=0\Rightarrow 0)\]

This means a "single-survivor" position takes the survivor's **original value** instead of dividing by 2 — exactly how TIES differs from plain averaging and prevents sign interference.

## Task

Given two task vectors \(\delta_1,\delta_2\) (same-shape matrices), the drop probability `dare_p` (\(0\le p<1\)), two 0/1 masks `mask1/mask2` (1=keep), and the keep count `keep_n` (\(1\le\text{keep\_n}\le N\), with \(N\) the total element count), return the pair **`(dare_merged, ties_merged)`**:

- `dare_merged` = \(\tfrac12(\delta_1'+\delta_2')\), where \(\delta_t'=m_t\odot\delta_t/(1-p)\);
- `ties_merged` = the result of applying the TIES three steps to \(\delta_1',\delta_2'\).

Both are float32 tensors of the input shape; tolerance \(10^{-5}\).

## Hints

- With `p=0`, all-ones masks, and `keep_n=N`: DARE degenerates to plain averaging; TIES degenerates to "average where signs agree, zero where they conflict".
- TRIM's tie-breaking is the determinism key: sort by `(magnitude descending, index ascending)` and take the first `keep_n`.
