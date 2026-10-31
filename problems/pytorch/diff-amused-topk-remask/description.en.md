# AMUS Remasking: Push the Top-k Least Confident Positions Back Under the Mask

## Problem Statement

Discrete diffusion of the MaskGIT family (which is what AMUS uses) does not add noise to continuous values; it replaces tokens with a dedicated `mask_token_id`. Every denoising step re-predicts all masked positions but **keeps** only the most confident few, leaving the rest masked until the final step commits everything. How many to keep, and which ones, is the whole algorithm of a single step.

AMUS's selection criterion is not the raw probability but the **log probability with Gumbel noise added**: the hot temperatures of the early steps let randomness bury even some decent positions so nothing gets locked in too soon, while the annealed tail behaves like "pick the lowest probabilities". On top of that the per-step mask count passes two clamps — it may not exceed the number of positions currently masked (otherwise the run never finishes), and it may not drop below 1 (otherwise the step does nothing).

This task is a **single step** and needs no model: the categorical draws and the uniform variates feeding the Gumbel noise are passed in explicitly, and you only have to apply the rules below.

## Rules

Let the batch size be \(B\), the sequence length \(S\), the vocabulary size \(V\), the number of inference steps \(N\), and \(m\) the `mask_token_id`.

1. **Unknown positions and candidates**: \(\text{unknown}_i = (\text{sample}_i == m)\),
   \[
   \text{committed} = \text{where}(\text{unknown},\ \text{drawn\_tokens},\ \text{sample})
   \]
   masked positions take the caller's drawn token, already-decoded positions stay as they are.
2. **Step index**: the scheduler walks timestep from \(N-1\) down to \(0\) while the ratio table advances from index \(0\) to \(N-1\), hence
   \[
   \text{step\_idx} = N - 1 - \text{timestep},\qquad \text{ratio} = \frac{\text{step\_idx} + 1}{N}
   \]
3. **Mask ratio** (then multiplied by `starting_mask_ratio`):
   \[
   \text{mask\_ratio} = \begin{cases}\cos(\text{ratio}\cdot \pi / 2) & \text{cosine}\\ 1 - \text{ratio} & \text{linear}\end{cases}
   \]
4. **Two clamps** (the last two are per batch row; `ratio` is shared across the batch):
   \[
   \text{mask\_len} = \big\lfloor S \cdot \text{mask\_ratio} \big\rfloor
   \;\xrightarrow{\ \min(\,\cdot,\ \sum \text{unknown} - 1\,)\ }\;
   \xrightarrow{\ \max(\,\cdot,\ 1\,)\ }
   \]
   The lower bound 1 is **unconditional**: even a row with no unknown position at all still buries one token.
5. **Selected probability and sentinel**: with \(p = \text{softmax}(\text{logits})\), take \(p_{\text{sel},i} = p_{i,\text{committed}_i}\) and then
   \[
   p_{\text{sel}} \leftarrow \text{where}(\text{unknown},\ p_{\text{sel}},\ \texttt{finfo(float64).max})
   \]
   decoded positions are pushed to a huge value so that they **never** compete.
6. **Gumbel noise and confidence**: let \(u\) be `gumbel_uniforms`, \(\text{temps} = \text{linspace}(\text{start}, \text{stop}, N)\) and \(T = \text{temps}_{\text{step\_idx}}\):
   \[
   g = -\ln\!\big(\operatorname{clamp}(-\ln(\operatorname{clamp}(u,\ 10^{-20})),\ 10^{-20})\big),\qquad
   c = \ln\!\big(\operatorname{clamp}(p_{\text{sel}},\ 10^{-20})\big) + T \cdot g
   \]
   Both clamps matter: \(u = 0\) gives \(g \approx -3.829\) and \(u = 1\) gives \(g \approx 46.05\); a zero probability is counted as \(10^{-20}\). Note the sentinel replaces the probability **before** the logarithm, so a decoded position's confidence is \(\ln(1.798\times 10^{308}) + T g \approx 709.78 + T g\), not the constant itself.
7. **Strictly below the element at index mask\_len**: sort each row's \(c\) ascending and use the element with index \(\text{mask\_len}\) as the cut-off,
   \[
   \text{masking}_i \iff c_i < c_{(\text{mask\_len})},\qquad
   \text{prev\_sample} = \text{where}(\text{masking},\ m,\ \text{committed})
   \]
   The test is **strictly less**, not \(\le\): on ties nothing gets buried, so the actual number of remasked positions can be smaller than \(\text{mask\_len}\).
8. **Final commit**: when \(\text{timestep} = 0\) there is no remasking at all — \(\text{prev\_sample} = \text{committed}\), `masking` is all False and `mask_len` is all 0.

Return `(prev_sample, committed, masking, mask_len)`: the first two are `int64` of shape \((B, S)\), `masking` is `bool` of shape \((B, S)\), `mask_len` is `int64` of shape \((B, 1)\). All intermediate math is done in `float64`.

## Example

\(B=1,\ S=6,\ V=3,\ m=2\), remaining inputs:

```
logits (6 positions x 3 tokens):
  pos0: [ 2,  0, -1]   pos1: [ 0,  0,  0]   pos2: [ 1,  1,  0]
  pos3: [ 3,  0,  0]   pos4: [-1,  2,  0]   pos5: [ 0,  1,  2]
sample          = [[2, 0, 2, 2, 2, 2]]
drawn_tokens    = [[1, 0, 0, 2, 2, 0]]
gumbel_uniforms = [[0.2, 0.5, 0.9, 0.1, 0.75, 0.4]]
timestep = 2, num_inference_steps = 4, mask_token_id = 2
masking_schedule = "cosine", temperature = (2.0, 0.0), starting_mask_ratio = 1.0
```

`committed = [[1, 0, 0, 2, 2, 0]]`: position 1 is already decoded (value 0), the other five take their drawn candidate.
`step_idx = 4 - 1 - 2 = 1`, `ratio = 2/4 = 0.5`, `mask_ratio = cos(π/4) = 0.70711`, `floor(6 × 0.70711) = 4`; there are 5 unknown positions so `min(4, 5-1) = 4` and `mask_len = [[4]]`. The temperature is `linspace(2, 0, 4)[1] = 4/3` and the six confidences are:

```
confidence = [-2.804359, 710.271397, 2.138495, -4.206966, -0.508647, -2.291044]
```

Position 1 is the sentinel. Sorting ascending and taking the element at index 4 (the 5th smallest), \(2.138495\), as the cut-off, exactly four positions — 0, 3, 4, 5 — are strictly below it and get buried under \(m=2\), while position 2, which *is* the cut-off, survives:

```
masking    = [[True, False, False, True, True, True]]
prev_sample = [[2, 0, 0, 2, 2, 2]]
```

With the same tensors and only the parameters changed:

- `timestep=3` (the first denoising step): `ratio=0.25`, `floor(6 × 0.92388)=5`, clamped back to `min(5, 5-1)=4`; `masking` and `prev_sample` as above;
- `timestep=1`: `ratio=0.75`, `mask_len=[[2]]`, `masking=[[True, False, False, True, False, False]]`, `prev_sample=[[2, 0, 0, 2, 2, 0]]`;
- `masking_schedule="linear"`, `timestep=2`: `mask_ratio=0.5`, `mask_len=[[3]]`, `masking=[[True, False, False, True, False, True]]`, `prev_sample=[[2, 0, 0, 2, 2, 2]]`;
- `masking_schedule="linear"`, `starting_mask_ratio=0.0`, `timestep=2`: the `floor` gives 0 but is raised to 1, burying exactly position 3 — `masking=[[False, False, False, True, False, False]]`, `prev_sample=[[1, 0, 0, 2, 2, 0]]`;
- `timestep=0`: `prev_sample == committed == [[1, 0, 0, 2, 2, 0]]`, `masking` all False, `mask_len=[[0]]`.

## Constraints

- `logits` must be a 3-D `float32` / `float64` tensor of shape \((B, S, V)\) with \(B \ge 1\), \(S \ge 2\), \(V \ge 1\) and no nan / inf;
- `sample` and `drawn_tokens` must be `int32` / `int64` tensors of shape \((B, S)\) with values in \([0, V)\);
- `gumbel_uniforms` must be a `float32` / `float64` tensor of shape \((B, S)\) with values in \([0, 1]\) and no nan / inf;
- `timestep`, `num_inference_steps` and `mask_token_id` must be Python ints (booleans do not count): \(N \ge 1\), `timestep` in \([0, N)\), `mask_token_id` in \([0, V)\);
- `masking_schedule` must be the string `"cosine"` or `"linear"`;
- `temperature` is a real number, or a tuple/list of exactly two real numbers, all with absolute value at most 8 (this keeps the sentinel strictly above every unknown position); a single value \(T\) means \((T, 0.01)\);
- `starting_mask_ratio` must be a finite real in \([0, 1]\);
- any violation raises `ValueError`;
- properties: `mask_len` always lies in \([1, S-1]\); as long as the row has an unknown position, decoded positions are never masked; with `temperature=(0,0)` the buried set is exactly the `mask_len` unknown positions with the lowest selected probability; when confidences tie inside a row (e.g. everything decoded and identical `gumbel_uniforms`) `masking` can be all False, burying fewer than `mask_len`.
