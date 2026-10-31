# Sampling: Mixture-of-Samplers Distribution Blending (MoA)

## Problem

Every earlier problem in this collection has exactly **one distribution source**: one logits vector, one set of hyperparameters. But one set of hyperparameters can only express one stance — high temperature with a wide nucleus for diversity, or low temperature with a narrow nucleus for accuracy; you cannot have both at once. MoA (mixture-of-samplers, the "run several sampling chains in parallel and merge them" assembly technique from the llama.cpp ecosystem) turns that conflict into a **vote**: \(E\) routes each produce their own distribution with their own temperature and their own top-\(p\), each route **oversamples** \(m\) tokens, the \(E\cdot m\) draws are unioned into one **joint candidate set** \(\mathcal U\), and the \(E\) distributions are then **blended with caller-supplied weights over \(\mathcal U\)** into a single distribution — from which exactly one token is drawn.

The point is that what gets composed is **the distribution itself**, and the blending happens in **probability space**. This is not "finding a better temperature" and not "cutting the candidate list down" — dynamic-temperature style rescaling only relabels the temperature, truncation styles only delete candidates, while the final distribution here may belong to no single route and may not equal any route simply reheated.

## Rules

- Inputs: an \(E\times V\) logits matrix (row \(e\) is route \(e\)'s next-token scores), non-negative weights \(w\) of length \(E\), temperatures \(t\) of length \(E\), nucleus masses \(p\) of length \(E\), the per-route oversample count \(m\), an \(E\times m\) block of uniform draws (one per draw per route), and the uniform draw \(u\) used for the single final sample. **All randomness comes from the arguments.**
- **① Each route produces its own distribution.** When \(t_e \le 0\) it degenerates to one-hot: \(1\) at the largest logit of row \(e\) (lowest index on ties), \(0\) elsewhere; otherwise \(p^{(e)} = \mathrm{softmax}(\text{logits}_e / t_e)\). Then apply that route's own top-\(p\) nucleus: sort \(p^{(e)}\) **descending** by value (ties by ascending index), keep the prefix up to the point where the descending cumulative sum **first reaches \(\ge p_e\)**, always keeping at least \(1\) element; renormalize inside the prefix and map back to the original index order — call it \(q_e\) (strictly \(0\) outside the nucleus).
- **② Oversample into the joint candidate set.** Route \(e\) uses its \(m\) uniform draws in turn, taking the inverse CDF of \(q_e\) in the **original index order**, drawing \(1\) token each time (routes are independent, sampling is with replacement, so the same token can be drawn several times). The **union, deduplicated**, of all \(E\cdot m\) draws is \(\mathcal U\), listed in ascending token-index order. Note the union is independent of the weights: a route with \(w_e = 0\) still pushes its draws into \(\mathcal U\).
- **③ Weighted blend.** First renormalize every route over \(\mathcal U\): \(r_e(t) = q_e(t) \big/ \sum_{t' \in \mathcal U} q_e(t')\); a route whose total mass on \(\mathcal U\) is \(0\) contributes nothing to the blend. Then
  \[
  p_{\text{mix}}(t) = \sum_{e} \frac{w_e}{\sum_{e'} w_{e'}}\, r_e(t), \qquad t \in \mathcal U
  \]
  and normalize once more over \(\mathcal U\) to obtain the output distribution `filtered`: positions outside \(\mathcal U\) are strictly \(0\).
- **④ A single sample.** Take the inverse CDF of `filtered` in the **original index order** with \(u\) (the first index whose cumulative sum is \(\ge u\)) to get `chosen`. The whole pipeline draws only once, here.
- Degenerate convention: if \(u\) exceeds the largest cumulative sum (e.g. \(\mathcal U = \varnothing\) makes the distribution all zeros), the inverse CDF falls back to the last index \(V-1\).
- Outputs, three of them: `filtered` (length \(V\), sums to \(1\)), `chosen`, and `in_union` (a boolean tensor of length \(V\) marking \(\mathcal U\)).

## Example

```
E = 2 routes, V = 5, m = 2

logits   = [[2.0, 1.0, 0.0, -1.0, -2.0],    ← route 0
            [0.0, 1.0, 2.0,  1.0,  0.0]]    ← route 1
weights  = [0.7, 0.3], temps = [1.0, 0.8], top_ps = [0.9, 0.9]
u_experts = [[0.1, 0.6], [0.3, 0.95]], u = 0.5

① Route 0: softmax(logits_0 / 1.0) = [0.636409, 0.234122, 0.086129, 0.031685, 0.011656]
   descending cumulative sum = [0.636409, 0.870530, 0.956659, 0.988344, 1.0], first ≥ 0.9 stops at the 3rd → nucleus = {0,1,2}
   q_0 = [0.665241, 0.244728, 0.090031, 0.0, 0.0]
   Route 1: softmax(logits_1 / 0.8) = [0.047252, 0.164925, 0.575646, 0.164925, 0.047252]
   descending = [0.575646(idx 2), 0.164925(idx 1), 0.164925(idx 3), 0.047252(idx 0), 0.047252(idx 4)]
   cumulative = [0.575646, 0.740571, 0.905496, ...] → nucleus = {2,1,3}
   q_1 = [0.0, 0.182138, 0.635724, 0.182138, 0.0]

② Oversampling (inverse CDF in original order): CDF of q_0 = [0.665241, 0.909969, 1.0, 1.0, 1.0]
   u = 0.1 → index 0; u = 0.6 → index 0 (both shots hit the same token)
   CDF of q_1 = [0.0, 0.182138, 0.817862, 1.0, 1.0]
   u = 0.3 → index 2; u = 0.95 → index 3
   U = {0, 2, 3} (4 draws, only 3 distinct candidates) → in_union = [1, 0, 1, 1, 0]

③ Each route is first renormalized over U (values listed in the order U = [0,2,3]):
   Σ_U q_0 = 0.755271 → r_0 = [0.880797, 0.119203, 0.0]
   Σ_U q_1 = 0.817862 → r_1 = [0.0,        0.777300, 0.222700]
   p_mix = 0.7·r_0 + 0.3·r_1 = [0.616558, 0.316632, 0.066810], total mass on U is already = 1.0

④ CDF of filtered in original order = [0.616558, 0.616558, 0.933190, 1.0, 1.0], u = 0.5 → first ≥ 0.5 is index 0

Output: filtered = [0.616558, 0.0, 0.316632, 0.066810, 0.0], chosen = 0, in_union = [1, 0, 1, 1, 0]
```

Two degenerate cases:

```
Greedy route (temperature = 0): logits = [[-1.0, 5.0, 0.0, 2.0]], weights = [1.0], temps = [0.0], top_ps = [1.0],
m = 3, u_experts = [[0.2, 0.6, 0.99]], u = 0.5
  q_0 = one-hot(index 1) → all three shots land on index 1 → U = {1} → the blend is exactly that one-hot
  Output: filtered = [0.0, 1.0, 0.0, 0.0], chosen = 1, in_union = [0, 1, 0, 0]

A route with weight 0 still enters U: logits = [[4.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0]], weights = [0.0, 1.0],
temps = [1.0, 1.0], top_ps = [1.0, 1.0], m = 1, u_experts = [[0.1], [0.1]], u = 0.5
  route 0 draws index 0, route 1 draws index 3 → U = {0, 3}
  route 1 softmax = [0.017362, 0.017362, 0.017362, 0.947915], its mass on U = 0.965277
  → r_1 = [0.017986, 0.982014] (in the order of U); after weight normalization only route 1 carries mass
  Output: filtered = [0.017986, 0.0, 0.0, 0.982014], chosen = 3, in_union = [1, 0, 0, 1]
```

## Constraints

- Judging environment: **CPU only**; **torch / numpy only**; **completely offline** — no llama.cpp, no transformers sampler / logits-processor classes, no network dependency of any kind. **All randomness must be an explicit argument**: the uniform draws are supplied via \(u\) and \(u_{\text{experts}}\); you may not call `manual_seed`, `torch.rand` or `torch.multinomial` yourself.
- \(E \ge 1\), \(V \ge 1\), \(m \ge 1\); weights are non-negative with \(\sum_e w_e > 0\); temperatures may be any real number (\(\le 0\) means greedy); top-\(p \in (0,1]\); every component of \(u\) and \(u_{\text{experts}}\) lies in \([0,1)\).
- Identities that always hold: \(|\mathcal U| \le E\cdot m\); `filtered` is strictly \(0\) outside \(\mathcal U\) and sums to \(1\); whenever \(\mathcal U\neq\varnothing\), `chosen` lies inside \(\mathcal U\).
- When \(E = m = 1\), \(|\mathcal U| = 1\) and the blended distribution necessarily degenerates into the one-hot of that route's own single draw.
- Numerics: float32 intermediates, judged with `atol = rtol = 1e-5`.
