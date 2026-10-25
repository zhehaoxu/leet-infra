# Sampling: Epsilon Sampling Relative Threshold Cut

## Problem

top-p truncates by **cumulative mass** and top-k by **rank**, both sharing one blind spot: their judgment of "how much to keep" is independent of the distribution's **absolute probabilities** — a pathologically smoothed distribution (a long, spread-out tail) still looks reasonable to top-p. Epsilon truncation (Hewitt et al., *Truncation Sampling as Language Model Desmoothing*, arXiv:2210.15191) looks at the probabilities themselves: **cut the entire tail below a threshold**. This problem pins the **relative threshold** convention

\[\text{thr}=\varepsilon\cdot p_{\max},\qquad \text{keep}=\{i: p_i\ge\text{thr}\}\]

(the threshold anchors to the row's maximum probability, comparable across distributions; **equal to the threshold survives**, so the top-probability token always survives and `filtered`'s denominator stays positive). After the cut, two mass-handling modes: `renorm=True`, the traditional route — renormalize the survivors; `renorm=False` (the desmoothing-paper flavor) — survivors stay **untouched**, and the cut mass "hole" is dumped entirely onto the row's **original argmax** token (smallest index on ties), leaving every other survivor's probability unchanged. Implement it: randomness enters via the explicit uniform `u`, inverse-transform-sampled on `filtered`'s CDF in original order. CPU, float32, fully deterministic.

## Rules

- Factory `solve(logits, eps, renorm, u)`; the full contract lives only in the starter comments and is the single judging standard.
- Procedure:
  1. `probs = softmax(logits)`;
  2. per-row threshold \(\text{thr}=\varepsilon\cdot p_{\max}\) (`p_max` is that row's maximum probability); **cut candidates with \(p_i<\text{thr}\)** (the test is \(\ge\), equality survives);
  3. two renorm conventions:
     - `renorm=True`: renormalize the survivors `filtered = probs·keep / sum(kept)`;
     - `renorm=False`: survivors are **not** scaled; the hole \(=1-\sum_{\text{kept}}p_i\) is `scatter_add`-ed onto the row's `argmax` (highest original probability, smallest index on ties), every other survivor keeps its raw probability;
  4. `chosen`: inverse transform of `u` on the CDF of `filtered` in **original order** (first index whose prefix sum is \(\ge u\); fallback to the last index).
- Return `(filtered, chosen)`: `filtered` has the same shape as `probs`, rows sum to 1, cut positions strictly 0.
- Everything is float32 on CPU and deterministic; ready-made samplers are forbidden.

## Example

`logits=[2.0, 1.0, 0.5, -1.0, -2.0]`, `eps=0.15`, `u=0.95`.

softmax gives \(p=[0.6027,\,0.2217,\,0.1345,\,0.0300,\,0.0110]\); the threshold is \(0.15\times0.6027=0.0904\) → keep the first 3, cut the last two (total mass 0.0411).

- `renorm=False`: the three survivors stay **raw**, and 0.0411 is added to token 0: `filtered=[0.6438, 0.2217, 0.1345, 0.0, 0.0]`.
- `renorm=True`: the three divide by 0.9589: `filtered=[0.6285, 0.2312, 0.1402, 0.0, 0.0]`.

Under both conventions the CDF first reaches \(\ge0.95\) at token 2 → `chosen=2`.

```
renorm=False:  [0.6438, 0.2217, 0.1345, 0.0, 0.0]   chosen=2
renorm=True :  [0.6285, 0.2312, 0.1402, 0.0, 0.0]   chosen=2
```

## Constraints

- \(0\le\varepsilon\le1\); `logits` may carry batch dims (threshold, cut, renormalization per row); `u` shaped to match the batch dims
- Each row of `filtered` sums to 1 with cut positions strictly 0, and `chosen` lies in the kept set; floats `atol=rtol=1e-5`, `chosen` with `torch.equal`
- Four discriminating criteria the judge asserts directly: \(\varepsilon=0\) → threshold 0, everything kept, both conventions equal the raw softmax (the desmoothing convention's hole is 0, making "peak-filling" a harmless no-op); \(\varepsilon=1\) → only the argmax qualifies, both conventions are one-hot and `chosen` is exactly `argmax`; under `renorm=False` the non-argmax survivors' probabilities are **bit-identical** to the raw softmax (`atol=1e-7`) — mass flows only to the argmax; at a high threshold (\(\varepsilon=0.9\)) the kept set shrinks and both conventions approach one-hot
