# In Depth: llama.cpp Full Pipeline Stage Ordering

## Source

- The algorithm: llama.cpp's sampling-chain design (the `llama_sampler_chain` assembly order: logit-bias/penalties → dry → top-k → top-p → min-p → xtc → temp → dist; exact file and line numbers **to be re-sourced**, commonly `common/sampling` and `src/llama-sampling.cpp`). The chain convention's core claim: **the probabilities read by truncation criteria are untempered** — temperature only reshapes the sharpness of the final distribution, never who gets truncated. HuggingFace's `LogitsProcessorList` habitually puts the temperature processor at the head of the chain, making the two ecosystems' top-p non-interchangeable. The judge environment is offline; **the stage order and per-stage semantics are defined entirely by the starter comments**.
- Each single-stage algorithm already has its own problem in this collection: `penalty-sampling` (repeat/frequency/presence penalties), `dry-sampling` (flush-against-the-end repetition penalty), `top-k-sampling`, `top-p-sampling`, `min-p-sampling`, plus this chain's `logit-bias-sampling` and `xtc-sampling`. Their **adjudication details** (equality directions, min_keep backstops, reversed-window matching) carry over verbatim from those accepted conventions; this problem's new subject is **the ordering** itself.
- One-line summary: **single-operator problems are "what to do"; the pipeline problem is "in what order"**. The same operator set with temperature in a different place yields a different top-p surviving set — ordering is part of the behavior and the largest source of engine-compatibility bugs.

## Full derivation and index convention

**Snapshot semantics**: the 8 rows of `masks` are fixed to stages 0–7; row \(i\) = the surviving set after stage \(i\) ran. A hidden corollary: with `temp_first=True` temp runs **early**, but its snapshot still lands on row 7 — so row 7's surviving set is **not guaranteed** to be the tail of the row-6 chain: under the HF convention row 7 is "temp just ran, truncations have not" (in the judge's example temp=0.5 removes nothing, so row 7 shows the full 9), while rows 3–6 are the tempered truncation chain. That is exactly the shape the judge's `hf_vs_llama_temperature_order` assertion probes.

**Temperature × truncation interaction** (the mathematical core of this problem): the truncation criteria (top-p's accumulation, min-p's floor, xtc's threshold) act on the softmax of "the current surviving logits". Dividing logits by \(1/T\) (\(T>1\)) flattens the softmax: head probabilities drop, tail probabilities rise. Therefore:

- llama convention (temp last): truncations see the **original sharpness** — heat never makes top-p keep more candidates;
- HF convention (temp first): truncations see the **flattened distribution** — for the same \(p=0.6\), tempered head tokens reach the threshold later... or collapse to a single token (the judge's example uses \(T=0.5\), sharpening, and top-p collapses to 1 token).

So `temp_first` is a **behavior switch**, not an implementation detail: the judge runs both conventions on the same parameters and asserts the top-p rows differ, and that under the llama convention the temp row \(\subseteq\) the xtc row (temp runs last and can only delete).

**Shared window/matching conventions**: `penalty_last_n` and `dry_penalty_last_n` both use "negative = whole history, 0 = off, positive = trailing \(n\)"; dry runs longest-common-prefix matching on the **reversed** window, equivalent to "the repeat must sit flush against the window's end" — verbatim the convention accepted by `dry-sampling` (including "only strictly longer repeats register" and no penalty for zero-length matches when `dry_allowed_length=0`).

**The three min_keep backstops**: top-p cuts where "first \(\ge p\) and at least \(mk\) kept"; min-p stops at "first below the floor with at least \(mk\) kept"; xtc requires "at least \(mk\) survive the cut". top-k is the only unprotected truncation (llama.cpp convention) — which is why row 3 dares to cut 9 → 8 in the example.

**Final inverse transform**: `probs` normalizes over the survivors, CDF in original order; the hit test is **the first prefix sum \(\ge u\)** (equality hits), same as the sibling problems; the fallback takes the last survivor in descending order (so a hair-thin float prefix never throws).

## Key points and common mistakes

1. **Ordering bugs are caught by snapshots**: comparing only the final distribution cannot catch swaps like "bias before penalties" (both mutate logits; the distribution may barely change). The judge compares all 8 snapshot rows with `torch.equal` — any misordered row betrays itself immediately.
2. **The truncations' temperature domain**: with `temp_first=False` the top-p/min-p/xtc probabilities **exclude** temperature. Applying the softmax early with temperature is the natural leakage of HF habits; the dedicated assertion (the two conventions' top-p rows differ) targets exactly that.
3. **Row 7's snapshot timing** (see derivation): under HF, row 7 is not "the end of the chain" but "temp just executed". Implementing it as "execute last, snapshot last" makes row 7 show the 1–2 post-truncation survivors, contradicting the assertion.
4. **Penalties' multiply/divide directions**: \(x\le0\) multiplies by repeat, \(x>0\) divides (pushing positive logits down and negative ones up — symmetric de-peaking), then uniformly subtracts \(c\cdot\text{freq}+\text{present}\). Uniform multiplication, uniform division, or a miscounted freq/present all flip row 1's snapshot.
5. **dry's flush-against-the-end constraint**: the repeat must hug the window's end (reversed matching implements it); repeats in the middle of the window do not count; per token only the longest repeat is penalized, updated only when "strictly longer" — identical to `dry-sampling`; do not invent a new convention here.
6. **xtc keeps the tail**: cut everything in the qualifying head prefix **except the last** — xtc's semantics are "thin out the clumped head, keep the one at the boundary". Cutting the other way (keep head, drop tail) fails the random comparison instantly.
7. **banned masks, never edits logits**: does a banned token's logit participate in min-p's "max logit"? No — it is not in the surviving set, and every statistic "normalized over survivors" excludes it naturally. Implementing ban as logit = −inf is a different convention and shifts min-p's floor and top-p's cumulants.
8. **`min_keep` does not protect top-k**; and every backstop means "then don't cut", never "cut and refill afterwards".

## Differences from neighboring problems

- **vs the single-operator problems (`top-p/top-k/min-p/dry/penalty/logit-bias/xtc-sampling`)**: each isolates one stage with no "chain" in its inputs; this problem's input is the **full knob set** and the subject is inter-stage **ordering and interaction** (temperature placement, truncation domain, snapshot timing). Single-operator problems never compare multi-row mask snapshots.
- **vs `mirostat-v2-sampling` / `adaptive-p-sampling` / `dynatemp-sampling`**: those are **adaptive** samplers (thresholds/temperature react to generation feedback); here every knob is a constant — no feedback loop, just one assembly and one pass.
- **vs `inf-spec-decoding-rejection` (speculative-decoding rejection sampling)**: both consume "a distribution plus random numbers", but speculative verification decides **accept/reject of draft tokens** (a target/draft likelihood ratio) with no truncation chain; this problem is the **single-step decode pipeline after one forward pass**.
- **vs `contrastive-sampling` / `moa-sampling`**: those remix candidate distributions (contrastive search, Mixture-of-Agents) and never maintain a "surviving set"; this problem's intermediate state is boolean masks, not convex combinations of distributions.

## Cross-check convention and complexity

- Main criterion: the independent recompute `_oracle` — pure-Python list simulation step by step (no Z-algorithm; dry uses a naive reversed LCP; no vectorized softmax — recomputed over survivors at every step), compared against the reference (torch-vectorized) over 10 random-logit trials × {temp_first} × {xtc on/off} × {temp} × {top_p} × {min_p} full grids: masks `torch.equal`, probs `allclose(1e-5)`, chosen `torch.equal`.
- Dedicated assertions: HF vs llama top-p surviving-set difference (same parameters); "every knob off = raw softmax" (the pipeline's degenerate anchor); under the llama convention temp row ⊆ xtc row.
- Complexity: each stage \(O(V\log V)\) (sorting) or \(O(V)\) (scans); whole chain \(O(8\cdot V\log V)\); snapshots are 8×V booleans — the same order as a real engine's per-step overhead, comfortably runnable on CPU.
