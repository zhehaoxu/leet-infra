# Speculative Decoding: Per-Position Multi-Candidate Batched Verification with Residual Sampling

## Problem

Textbook speculative sampling (Leviathan/Chen, arXiv:2211.17192 / 2302.01318) verifies only **one** token per draft position; when the draft side can emit **several candidates per position** (multiple draft heads, an n-gram pool, beam expansion — SpecInfer's token tree has exactly such parallel sibling nodes per level, arXiv:2305.09781), each position should verify its \(C\) candidates **side by side, first come first served**, so that the probability of "accepting at least one" rises substantially. This problem implements a single iteration of Multi-Candidate Speculative Decoding (MCSD): \(\gamma\) positions, \(C\) candidates per position, batched verification + residual sampling.

Core mechanism: the local state of position \(j\) starts from copies of \(P[j]\) and \(Q[j]\); candidates \(i\) are tried in order with acceptance probability \(\mathrm{ratio}=\min(1,p[x]/q[x])\) (\(u\le\text{ratio}\) accepts, **equality included**); on rejection the residual distribution is updated to \(\mathrm{norm}(\max(p-q,0))\) (optional **without replacement**: remove the rejected candidate's draft mass from the local \(q\) and renormalize), and the residual is **reused by later candidates of the same position**; when all \(C\) candidates are rejected the iteration **terminates on the spot**, the final token is drawn from the **current residual**, and no later position is verified. If all \(\gamma\) positions are accepted, one bonus token is drawn from \(P[\gamma]\). Randomness is injected through the explicit uniforms `u` and `u_end`. CPU, no RNG calls anywhere.

## Rules

- The full signature, parameter semantics and empty-input conventions are written only in the starter's docstring — that is the sole judging contract. Inputs: `P` (g+1, V) target distributions (row \(g\) is the bonus row), `Q` (g, V) draft distributions, `cand` (g, C) candidate ids, `u` (g, C) acceptance coins, `u_end` scalar closing uniform, `no_replace` 0/1 switch.
- **Per-position sequential execution** (the local \(p,q\) of position \(j\) start as copies of \(P[j]\), \(Q[j]\)):
  1. Try \(i=0..C-1\) in order with \(x=\text{cand}[j][i]\): when \(q[x]>0\), \(\text{ratio}=\min(1,p[x]/q[x])\); when \(q[x]=0\), record **1 (must accept)** if \(p[x]>0\), **0** otherwise (only \(u=0\) accepts then). **Every attempt counts one trial** (rejected ones included).
  2. \(u[j][i]\le\text{ratio}\): output \(x\), set `picks[j]=i`, move to position \(j+1\); **residual and \(q\) modifications do not carry across positions**.
  3. Rejection: **two ordered steps** — first \(p\leftarrow\mathrm{norm}(\max(p-q,0))\) (element-wise subtraction, clamp negatives, divide by the sum; if the sum is 0 keep \(p\) unchanged); if `no_replace=1` then also \(q[x]\leftarrow0\), \(q\leftarrow\mathrm{norm}(q)\) (if the sum is 0 after removal, do not renormalize). A token that has been rejected once has zero mass in the new \(p\), so a repeated occurrence in `cand` can never be accepted afterwards.
  4. All \(C\) candidates rejected: record \(-1\) in `picks`, the iteration's output **terminates at position \(j\)**, the final token is drawn from the **current residual** (updated at most \(C\) times) by CDF reverse lookup, and **no later position is verified** (no tokens, no trials).
- **CDF reverse lookup** `cdf(d, w)`: return the **smallest** index \(k\) whose prefix sum is **strictly greater** than \(w\); if every prefix sum is \(\le w\), fall back to \(V-1\). The closing token (residual or bonus) uses `u_end`.
- **Return `(out, picks, trials)`**: `out` is a 1-D long tensor (length = number of accepted positions + 1, the last entry is always the closing token); `picks` records the winning candidate index per position (the last element is \(-1\) on exhaustion, in which case \(\text{len(out)}=\text{len(picks)}\); without exhaustion \(\text{len(out)}=\text{len(picks)}+1\)); `trials` is the total number of acceptance decisions (\(\le g\cdot C\); candidates after the truncating position are not counted).
- Empty inputs: with \(g=0\) it degenerates to plain decoding, output `[cdf(P[0], u_end)]`, empty `picks`, `trials=0`; \(C=0\) (with \(g>0\)) is equivalent to position 0 exhausting on the spot.
- Fully deterministic; calling any random number generator is forbidden.

## Example

Main example (\(\gamma=2,C=3,V=4\), every value is an exact multiple of 1/16):

\[P=\begin{bmatrix}0.5&0.125&0.25&0.125\\0.125&0.625&0.125&0.125\\0.25&0.25&0.25&0.25\end{bmatrix},\quad
Q=\begin{bmatrix}0.5&0.25&0.125&0.125\\0.25&0.25&0.25&0.25\end{bmatrix},\quad
\text{cand}=\begin{bmatrix}1&2&0\\0&0&2\end{bmatrix},\]
\[u=\begin{bmatrix}0.75&0.25&0.0\\0.75&0.5&0.25\end{bmatrix},\quad u_{\text{end}}=0.9,\quad \text{no\_replace}=0.\]

- Position 0: candidate 1 has \(\text{ratio}=0.125/0.25=0.5\), \(u=0.75>0.5\) → rejected → residual \(p=[0,0,1,0]\); candidate 2 has \(\text{ratio}=1/0.125>1\Rightarrow1\), \(u=0.25\le1\) → **accepted** → `picks[0]=1`, output token 2 (trials=2).
- Position 1 (fresh \(p,q\)): candidate 0 has \(\text{ratio}=0.125/0.25=0.5\), \(u=0.75\) → rejected → residual \(p=[0,1,0,0]\); candidate 0 appears again (\(p[0]=0\)) → \(\text{ratio}=0\), \(u=0.5\) → rejected; candidate 2 has \(p[2]=0\Rightarrow\text{ratio}=0\), \(u=0.25\) → rejected (trials=5) → all rejected, `picks=[1,-1]`, the closing token is looked up from residual \(p=[0,1,0,0]\) with \(u_{\text{end}}=0.9\) → token 1.
- Result: `out=[2,1]`, `picks=[1,-1]`, `trials=5`.

Contrast example (\(\gamma=1\), divergence between with/without replacement on the same data): candidate 0 (\(p[0]=0\)) is rejected for sure; with `no_replace=0` the \(q\) stays untouched, candidate 1 has \(\text{ratio}=0.5\) and \(u=0.375\) accepts → `out=[1,2]` (2 is the lookup of \(u_{\text{end}}=0.5\) on the uniform bonus row), `picks=[1]`, `trials=2`; with `no_replace=1` candidate 0's draft mass is removed and renormalized (\(q[1]\) grows from 0.25 to 0.5), candidate 1's \(\text{ratio}\) drops to 0.25 and \(u=0.375\) rejects → candidate 2 is a must-accept → `out=[2,2]`, `picks=[2]`, `trials=3`. Boundary: in the with-replacement mode, \(u[0][1]=0.5\) (exactly at the ratio) is still accepted, while \(0.5625\) falls through to candidate 2.

## Constraints

- \(0\le u\le1\), \(0\le u_{\text{end}}\le1\); `cand` entries may repeat, \(C\) may be 0; \(g\) may be 0
- All outputs must match an independent recompute step by step (`out`/`picks` via `torch.equal`, `trials` via integer equality); a 260-case random parameter sweep is cross-checked
- Three discriminative checks are asserted directly by the grader: with \(C=1\) every position collapses back to **textbook rejection sampling** (accept \(u\le\min(1,p/q)\), close from \(\mathrm{norm}(\max(p-q,0))\) — mutually corroborating with the criterion of `inf-spec-decoding-rejection`); \(u\) landing exactly on the ratio **accepts on equality**; `no_replace` only affects the denominators of later candidates within the same position (it never changes the update of \(p\), nor crosses positions)
