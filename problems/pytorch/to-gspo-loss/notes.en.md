# Solution Notes · GSPO

## The sequence ratio is the geometric mean of token ratios

$\prod_t\frac{\pi_\theta(o_t)}{\pi_\text{old}(o_t)}=\frac{\pi_\theta(y)}{\pi_\text{old}(y)}$, so the length-normalized sequence ratio $s=\big(\pi_\theta(y)/\pi_\text{old}(y)\big)^{1/|y|}$ is **exactly** the geometric mean of the per-token ratios. Two immediate consequences: implementation-wise, sum first, divide by the length, and `exp` once — numerically equivalent to "exp per token then geometric mean" but far more stable; and any single token's ratio can move $s$ only by $1/|o|$ — one token's distribution shift in a long answer cannot blow up the whole answer's ratio, which is precisely the smoothness GSPO wants.

## Why token-level clipping is unstable on MoE

Token-level PPO's surrogate clips per token, equivalent to assuming "each token's ratio can be adjusted independently". That approximation is tolerable for dense models; in MoE each token routes to different experts, routing may change after one step, and a token's $\rho_t$ can jump from 1 to tens — triggering clipping, truncating the gradient, and making the training signal noisy and biased. The sequence ratio averages the drift of $T$ tokens and makes the clipping decision once, substantially lowering variance on MoE. This is the paper's central empirical finding.

## Changing the clipped object changes the semantics

GSPO clips the ratio of the **whole answer**: $\min(s A,\ \mathrm{clamp}(s)A)$. Two behaviors differ from token level: all tokens of an answer share one clip/no-clip decision, so there is no fragmented gradient of "half the tokens clipped, half not"; and since $s$ is a geometric mean typically very close to 1 ($\epsilon$ is a tiny $3\times10^{-4}$), most samples never clip at all — clipping is a backstop for when the policy has drifted substantially.

## Contrast with DAPO/GRPO

All three optimize $\min(\rho A,\text{clip}\cdot A)$; the difference is the domain of $\rho$: GRPO/DAPO use a token-level $\rho$ averaged over tokens, GSPO uses a sequence-level $s$ averaged over answers. DAPO's token-level global mean insists "every token weighs equally", while GSPO returns entirely to "every answer weighs equally" — the two give opposite answers to "how should length influence the weight", and seeing them together best conveys this design dimension.

## Numerical details

`log s` must sum only over valid tokens and divide by the valid length; mixing padding deltas into the numerator, or using the tensor width as the denominator, makes $s$ depend on the amount of padding. The advantage is a sequence-level scalar of shape $(N,)$, neither centered nor standardized (that is the job of upstream RLOO/GRPO); this problem only covers "given advantages, compute the loss".
