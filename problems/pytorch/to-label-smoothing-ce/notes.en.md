# Solution Notes · Label Smoothing Cross-Entropy

## Deriving the Two-Term Form from Soft-Label Cross-Entropy

The target distribution $q$ is "$1-\alpha$ on the correct class, $\alpha/V$ on every other class". Substituting into $\mathcal{L}=-\sum_v q_v\log p_v$:

$$-\Big[(1-\alpha)\log p_y + \tfrac{\alpha}{V}\sum_v \log p_v\Big] = (1-\alpha)(-\log p_y) + \alpha\cdot\tfrac{1}{V}\sum_v(-\log p_v)$$

Note the smoothing term averages over **all $V$ classes** (the correct class is counted twice: once in the NLL and once in the mean) — this is exactly why the statement uses `logp.mean(-1)` rather than "excluding the correct class".

## The Connection to KL

$\frac{1}{V}\sum_v(-\log p_v) = \mathrm{KL}(u\,\|\,p) + \mathcal{H}(u)$, where $u$ is the uniform distribution and $\mathcal{H}(u)=\log V$ is a constant. So label smoothing is equivalent to "standard CE + $\alpha\cdot\mathrm{KL}(\text{uniform}\|p)$", nudging the prediction toward uniform and penalizing over-confidence — the same origin as `kl_div` in operators.

## Numerical Stability

Always use `F.log_softmax` (which subtracts the row maximum and then does logsumexp internally). If you write `torch.log(F.softmax(x))`, large logits make softmax overflow/underflow to 0, and log yields $-\infty$. The tests use `1000.0`-scale logits and "all-equal logits → loss=ln V" to verify stability and correctness.

## Self-check at α=0

When $\alpha=0$ it must be bitwise equal to `F.cross_entropy(reduction='mean')` — this is the strongest correctness anchor, and the tests assert it specifically.
