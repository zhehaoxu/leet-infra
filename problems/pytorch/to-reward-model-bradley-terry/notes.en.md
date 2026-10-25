# Solution Notes · Bradley-Terry Reward Model

## Pooling picks the "last valid token"

The reward is a single scalar for the whole response, but the backbone emits one output per token. The hidden state of the last valid token is used as the sequence representation because in a causal model the last real token has already aggregated information from the entire sequence (similar to using the end-of-sequence position for classification). The mask is not guaranteed to be contiguous (this problem uses `(mask*arange).max` to grab the largest valid index directly), which is more robust than "assuming padding is only on the right".

## Relation between Bradley-Terry and DPO

The loss here has the form $-\log\sigma(r_w - r_l)$; the only difference from DPO is that $r$ is the explicit output of a reward model, whereas DPO uses $\beta\log\frac{\pi_\theta}{\pi_\text{ref}}$ as an implicit reward — they are really the same logistic skeleton. The trained $r_\phi$ is exactly the model that scores responses in downstream PPO.

## Why use a stable log-sigmoid

The reward difference can reach fairly large positive/negative values, and `log(sigmoid(x))` underflows when $x$ is very negative. Use `F.logsigmoid` throughout ($\min(0,x)-\log(1+e^{-|x|})$); the oracle reproduces it independently with `math.log1p` so that a bug cannot silently pass.

## Note that the score is unbounded

The value head outputs a raw, pre-sigmoid reward scalar that can be positive or negative with no range; only the "difference between chosen and rejected" enters the loss, so adding a constant offset to every score does not affect the loss (it cancels out). `bias` mainly affects the baseline of downstream PPO rather than the ranking.
