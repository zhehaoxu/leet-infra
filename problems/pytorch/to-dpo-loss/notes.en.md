# Solution Notes · DPO Loss

## The single step from RLHF to DPO

The standard RLHF objective $\max_y r(y,x) - \beta\,\mathrm{KL}(\pi_\theta\|\pi_\text{ref})$ has the closed-form optimum $\pi^*(y|x)=\frac{1}{Z}\pi_\text{ref}(y|x)e^{r/\beta}$. Solving back for $r=\beta\log\frac{\pi^*}{\pi_\text{ref}}+\beta\log Z$ and substituting into the Bradley–Terry preference model $P(y_w\succ y_l)=\sigma(r_w-r_l)$, the partition function $Z$ cancels and one obtains the DPO loss containing only the policy/reference log-ratios. That is why the "implicit reward" in DPO, $=\beta\log\frac{\pi_\theta}{\pi_\text{ref}}$, is the origin of ① − ② in this problem.

## Why a stable log-sigmoid is mandatory

Writing $\log\sigma(x)$ directly as `torch.log(torch.sigmoid(x))` underflows sigmoid to 0 for very negative $x$ and log then gives $-\infty$. The identity $\log\sigma(x)=\min(0,x)-\log(1+e^{-|x|})$ guarantees $e^{-|x|}\le 1$ so nothing overflows. `F.logsigmoid` computes it exactly this way internally, and the oracle reproduces it with Python's `math.log1p`.

## Handling the sequence logprob

In a real implementation chosen_logp is "the sum of the answer's token logprobs", and a response mask must restrict the sum to the generated part (the prompt is not counted). This problem moves that aggregation outside and hands you the sequence-level logprob directly, focusing on assembling the loss and avoiding a full causal-LM forward pass.

## The role of β

The larger $\beta$, the larger the logit scale, and the closer sigmoid gets to a step function — stricter about "chosen must clearly beat rejected", which also means a tighter KL constraint toward the reference model. At $\beta=0$ the loss degenerates to the constant $\ln 2$ (no gradient); the tests cover that boundary explicitly.
