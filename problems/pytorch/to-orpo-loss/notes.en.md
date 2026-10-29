# Solution Notes · ORPO

## Why no reference model is needed

DPO requires a frozen reference model $\pi_\text{ref}$ to define the implicit reward and the KL constraint. ORPO's observation: while the **absolute likelihood** of the chosen answer is still low, comparing relative improvements of chosen/rejected is of limited value. So it puts the SFT term — "raise the chosen likelihood" — directly into the loss as the anti-drift regularizer, and lets the preference term only separate the **odds** of the two answers. Both terms share one model and one forward pass, eliminating the reference model's memory and compute.

## Why odds instead of probabilities

The log-odds $\ell-\log(1-e^\ell)$ live on the logit scale: they diverge as $p\to 1$, so samples where the chosen answer is already almost certain still contribute a sizeable gradient and the model does not stall when sigmoid saturates. The price is numerical fragility — you must use `log1p(-exp(ℓ))`; computing $p$ then $\log\frac{p}{1-p}$ destroys the significant digits as $\ell$ approaches 0. The contract validates that the mean logprob is strictly negative precisely because $\ell\ge 0$ makes $1-p\le 0$ and the odds undefined.

## Where length normalization happens

ORPO normalizes **before** taking odds: first $\ell=\text{logp}/|y|$, then the odds difference. Normalizing later (or not at all) systematically depresses long answers, whose logprobs are more negative. The test that feeds the same sequence logprob with different lengths — the SFT term must change accordingly — pins this step down.

## Contrast with DPO/IPO

The preference terms of the three losses act on three functions of $\beta\Delta$: DPO uses $-\log\sigma(\beta\Delta)$ (monotone, saturating), IPO uses $(\Delta-\frac{1}{2\beta})^2$ (a well-defined target margin, penalizing both directions), and ORPO uses $-\log\sigma(\text{logodds}_w-\text{logodds}_l)$ (odds scale, no reference model). Keeping all three in one collection is exactly to see "where the anchor comes from and how the loss shape governs the gradient".
