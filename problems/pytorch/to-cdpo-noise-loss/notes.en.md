# Solution Notes · cDPO

## Noise model: symmetric flips

cDPO assumes each preference label is flipped to the opposite one with the **same** probability $\varepsilon$, independent of the sample (symmetric noise). Under this model the probability of observing "chosen wins" is $(1-\varepsilon)\sigma(x)+\varepsilon(1-\sigma(x))$, whose negative log is the soft-label cross-entropy. So $\varepsilon$ has a very direct meaning: **what fraction of the data you believe is wrong**.

## Why ε is capped at 0.5

$\varepsilon>0.5$ would mean the observed labels are less trustworthy than random guessing; the sensible move then is to invert all labels rather than keep smoothing. Mathematically $\varepsilon$ and $1-\varepsilon$ are exchanged symmetrically in the loss, so $\varepsilon>0.5$ equals flipping the data and using $1-\varepsilon$. Hence the range is $[0,0.5]$; $\varepsilon=0.5$ is the extreme "labels carry no information": the soft labels are symmetric, the loss is minimized at $x=0$ with value $\ln 2$, and the model is only pushed toward being unable to distinguish — never toward one side.

## Stability and the two degenerate points

In the $\log\sigma(-x)$ term, $\sigma(-x)\to 0$ for large $x$, so a direct `log(sigmoid())` underflows to $-\infty$. The stable form $\log\sigma(z)=\min(0,z)-\log(1+e^{-|z|})$ is safe for all $z$. Two degenerate points are pinned: $\varepsilon=0$ matches DPO bit-for-bit; at $x=0$, $\log\sigma(0)=-\ln 2$ on both sides, so the weighted sum is $-\ln 2$ regardless of $\varepsilon$ — the tests sweep $\varepsilon\in\{0,0.1,0.25,0.5\}$.

## Relation to robust-DPO theory

Ray Chowdhury et al. prove that under symmetric label noise, the optimum of the smoothed DPO objective differs from the clean-data optimum by a bounded bias that grows with $\varepsilon$ — consistent with the intuition that smoothing only adds a reverse pull on already-separated samples. cDPO is thus a **conservative** correction: it does not try to identify which labels are wrong, it just caps the absolute scale of the logit via uniform soft labels.
