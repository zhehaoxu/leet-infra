# Solution Notes · GKD

## Why a generalized JSD can "tune" the KL direction

Plain JSD is $\frac12\mathrm{KL}(p\|m)+\frac12\mathrm{KL}(q\|m)$, symmetric in the two distributions. GKD replaces the weights by $\beta$ and $1-\beta$, giving a whole family: for small $\beta$, $m$ is essentially $p_s$, the second term $\mathrm{KL}(p_s\|m)$ is suppressed by its own weight, and the dominant term is $\mathrm{KL}(p_t\|p_s)$ — **forward KL**, which pushes the student to cover every continuation the teacher allows (mode-covering). As $\beta\to1$ the dominant term becomes $\mathrm{KL}(p_s\|p_t)$ — **reverse KL**, the student chases only the teacher's dominant mode (mode-seeking). LM distillation usually prefers reverse KL (less rambling), but tightening too early loses diversity; $\beta$ is the knob.

Strictly, it is $JSD/\beta$ and $JSD/(1-\beta)$ that converge to those KLs at the endpoints; setting $\beta=0$ or 1 directly also zeroes the other arm's coefficient, collapsing the loss to 0. Hence the open-interval contract.

## The mixture must be computed in log space

$m=\beta p_t+(1-\beta)p_s$ is defined in probability space, but $p$ comes from a softmax and can be tiny; computing $p_t(t-\log m)$ directly loses precision as $m\to0$. Using $\log m=\mathrm{logsumexp}([t+\log\beta,\ s+\log(1-\beta)])$ moves the mixture into log space, avoiding both underflow and the double rounding of "compute $m$ then log it".

## Temperature enters JSD only; no τ² on CE

Temperature scaling sharpens or flattens both distributions, which the distillation term needs. But the hard-label CE uses the student's **raw** logits — a label target is not a distribution, so temperature has no meaning there. Precisely because CE is unscaled, there should be no $\tau^2$ either: the $\tau^2$ in Hinton-style KD compensates for the $1/\tau^2$ shrinkage when *both* sides are divided by temperature. Here only one side is, so multiplying by $\tau^2$ would over-scale the JSD gradient.

## λ and data mixing

In the paper $\lambda$ corresponds to data mixing: some fraction of offline sequences (teacher-generated or ground truth), the rest on-policy student sequences. In the implementation it shows up as one scalar weight — $\lambda=1$ is pure SFT, $\lambda=0$ pure distillation, and this problem checks both endpoints. `ignore_index` must be excluded from the averages of **both** terms, otherwise the CE denominator disagrees with the JSD one.
