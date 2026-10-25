# Solution Notes · Switch Load-Balancing Auxiliary Loss

## Why $f_e \cdot P_e$

- $f_e$ is **not differentiable** (an integer count produced by argmax);
- $P_e$ is **differentiable** (the mean after softmax).

Their product has a gradient with respect to $P_e$: if some expert has a large $f_e$ (everyone is choosing it), the gradient pushes $P_e$ down; conversely, experts with small $f_e$ get pushed up. This is the classic trick of "using a hard metric as the weight and a soft metric as the thing being optimized".

If only $P_e$ were penalized alone (e.g. maximizing entropy), the gradient would be spread evenly over all tokens; if only $f_e$ were used, it would not be differentiable. The product form injects the "actual load" signal exactly into the gradient of the "routing distribution".

## Normalization details

Different implementations differ slightly:
- **HF Mixtral**: $f_e = \text{count}_e / T$ (denominator $T$), so $\sum_e f_e = k$;
- **Megatron-LM** `switch_load_balancing_loss_func`: $\tilde{f}_e = \text{count}_e / (T \cdot k)$ (denominator $T \cdot k$), so $\sum_e \tilde{f}_e = 1$.

The two differ by a factor of $k$. The statement follows the HF version; for $k=1$ (the scenario of the original Switch paper) they coincide.

## Lower bound analysis

From $\sum_e f_e = k$ and $\sum_e P_e = 1$, the Cauchy-Schwarz inequality gives:
$$ \sum_e f_e P_e \ge \frac{1}{E} \cdot k \cdot 1 = \frac{k}{E} $$
i.e. $E \cdot \sum_e f_e P_e \ge k$. For $k=1$ the lower bound is 1, matching the statement's example (perfectly balanced aux=1.0).

## Numerical details

`torch.bincount` only accepts a 1D `long`; flatten with `selected.reshape(-1)`. `minlength=E` guarantees a slot even for an expert selected 0 times.
