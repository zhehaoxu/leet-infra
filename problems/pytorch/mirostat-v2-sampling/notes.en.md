# Solution Notes · Mirostat v2

## The idea in one sentence

Maintain a surprisal threshold $\mu$: **truncate** the tail tokens more surprising than $\mu$ → **sample** → **feed back** the observed perplexity $\text{obs}$ of the drawn token to adjust $\mu$, so that the long-run perplexity converges to the target $\tau$.

$$s_i=-\log_2 p_i,\quad \text{keep the descending prefix until } s_i>\mu,\quad \mu'=\mu-\eta(\text{obs}-\tau)$$

- v2's simplification over v1: v1 has to maintain a dual feedback loop in the log domain, whereas v2 needs only **one** scalar $\mu$ and one "truncate → normalize → sample → update" cycle, which is why it is more popular in engineering practice.

## Truncation (the key part)

In C++, `cur_p` is already sorted by descending $p$, `std::find_if(-log2f(p) > mu)` returns the **first out-of-range** element, and `size = distance(begin, it)` is the length of the prefix to keep:

- If the violation happens at index $k$ → keep $k$ elements ($0..k-1$ are all within range).
- If the first element already violates ($k=0$) → `size==0` is clamped to 1, always keeping the most probable one.
- If none violate → keep everything.

This problem reproduces the descending order on the last dimension with `torch.sort(descending=True)`, then aligns exactly using `size = clamp(argmax(surp>mu) if any else n, min=1)`.

## Normalization / obs / feedback

Renormalizing the kept set yields `filtered`. Note that **obs uses the normalized** $\text{filtered}[\text{chosen}]$ (C++ performs two softmaxes: the original p before truncation sets the threshold, then the truncated set is softmaxed again, and `obs` is taken from the renormalized p). `mu_new = mu - eta*(obs - tau)`: the larger `obs` is (the more surprising), the smaller `mu_new` becomes → the harder the next truncation, and the closed loop pulls perplexity back to $\tau$.

## Consistency with the collection's contract

`filtered` is returned in the **original order**, and `chosen` is obtained by inverse transform over the original-order CDF (consistent with the other problems in the collection); one extra `mu_new` is returned to serve as the `mu` input of the next step. Under batching, each row is truncated independently and each row gets its own `mu_new`.
