# Solution Notes · DAPO

## Clip-Higher and entropy collapse

PPO's clipping pins $\rho$ inside $[1-\epsilon,1+\epsilon]$. For positive-advantage samples the upper bound directly limits "how fast a low-probability token's probability may rise"; once many tokens are pushed down together, policy entropy drops quickly, sampling approaches greedy decoding, and exploration stops. DAPO observes that in large-model RL entropy collapse tends to start from **a few over-suppressed tokens**, so it loosens the upper bound alone ($\epsilon_\text{high}$ often 0.28 versus $\epsilon_\text{low}$ 0.2). Note this only affects "how much can be raised"; the negative-advantage side is not loosened — for $A<0$, $\min(\rho A,\bar\rho A)$ picks the more negative unclipped term and the upper bound never participates.

## Dynamic sampling: why a whole group can be dropped

In-group advantages are centered; when $acc_b=0$ (all wrong) or $1$ (all correct) the std is 0 and every advantage is 0, so every token's surrogate in that group is 0. Such groups contribute exactly zero gradient yet occupy a slot and consume sampling budget. DAPO filters them out and spends the saved compute on more useful groups. Because the aggregation is a **global token average**, dropping entire groups does not change the relative weights of the kept ones — with "average per answer, then combine", deleting a group would reweight the others, which is why "group filtering + token-level averaging" must be used together.

## Why token-level loss matters

Per-answer averaging gives every answer the same total weight, so an individual token in a long answer weighs only $1/|o|$ of one in a short answer. In RL this systematically weakens the learning signal of tokens in long sequences; the model leans toward writing less, or fails to learn key decisions late in long outputs. DAPO pools all kept tokens into one average, so every token weighs the same and length no longer implicitly tunes the gradient.

## Relation to GRPO / Dr. GRPO

DAPO's advantage still uses GRPO's "group centering + divide by std", only with different ranges for numerator and denominator (filtered groups, global token pool). Contrast with Dr. GRPO: Dr. GRPO argues division by std causes difficulty bias and removes it outright, while DAPO keeps the std but deletes "uninformative groups" via filtering — attacking the same problem from another direction. Keeping all three conventions in one collection separates the two independent knobs "normalization" and "sampling selection".

## Degenerate and boundary cases

When the whole batch is filtered, $N_\text{valid}=0$, and the loss is defined as 0 rather than NaN — real training would skip such a batch, but the function itself must return a definite value. Rewards must be strictly 0/1 (not "approximately binary") because the filter is the strict inequality $0<acc_b<1$; float rewards would let an "all-correct" group slip through as $1-10^{-17}$.
