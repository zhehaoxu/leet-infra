# Solution Notes · RLOO

## Why the baseline introduces no bias

The policy gradient is $\mathbb{E}_{y\sim\pi_\theta}[\nabla\log\pi_\theta(y)\,r(y)]$. For any baseline $b$ that depends only on **other samples** and not on the current $y$,
$$\mathbb{E}_{y\sim\pi_\theta}[\nabla\log\pi_\theta(y)\,b]=\sum_y \nabla\pi_\theta(y)\cdot b=b\,\nabla\sum_y\pi_\theta(y)=0.$$
RLOO sets $b$ to the mean reward of the other answers in the group: it uses only other samples and thus never depends on $y_i$, so subtracting it leaves the gradient expectation unchanged while greatly reducing variance (the more correlated the in-group rewards, the bigger the reduction). This is why no value network is needed — one group sample supplies the baseline for free.

## Why leave-one-out is mandatory

With the group mean including self, $\bar r_b$, the quantity $A_{b,i}=r_{b,i}-\bar r_b$ contains $r_{b,i}$ twice, making the advantage negatively correlated with its own reward: rewards above the mean get shrunk and below the mean get lifted, compressing the magnitude by roughly $1/G$. The leave-one-out baseline excludes self, so for $G=2$ the advantage degenerates exactly to the raw difference $r_0-r_1$; for larger $G$ the two conventions diverge less but in the same direction. The test distinguishes them with $[1,0,3]$ at $G=3$: correct is $[-0.5,-2.0,2.5]$, while a denominator of $G$ gives $[-1/3,-4/3,5/3]$.

## The zero-sum property and its numerical consequences

$\sum_i A_{b,i}=\sum_i r_{b,i}-\frac{1}{G-1}\sum_i\sum_{j\ne i}r_{b,j}=\sum_i r_{b,i}-\frac{G-1}{G-1}\sum_j r_{b,j}=0$. Two uses: an implementation can compute all leave-one-out means at once as "(total − self)", avoiding a double loop; and it provides a data-independent self-check — if any group's advantages do not sum to zero, the baseline convention is wrong.

## Difference from GRPO

GRPO's in-group advantage is $(r-\bar r)/\mathrm{std}$ — both centered and standardized; RLOO only centers, and its baseline is leave-one-out. The former is insensitive to reward scale but amplifies noise by the in-group variance; the latter keeps the absolute reward scale. The collection places both side by side (see "Dr. GRPO + REINFORCE++") precisely to expose the two orthogonal choices "divide by std or not" and "baseline includes self or not".
