# Solution Notes · Min-p Sampling

## Core idea

Min-p is neither "take the first k" nor "accumulate up to p"; it sets **a floor relative to the highest probability**:
keep only $p_i \geq p \cdot p_{\max}$. When the model is confident, $p_{\max}$ is large → the floor is high → it cuts aggressively;
when perplexed, $p_{\max}$ is small → the floor is low → it is lenient. One hyperparameter adapts to confidence.

## Equivalent criterion: logit domain

Because softmax is monotone:
$$p_i \ge p \cdot p_{\max} \iff \text{logit}_i \ge \text{logit}_{\max} + \ln p$$
The llama.cpp implementation operates directly in the logit domain to avoid computing softmax first. This implementation is written in the probability domain; the two are mathematically equivalent.

## Inverse transform sampling

Same as top-k: on the CDF of filtered (original order, 0 at non-kept positions) take the first position with value $\ge u$.

## Comparison with top-p / top-k

| Method | Retention criterion | For flat distributions | For sharp distributions |
|------|---------|-----------|-----------|
| top-k | fixed count k | too many noisy tokens are still kept | may cut useful candidates |
| top-p | cumulative mass p | large truncation nucleus | small truncation nucleus |
| **min-p** | p·p_max floor | high floor → cuts more | low floor → keeps more |

The three are complementary; in llama.cpp's default pipeline they are **chained**: top-k coarse filtering → top-p → min-p fine-tuning.
