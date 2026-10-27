# Solution Notes · Acceptance-Rate Accounting

## Expected tokens in two lines

\(e(\gamma)\) needs no probability tree; write it as an accumulation of prefix products: keep a running probability `prod` of "the first \(i\) positions all hit", multiply by `alpha[i]` at each step and add it into \(e\). The base term 1 is the bonus token—the target always samples one token after the end of the draft, and that token is what makes speculation never a loss.

Note that "the \(i\)-th term is the first \(i\) positions all hitting": once position \(i\) is rejected, a lucky guess at \(i+1\) no longer counts because the sequence has already diverged. Hence the product rather than a single \(\alpha_i\).

## The denominator is time, not steps

In \(1 + \gamma c\), the 1 is one target forward and \(\gamma c\) is the per-token draft cost. Two common mistakes:

1. Billing the draft as a single forward (\(1 + c\)) hugely overestimates the speedup;
2. Forgetting that drafts are autoregressive (assuming \(c \ll 1\) is negligible)—a small model may have \(c = 0.05\), which already consumes half the time at \(\gamma = 10\).

## Shape of the optimal gamma

The numerator converges exponentially to \(1/(1-\alpha)\) while the denominator grows linearly, so \(s(\gamma)\) rises then falls. Large \(\alpha\) and small \(c\) push the optimum later; with \(c \ge 1\) (a draft no cheaper than the target) not speculating at all wins.

Exact ties are common: with \(c = 1\) and \(\alpha = 1\) every term equals 1, so "no speculation" and "speculate" are equivalent, and only the "smallest \(\gamma\)" convention keeps results reproducible. `argmax` returning the first maximum matches that convention for free.

## Boundaries

- \(\alpha = 0\): \(e(\gamma)\equiv 1\) while the denominator keeps growing, so \(s\) decreases monotonically and the optimum is \(\gamma = 0\)—a draft that is never accepted is worthless.
- \(\alpha = 1\): \(e(\gamma) = 1+\gamma\) and \(s = (1+\gamma)/(1+\gamma c)\); \(\gamma\) saturates when \(c<1\) and collapses to 0 when \(c>1\).
- Per-position rates can differ: earlier positions usually score higher, so using an average underestimates the relative advantage of short drafts. This is why the input is a per-position vector rather than a single \(\alpha\).

## Related problems

- `spec-typical-acceptance`: turns accept/reject into a per-token threshold rule—the algorithmic side of the same payoff account.
- `inf-spec-decoding-rejection`: rejection sampling for a single draft, responsible for per-step correctness; this problem covers the multi-step expected payoff.
- `spec-sequoia-depth-bounded-dp`: folds draft-tree depth/width into the optimization, generalizing this one-dimensional \(\gamma\) search.
