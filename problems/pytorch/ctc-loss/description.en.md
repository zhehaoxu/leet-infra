# CTC Loss (Forward Algorithm)

## Problem

Connectionist Temporal Classification (Graves et al., ICML 2006) lets an RNN/Transformer output variable-length sequences without needing alignment. It reserves a blank symbol `blank` and maps a per-step emission path to the target label by "merging consecutive repeats, then deleting blanks". Given the log probability of each class at each time step, the CTC loss is the negative log of the sum of probabilities of all paths that collapse to the target label:

\[ \mathcal{L} = -\log \sum_{\pi:\,\mathcal{B}(\pi)=\ell} \prod_{t} y_{t,\pi_t} \]

Accumulating directly in log space with dynamic programming (the forward variable \(\alpha\)) avoids enumerating the paths exponentially.

## Rules

Implement `solve(log_probs, labels)` and return a 0-D scalar tensor (i.e. the negative log-likelihood \(-\log p(\ell\mid x)\)). `log_probs` is a float32 **log** probability of shape \((T, V)\) (each row is the `log` of a softmax distribution), and the blank symbol is fixed to be the last class, `blank = V-1`; `labels` is int64 of shape \((L,)\) holding the target class indices (excluding blank). Use the standard CTC forward algorithm: expand the labels into \(\ell'=[\text{blank},\ell_1,\text{blank},\ell_2,\ldots,\text{blank}]\) (length \(L'=2L+1\)), recurse over time steps for \(\alpha_t(k)\), allowing a direct jump without passing through blank when adjacent classes repeat; the terminal state is the sum of \(\alpha_T(L'-1)\) and \(\alpha_T(L'-2)\), and return its negation.

## Examples

`probs=[[0.3,0.6,0.1],[0.5,0.2,0.3],[0.1,0.4,0.5]]` (\(T=3, V=3\), `blank=2`), `log_probs=log(probs)`:

```
labels=[0]    -> 1.78379130
labels=[0,1]  -> 1.77195680
```

## Constraints

- CPU float32; each row of `log_probs` is the log of some softmax distribution (including negative values).
- `blank = V - 1`; `labels` takes values in \([0, V-2]\) (excluding blank).
- The recursion must combine terms in log space with a stable log-add-exp to avoid underflow.
- The return value is the scalar negative log-likelihood; when the labels are empty (\(L=0\)), the loss equals the negative log of the "all-blank path".
