# Top-p / Nucleus Sampling In Depth

## Why a "nucleus" instead of "Top-k"

Top-k always keeps k tokens, but the entropy of a language model's next step varies enormously: when
predicting a proper noun the distribution is very peaked (1–2 candidates suffice), while in free-form text
it is very flat (dozens are needed). **Nucleus sampling** adapts: it takes the smallest prefix whose
cumulative probability first reaches `p`, so the nucleus is naturally small for peaked distributions and
naturally larger for flat ones — the "mass" of the truncation set stays constant while its "size" floats.

## Meaning of the test `cumsum - prob < p`

We want "keep elements up to and including the one where the cumulative sum first reaches ≥ p". An
equivalent form: for the `i`-th element (in descending order), if the cumulative sum **before** it,
`cumsum_i - sorted_prob_i`, is still `< p`, then the cutoff point has not been reached yet and the element
is kept. This makes exactly the "first token that crosses p" included (otherwise we might keep nothing or
keep less than mass p), and it always keeps at least the Top-1 (no matter how small `p` is, 1 is kept).

## Inverse-transform sampling (Inverse-CDF)

Given the normalized in-nucleus distribution and a uniform number $u\in[0,1)$, taking the first position on
the in-nucleus CDF with value $\ge u$ completes the sampling — this is the standard way of writing
"sampling according to probabilities" as "a deterministic function + external noise", in the same spirit as
the Gumbel problem's "move the randomness outside", and it also makes judging reproducible (`u` is an
explicit input).

## The combination used in practice

Real decoding often **stacks several constraints**: temperature-scale the logits, then top-k truncation,
then top-p truncation, and finally multinomial sampling; it may also include `min_p` (dropping tokens below
`p·max_prob`), `repetition/frequency penalty`, and `typical sampling`. This problem implements the most
central of these, top-p + sampling; understanding it is enough to read the `samplers` modules of those
engines.

## Numerical caveat

In floating point the final CDF entry may be slightly less than 1, so a very large `u` finds no matching
position; the implementation falls back to the last kept element inside the nucleus when there is no hit,
avoiding an out-of-range access.
