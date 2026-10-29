# Notes: The Branch Condition in Compensated Summation and Why Non-finite Data Must Degrade

## Origin

Kahan published compensated summation in a 1965 internal technical note; Babuska and Neumaier subsequently improved it into the form used here (Higham's *Accuracy and Stability of Numerical Algorithms* carries the full derivation). NumPy's `np.sum` defaults to pairwise on float64 (`math.fsum` takes a separate exact path), and CUDA's `cub::DeviceReduce` merges in blocks as well. The interview framings are usually: "why does loss accumulation drift late in training", "why can't all-reduce just be one flat tree reduce", "why does fp16 summation need a scale". Explaining where the error of each mode comes from is worth more than memorizing the update.

## What Each of the Four Modes Guards Against

The rounding error of a single double addition \(s+v\) is exactly "the low bits that got dropped", and every trick in compensated summation is about computing that piece and keeping it:

- \(c=(t-s)-y\) in `kahan` relies on Sterbenz's lemma: when \(|y|\le|s|\), \(t\) and \(s\) differ by less than a factor of two, so \(t-s\) is representable exactly and \(c\) is precisely the dropped part. It needs \(|y|\le|s|\) (the new element must not exceed the running sum), otherwise \(t-s\) swallows the low bits of \(y\) as well.
- `kahan_babuska` (Neumaier) branches explicitly on \(|s|\ge|v|\), writing \((s-t)+v\) in one case and \((v-t)+s\) in the other, so both "a big number eats a small one" and "small numbers pile onto a big one" get compensated. On the example \([10^{16},1,-10^{16},1]\) only this mode returns the true value 2.
- `pairwise` gives up per-step compensation and shortens the addition chain instead: a sequential accumulation of \(n\) elements has an error bound of \(n\cdot\epsilon\), while a bisection merge is only \(\log_2(n/\text{block})\) deep, giving roughly \((\text{block}+\log_2(n/\text{block}))\epsilon\). That is also why `block` exists — inside a block the accumulation is still sequential.

## Core Ideas and Pitfalls

- **Return `s` or `s+c`**: plain Kahan returns only \(s\) by definition (\(c\) feeds the next step), while Neumaier must return \(s+c\). Writing Kahan as \(s+c\) too makes it disagree with the expected example value and the judge fails immediately.
- **What the branch compares**: Neumaier compares \(|s|\ge|v|\) (partial sum versus addend), not \(|t|\) or \(|y|\); using \(|t|\ge|v|\) takes the wrong branch whenever \(t=s\).
- **Non-finite data must degrade to naive globally**: compensation necessarily computes `s - t`, and `inf - inf = nan`. Without a special path for \([+\infty, 1.0]\) a hand-written compensation returns NaN although the correct answer is \(+\infty\). Hence the rule "if the flattened data ever holds a non-finite value, use sequential accumulation", which also makes \(+\infty\) together with \(-\infty\) come out NaN naturally.
- **`pairwise` is not always better**: with `block=2` the two block sums cancel exactly and give 0.0, worse than naive's 1.0. That assertion is kept on purpose to kill the illusion that pairwise always wins — it improves an error bound, not luck.
- **Empty tensor**: with `numel == 0` all four modes return 0.0, and the recursive split of `pairwise` must not blow up.
- **Widen first**: float32 input has to be promoted before accumulating, otherwise the unit is 1e-7 and every integer relation in the example breaks.

## How It Differs from Neighboring Problems

`cumsum-scan` must keep every prefix sum (one compensation chain cannot cover all outputs), `reduce-axis` tests axis-reduction shape and broadcast semantics rather than precision, and `fp16-loss-scaling` is about a dynamic scale rather than addition-order error. This problem is unique in "same data, four modes, three different integers" plus the degradation agreement for non-finite input.

## Cross-Check Basis and Complexity

The oracle is `math.fsum` — CPython's Shewchuk implementation that tracks every magnitude in an array of partial sums and rounds once at the end, structurally different from all four reference paths. On well-conditioned data every mode must land inside 1e-12 relative error; on the example the results must be bitwise 1.0/1.0/2.0/0.0. The `beats_naive` case adds 1.0 to ten thousand copies of 1e-16: naive stays at 1.0 (1e-16 is far below ulp(1.0)=2.2e-16 and everything is swallowed), while Neumaier is within 1e-15 of fsum and strictly better than naive. Time \(O(n)\); the `pairwise` merge needs \(O(n/\text{block})\) extra space.
