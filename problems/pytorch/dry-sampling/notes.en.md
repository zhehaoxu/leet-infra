# Solution Notes · DRY Sampling

## What It Penalizes

DRY aims to break **n-gram loops**. Given the last $N$ tokens of the context (`last`), we want to know:
"treating the last few tokens as a pattern, how many times did it appear earlier?" If the trailing `a b c` has already
appeared verbatim before, then the token emitted after another `a b c` would **continue** this loop, so we penalize that
"next token." The penalty grows **exponentially** with the repetition length, and a sufficiently long loop is penalized to the point of being nearly impossible to appear again.

## Reverse Z-Algorithm

llama.cpp reverses `last` (`rat(j)` = the $j$-th from the end), then runs the **Z-algorithm**:
$z[k]$ = the longest common prefix of `rev` and `rev[k:]`. This is exactly "the length of the prefix that still matches after anchoring at the end and shifting back by $k$"
—i.e., a repetition that can be aligned backwards from the current position. The Z-algorithm uses a pair of cursors `[lt, rt]` (the current Z-box) to bring the naive $O(N^2)$ down to $O(N)$:

```
for k in 1..N-1:
    if k > rt:                 # outside the box, naive extension
        n = 0; while k+n<N and rev[n]==rev[k+n]: n+=1
        z[k] = n; if n>0: lt,rt = k, k+n-1
    else:                      # inside the box, reuse the computed z[k-lt]
        p = k - lt; right = rt - k + 1
        if z[p] < right: z[k] = z[p]
        else: continue extending from rt+1 → z[k] = i-k; lt,rt = k, i-1
```

This problem permits a naive implementation (a while comparison for each $k$); judging uses an **independent brute-force longest common prefix** as the oracle,
and the two paths must agree value by value.

## Converting to the Penalized Token (common pitfall)

In C++, step3 uses `rat(last_n_repeat - 2 - i)`, where `dry_repeat_count[i]` stores the $z$ value for offset
$k = (N-1) - i$. After substitution and simplification, when shift $k$ hits, the penalized token is exactly
`last[N - k]` in **chronological order** ($k = 1..N-1$). Be sure never to write an off-by-one such as `last[N-1-k]`: that penalizes the wrong
position and the whole problem fails. Verify repeatedly with the two hand-computed examples (see the statement).

- `history=[5,6,7,5,6,7]`: only $k=3$ has $L_3=3$ meeting the threshold → `last[6-3]=last[3]=5`.
- `history=[0,1,0,1,0,1]`: $k=2$ has $L_2=4$ meeting the threshold → `last[6-2]=last[4]=0`.

## Other Details

- The same token hit by multiple shifts: take the **largest** repetition length (highest power, heaviest penalty).
- `max_exponent` prevents floating-point overflow in `base**exp`: it is only enabled when `base > 1.000001`, and is not needed for the normal examples.
- Early exits: `mult==0`, `base<1`, `last_n==0`, and $N\le$ `allowed`—these four cases return the input unchanged.
