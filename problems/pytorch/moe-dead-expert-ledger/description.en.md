# MoE: Load EMA and the Dead-Expert Ledger with Hysteresis

## Task

Implement `solve`: given the token counts each expert received at each step, keep a load EMA and run a **hysteresis state machine** per expert — how many consecutive steps below the death line make an expert dead, and how many consecutive steps above the revival line are required to bring it back. Return the EMA, the per-step alive/dead state, the live run counter, plus three ledgers: first death step, number of flips, and total dead steps.

This is the most hand-written piece of MoE training observability: expert collapse (routing collapse) is not "one step with zero load" but "a stretch without work", and if death and revival share a single threshold, states flap back and forth whenever the load jitters around it — hence the double threshold in practice. Both pieces are made pure functions here: no hidden state, no RNG, the input is a stretch of history. Parameter re-initialization and the weight copies of an expert reload are out of scope.

## Inputs

- `load`: CPU float64 `[S,E]`, non-empty 2-D, finite, element-wise non-negative. Row $s$ holds the token counts of the $E$ experts at step $s$.
- `beta`: a Python real (`bool` does not count), $0\le \beta<1$, the EMA retention rate.
- `death_threshold`, `revive_threshold`: Python reals (`bool` does not count), both $\ge 0$ and finite, with $\text{revive}\ge\text{death}$ (an inverted pair raises `ValueError`).
- `patience`, `revive_patience`: Python `int`, $\ge 1$ (`bool` does not count; values above $S$ are allowed).

Noncontiguous tensors are accepted and no input is mutated.

## Exact rules

Every expert runs the following independently, starting **alive** with both counters at 0 and the EMA at 0.

**EMA**: **the recurrence is applied at every step**, with no warm-up exemption:

$$
\mathit{ema}^{(0)}=0,\qquad
\mathit{ema}^{(s)}=\beta\,\mathit{ema}^{(s-1)}+(1-\beta)\,\mathit{load}[s].
$$

So the very first step already gives $\mathit{ema}^{(0)}=(1-\beta)\,\mathit{load}[0]$.

**State machine**: after updating the EMA at step $s$, advance only the counter belonging to the **current state**:

$$
\begin{aligned}
\text{alive state:}\ & \mathit{miss}\ \leftarrow\begin{cases}\mathit{miss}+1,&\mathit{ema}^{(s)}<\text{death}\\ 0,&\text{otherwise}\end{cases}
\qquad \mathit{miss}\ge\text{patience}\ \Rightarrow\ \text{go dead}\\[4pt]
\text{dead state:}\ & \mathit{back}\ \leftarrow\begin{cases}\mathit{back}+1,&\mathit{ema}^{(s)}\ge\text{revive}\\ 0,&\text{otherwise}\end{cases}
\qquad \mathit{back}\ge\text{revive\_patience}\ \Rightarrow\ \text{go alive}
\end{aligned}
$$

Three literal conventions:

- Death uses a **strict** $\mathit{ema}<\text{death}$, revival uses **greater-or-equal** $\mathit{ema}\ge\text{revive}$. An EMA inside $[\text{death},\text{revive})$ therefore accumulates on neither side — that is the hysteresis band.
- Only one branch runs per step: a death and a revival can never happen back to back within the same step.
- On a flip, **both counters go to zero**, so `streak` is necessarily 0 on the step where a flip happens.

**Ledger**: `dead[s]` records the state at step $s$ **after the state machine has run**; `streak[s]` records the live counter of that state (`miss` while alive, `back` while dead); `first_dead[e]` is the step where it first went dead, or $-1$ if it never did; `state_changes[e]` counts flips in both directions; `dead_steps[e]` counts the steps where `dead` is true.

## Return value

`(ema, dead, streak, first_dead, state_changes, dead_steps)`:

| Item | Shape / type | Meaning |
| --- | --- | --- |
| `ema` | float64 `[S,E]` | EMA load after each step's recurrence |
| `dead` | bool `[S,E]` | Alive/dead state at the end of each step |
| `streak` | int64 `[S,E]` | Run counter of the current state |
| `first_dead` | int64 `[E]` | First death step, `-1` if never dead |
| `state_changes` | int64 `[E]` | Number of flips (deaths and revivals both count) |
| `dead_steps` | int64 `[E]` | Number of steps spent dead |

Compare floats with `rtol=atol=2e-11` and every other tensor bit for bit.

## Example

$S=8$, $E=3$, `beta=0.5`, `death_threshold=1.0`, `revive_threshold=2.0`, `patience=2`, `revive_patience=2`:

```python
load = [[4, 0, 0],
        [4, 0, 0],
        [4, 0, 0],
        [4, 0, 3],
        [4, 0, 3],
        [4, 0, 3],
        [4, 0, 3],
        [4, 0, 3]]

# ema          s0..s7 = e0: [2, 3, 3.5, 3.75, 3.875, 3.9375, 3.96875, 3.984375]
#                         e1: all 0       e2: [0, 0, 0, 1.5, 2.25, 2.625, 2.8125, 2.90625]
# dead         [[0,0,0], [0,1,1], [0,1,1], [0,1,1], [0,1,1], [0,1,0], [0,1,0], [0,1,0]]
# streak       [[0,1,1], [0,0,0], [0,0,0], [0,0,0], [0,0,1], [0,0,0], [0,0,0], [0,0,0]]
# first_dead [-1, 1, 1]   state_changes [0, 1, 2]   dead_steps [0, 7, 4]
```

Step by step:

- **e0** stays far above the death line every step, so `miss` is always 0 and it never leaves alive. Note that s0's EMA is $0.5\times0+0.5\times4=2$: the zero initial value makes the first step systematically low, here just not low enough.
- **e1** never gets any load: s0's EMA is 0 ($<1$), one miss; s1 reaches 2 = `patience` → **dead**, with `streak` back to 0 on that step. Its EMA stays 0 forever, never reaching the revival line 2, so s1..s7, seven steps in total, are dead.
- **e2** also dies at s1 after two misses. At s2 the load is still 0, so `back` does not move. At s3 the load 3 lifts the EMA to 1.5, which sits inside the band $[1,2)$: it counts as **neither a miss nor revival progress**, so `back` stays 0. Only s4's EMA of 2.25 starts revival progress (`back=1`), and s5's 2.625 pushes `back` to 2 = `revive_patience` → **alive** again, `streak` 0 on that step. It spent s1..s4, four steps, dead.

With `beta=0.5` and integer loads, every EMA value is an exact binary fraction, so the table is checkable by hand.
