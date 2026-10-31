# Block Diffusion Sampling: One Commit Step With Two Quota Regimes

## Problem Description

A block diffusion language model (non-causal parallel decoding in the BD3-LM family) does not emit one token at a time: it lays the \(L\) positions to be generated into a block and iterates \(N\) times. At every step the network offers a candidate token and a confidence for the whole block, and the scheduler decides **which positions get committed this round**; the rest are looked at again next round. There are only two sources of commitment criteria — confidence above threshold, and the quota that spreads the block over the steps.

diffusers' `BlockRefinementScheduler.step` implements two **different** quota regimes inside the same function, selected by whether a mask token id is passed:

- **Mask filling** (`mask_token_id` given): whether a position is currently a mask decides if it can be committed, and this step's quota is the per-step quantity sitting in **cell \(t\)** of the table that spreads \(L\) over \(N\) steps.
- **Uniform corruption** (`mask_token_id=None`): positions start out filled with random tokens, and the scheduler keeps track of which ones are already committed (`committed`); the quota is how many positions should be committed **cumulatively** after the first \(t+1\) steps, and the difference is what this step must add.

For most \((L,N)\) the two regimes commit a different number of positions per step: with \(L=10,N=6\) the first gives \([2,2,2,2,1,1]\) while the second's cumulative targets are \([2,4,5,7,9,10]\), i.e. increments \([2,2,1,2,2,1]\) — the prefix sums of the first are \([2,4,6,8,9,10]\), one too many at steps 3 and 4. Swap them and the block converges with the wrong rhythm.

This problem lifts that single discrete decision out of the loop: the network sampling is already done (`sampled_tokens`, `sampled_probs` are handed to you) and the random re-noising is already done (`random_tokens` is handed to you); you only compute the new block plus the two commit/editing index maps. No logits, no softmax, no random numbers.

## Rules

Let \(B\) be the batch, \(L\) the block length, \(t\) the `timestep`, \(N\) the number of steps, \(p_i\) the \(i\)-th entry of `sampled_probs`, \(\tau\) the `threshold` and \(\tau_e\) the `editing_threshold`. Every comparison involving \(p\) **first downcasts \(p\) to float32**.

### Mask filling mode (`mask_token_id` is not `None`)

The per-step quota table puts the remainder first:

\[ q_i=\Big\lfloor\frac{L}{N}\Big\rfloor+1\ (i<L\bmod N),\qquad q_i=\Big\lfloor\frac{L}{N}\Big\rfloor\ (i\ge L\bmod N),\qquad \text{quota}=q_{\min(t,\,N-1)} \]

\(A=\{i:\ \text{sample}_i=\texttt{mask\_token\_id}\}\) is the set of undecided positions in a row; a position outside \(A\) compares as \(-\infty\) (not 0). Rows are handled independently:

\[ S=\{i\in A:\ p_i>\tau\},\qquad
\text{transfer}= \begin{cases} S & |S|\ge \text{quota}\\[2pt] \text{the first }\min(\text{quota},|A|)\text{ of }A\ (p\ \text{descending, ties by ascending index}) & |S|<\text{quota} \end{cases} \]

`transfer` is identically empty for the whole batch only when **the block holds no mask at all** (\(A=\varnothing\)); a row without masks only blanks that row.

Editing applies only to positions that are **already committed and not part of the prompt**. First the switch:

\[ \text{editing is on}\iff \tau_e\neq\texttt{None}\ \text{and}\ \tau_e>0 \]

When it is on, \(\text{editable}=\neg A\wedge\neg\texttt{prompt\_mask}\) (a `prompt_mask=None` counts as all `False`) and

\[ \text{editing}=\text{editable}\wedge(p_i>\tau_e)\wedge(\texttt{sampled\_tokens}_i\neq \text{sample}_i) \]

Finally `prev_sample` starts from `sample` and takes `sampled_tokens` wherever `transfer | editing` fires; every other position is **kept as is** (a mask stays a mask).

### Uniform corruption mode (`mask_token_id` is `None`)

\[ \text{base}= \begin{cases} \varnothing & t=0\\ \texttt{committed} & t>0 \end{cases},\qquad
\text{target}=\Big\lceil \frac{(t+1)L}{N} \Big\rceil=\frac{(t+1)L+N-1}{N}\ \text{(integer division)},\qquad
\text{needed}=\max\big(0,\ \text{target}-|\text{base}|\big) \]

Ranking treats every position inside \(\text{base}\) as \(-\infty\) and shoves it to the back; the rest go by \(p\) descending (ties by ascending index), ranks start at 0:

\[ \text{transfer}=\neg\text{base}\wedge\big(\text{rank}<\text{needed}\ \vee\ p_i>\tau\big) \]

\[ \text{editing}=\text{base}\wedge(\texttt{sampled\_tokens}\neq\text{sample})\wedge(p_i>\tau_e)
\quad\text{on whenever }\tau_e\neq\texttt{None}\text{ (}0\text{ and negatives still enable it)} \]

Let \(\text{after}=\text{base}\vee\text{transfer}\):

\[ \text{prev\_sample}_i= \begin{cases} \texttt{sampled\_tokens}_i & \text{transfer}_i\ \text{or}\ \text{editing}_i\\ \text{sample}_i & \text{after}_i\ \text{but neither fired}\\ \texttt{random\_tokens}_i & \neg\text{after}_i \end{cases} \]

`committed` and `random_tokens` are completely ignored in mask filling mode, and `prompt_mask` is completely ignored in uniform corruption mode.

### Five places people write backwards

1. **The two quotas are not interchangeable**: mask mode eats the **per-step quantity** \(q_{\min(t,N-1)}\), uniform mode eats the **cumulative target** \(\lceil (t+1)L/N\rceil\) and then subtracts what is already committed. Writing the latter as the prefix sum of the former is wrong — for \(L=10,N=6\) the prefix sums are \([2,4,6,8,9,10]\) while the cumulative targets are \([2,4,5,7,9,10]\).
2. **When \(|S|\ge\) quota the commit set is every above-threshold position**, not "the first quota of them"; \(q_i=0\) (the tail when \(N>L\)) still takes this branch, so a single above-threshold position commits.
3. **An out-of-range \(t\) is not an error**: after \(\min(t,N-1)\) the last cell's quota is used; in uniform mode \(t\ge N\) pushes target past \(L\), so `needed` sweeps up every uncommitted position at once.
4. **The \(\tau_e\) switch is asymmetric between the modes**: mask mode requires \(\tau_e>0\), so \(0.0\) and negatives **disable** editing; uniform mode only checks `is not None`, so \(0.0\) and negatives **enable** it.
5. **The final `random_tokens` overwrite uses `after`** (positions committed this very step are already protected), and a `committed` passed at \(t=0\) must be ignored.

## Examples

### Example 1: mask filling

```
L = 6, N = 4  ->  floor(6/4)=1, 6%4=2  ->  quota table = [2, 2, 1, 1]
t = 1  ->  quota = 2
mask_token_id = 9, threshold = 0.9, editing_threshold = 0.6
prompt_mask   = [False, False, True, False, False, False]

sample        = [[9, 9, 3, 9, 7, 9],
                 [9, 9, 9, 9, 9, 9]]
sampled_probs = [[0.93, 0.40, 0.99, 0.61, 0.88, 0.12],
                 [0.10, 0.93, 0.30, 0.95, 0.91, 0.05]]
sampled_tokens= [[4, 8, 1, 6, 2, 0],
                 [0, 5, 2, 7, 3, 8]]
```

Row 0 has \(\{0,1,3,5\}\) undecided and only index \(0\) (0.93) clears the threshold — \(|S|=1<2\), so the top-\(k\) branch runs and takes the best 2 of \(\{0,1,3,5\}\) by \(p\), namely \(0\) (0.93) and \(3\) (0.61). Row 1 is undecided everywhere, three positions clear the threshold (\(\{1,3,4\}\)), \(|S|=3\ge2\), so **all three commit**. Editing only looks at non-mask positions: in row 0, position \(2\) has \(0.99>0.6\) and a changed token but `prompt_mask` blocks it; position \(4\) has \(0.88>0.6\) and \(2\neq 7\) → edit. Row 1 has no non-mask position, so its editing map is empty.

```
prev_sample           = [[4, 9, 3, 6, 2, 9],
                         [9, 5, 9, 7, 3, 9]]
transfer_index        = [[True, False, False, True, False, False],
                         [False, True, False, True, True, False]]
editing_transfer_index= [[False, False, False, False, True, False],
                         [False, False, False, False, False, False]]
```

Changing `t` to 5 (\(>N-1\)) still takes the table's last cell, quota 1, so row 0 with \(|S|=1\ge1\) commits only \(\{0\}\); changing `editing_threshold` to `0.0` empties the editing maps here, because the switch demands \(>0\).

### Example 2: uniform corruption

```
L = 6, N = 4, t = 2  ->  steps_done=3, target = (18+3)//4 = 5
threshold = 0.9, editing_threshold = 0.5, mask_token_id = None

committed     = [[True, True, False, False, True, False],
                 [False, False, False, False, False, False]]
sample        = [[7, 2, 9, 4, 5, 1],
                 [6, 3, 0, 8, 2, 4]]
sampled_probs = [[0.95, 0.20, 0.88, 0.61, 0.30, 0.72],
                 [0.10, 0.55, 0.70, 0.25, 0.92, 0.40]]
sampled_tokens= [[1, 9, 3, 6, 8, 2],
                 [5, 7, 4, 0, 9, 3]]
random_tokens = [[1, 8, 6, 2, 3, 9],
                 [7, 1, 5, 8, 0, 6]]
```

Row 0 already has 3 committed → needed = 2; among the uncommitted \(\{2,3,5\}\) the ranking by \(p\) is \(2\)(0.88) rank 0, \(5\)(0.72) rank 1, \(3\)(0.61) rank 2 → the top two enter `transfer`, and \(3\) neither clears the threshold nor wins quota. Editing on already-committed positions: \(0\) has \(0.95>0.5\) and \(1\neq7\) → edit; \(1\) (0.20) and \(4\) (0.30) fail \(\tau_e\). Row 1 gets needed = 5, so the worst-ranked position \(0\) (0.10) misses and the other five commit.

```
prev_sample           = [[1, 2, 3, 2, 5, 2],
                         [7, 7, 4, 0, 9, 3]]
transfer_index        = [[False, False, True, False, False, True],
                         [False, True, True, True, True, True]]
editing_transfer_index= [[True, False, False, False, False, False],
                         [False, False, False, False, False, False]]
```

`prev_sample[0][3]=2` comes from `random_tokens`: row 0 column 3 did not commit this step, so re-noising overwrote it (its old value was 4). `prev_sample[0][1]=2` and `prev_sample[0][4]=5` stay as they are, because they sit in `committed` but did not trigger editing.

## Constraints

- `sampled_tokens`, `sample`, `random_tokens` must be integer tensors (not bool, not floating point) with values \(\ge 0\); `sampled_probs` must be a finite floating point tensor with values in \([0,1]\); `committed` and `prompt_mask` must be bool tensors.
- The first three must share shape \((B,L)\) with \(B\ge1\), \(L\ge1\); when non-`None`, `committed`/`random_tokens` must be \((B,L)\) and `prompt_mask` must be \((L,)\).
- Uniform corruption mode (`mask_token_id is None`) requires `committed` and `random_tokens`; mask filling mode may omit them (they are unused anyway).
- `timestep`, `num_inference_steps`, `mask_token_id` must be Python ints (`bool` is illegal, `mask_token_id` may be `None`); \(t\ge0\), \(N\ge1\), \(\texttt{mask\_token\_id}\ge0\).
- `threshold` must be a finite real number; `editing_threshold` is `None` or a finite real number.
- The three return values `(prev_sample, transfer_index, editing_transfer_index)` all have shape \((B,L)\) with dtypes int64, bool, bool; compared element-wise, no tolerance. Any violated input constraint raises `ValueError`. `import diffusers` is forbidden.
