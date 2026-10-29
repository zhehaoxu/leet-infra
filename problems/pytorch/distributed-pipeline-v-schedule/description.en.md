# V-Shaped Placement: Pipeline Task Timeline and Activation Memory

## Scope

Split the model into $pv$ logical chunks on $p$ physical stages, with $v$ chunks per device. Virtual chunks do not create extra hardware: **a device executes at most one task per slot**. Build a dependency-respecting training timeline and measure activation lifetimes.

This is an explicit educational policy: **V-shaped/folded placement with ready-backward priority**, not a reproduction of Megatron interleaved 1F1B or ZBV. Actual zero-bubble methods separate input-gradient and weight-gradient work; backward is indivisible here, and zero bubbles are not guaranteed.

## Placement and dependencies

Chunks are indexed in forward order $c=0,\ldots,pv-1$ and micro-batches by $b=0,\ldots,m-1$. With $q=\lfloor c/p\rfloor$ and $r=c\bmod p$:

$$owner(c)=\begin{cases}r,&q\text{ even}\\p-1-r,&q\text{ odd}.\end{cases}$$

$v=1$ is a single sweep, $v=2$ is a V, and further sweeps alternate directions. For $p=2,v=2$, chunk owners are `[0,1,1,0]`.

Every chunk/micro-batch has one forward task $F(c,b)$ and one backward task $B(c,b)$, each lasting one slot:

- $F(0,b)$ has no dependencies; other forward tasks wait for $F(c-1,b)$.
- $B(c,b)$ waits for its own forward and, except at the final chunk, for $B(c+1,b)$.
- Communication takes zero time, but dependencies become usable only in the **next slot**, even between chunks on the same device.

## Deterministic policy

At each slot start, devices simultaneously choose from tasks whose dependencies completed in earlier slots:

1. Prefer a ready backward task, ordered by ascending micro-batch ID, then descending chunk ID.
2. Otherwise choose a ready forward task, ordered by ascending micro-batch ID, then ascending chunk ID.
3. Idle if no task is ready. All selected tasks complete together at the slot end.

Stop when all backward tasks complete, without trailing all-idle slots. Simply taking the union of virtual-chunk busy windows would hide physical-device conflicts.

## Time and activation accounting

For timeline length $T$, there are exactly $2mpv$ useful tasks:

$$util=\frac{2mpv}{pT},\qquad bubble=1-util.$$

One chunk/micro-batch activation occupies $a=L/(pv)\cdot act\_bytes$ bytes. Save it when forward completes and release it when backward completes. Track the maximum live count at slot boundaries separately on each device and multiply by $a$.

The baseline is an **all-forward-then-all-backward flush schedule using the same placement**, with peak $mva$ per device. Savings are the sum of baseline per-device peaks minus the sum of scheduled per-device peaks, **not the difference between simultaneous whole-cluster peaks**. Parameters, gradients, communication buffers and operation scratch space are excluded.

## Example

Use `p=2, m=2, v=2, L=8, act_bytes=1.0`. `Fcb/Bcb` denotes chunk c and micro-batch b; `.` denotes idle.

```text
slot      0    1    2    3    4    5    6    7    8    9
stage 0  F00  F01   .   F30  B30  F31  B31  B00   .   B01
stage 1   .   F10  F20  F11  F21  B20  B10  B21  B11   .
```

There are 16 tasks in 20 device-slots: `makespan=10`, `util=0.8`, `bubble≈0.2`.
An activation occupies 2 bytes; device peaks are `[6.0,8.0]` versus flush peaks `[8.0,8.0]`, saving `2.0` bytes.
