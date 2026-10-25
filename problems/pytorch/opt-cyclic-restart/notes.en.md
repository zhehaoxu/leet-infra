# In Depth: SGDR Cosine Warm Restarts and the Cycle Ledger

## Source
- Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, arXiv:1608.03983
  (equation (3): \(\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos(\frac{T_{cur}}{T_i}\pi))\), together with the
  statement that the period length is multiplied by \(T_{mult}\) after every restart, and \(\eta_{max}\) being the
  group's initial lr).
- Reference implementation: PyTorch's own `torch/optim/lr_scheduler.py::CosineAnnealingWarmRestarts`
  (`T_i=T_0`/`T_cur=last_epoch` in `__init__`, the cosine in `get_lr`, and inside `step()` the order
  `T_cur+=1` immediately followed by `if T_cur >= T_i: T_cur %= T_i; T_i *= T_mult`).
  The native class automatically takes one `step()` during construction (`last_epoch=-1` lands on \(T_{cur}=0\),
  lr=base_lr), which lines up with this problem's "construction is already step 0" convention, so the two can be
  compared bit for bit.
- The arXiv:1608.03983 recorded in todos.md #90(4) is correct (verified against the paper text and the native code).

## Core points and pitfalls
1. **Restart first, then write lr**: when `T_cur` reaches `T_i` it is immediately taken modulo and `T_i` multiplied by
   `T_mult`, and only then is lr computed from the **new** \((T_{cur},T_i)\). Hence \(T_{cur}/T_i\) never equals 1 and
   **the trough \(\eta_{min}\) is never emitted**; writing "lr first, restart second" adds a spurious \(\eta_{min}\) at
   the end of every period and shifts all restarts one step late.
2. **`T_cur % T_i`, not "set to 0"**: with `T_mult=1` the incremental form is equivalent to zeroing, but an
   implementation that locates the cycle by division while forgetting to grow `T_i` misaligns everything as soon as
   \(T_{mult}>1\). Period lengths are a **running product**.
3. **Restart points are not multiples of \(T_0\)**: with \(T_{mult}=2,T_0=3\) they sit at
   \(t=0,3,9,21,\dots\) (gaps \(3,6,12,24\)), i.e. \(T_0(T_{mult}^n-1)/(T_{mult}-1)\). The graded restart ledger
   targets exactly this.
4. **Construction is already step 0**: \(lr(0)=\) base_lr (the period peak); do not advance one extra step and read
   \(t=0\) as \(T_{cur}=1\).
5. **\(T_{mult}=1\) is a legal degeneration**: the period stays constant, `T_i` never grows, and restarts land on
   multiples of \(T_0\). \(T_0=1\) is also legal: it restarts every step, so lr is constantly base_lr.
6. **No end, no clamping**: unlike OneCycle/WSD this scheduler takes no total step count; reading arbitrarily far
   only keeps lengthening the periods - it never pins lr at \(\eta_{min}\), which is only a lower bound.
7. \(T_i\) grows as an **integer product**, but \(T_{cur}/T_i\) must still be evaluated in float (the native code uses
   `math.pi * T_cur / T_i`).

## Cross-check convention
- External reference: the **native `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`**, same `SGD(lr=base_lr)`
  start and same read convention for \(t=0..29\), item-wise absolute error < 1e-12 (seven configurations:
  \((T_0,T_{mult})\) over \((3,2),(4,1),(5,2),(2,3),(1,2),(6,2),(7,1)\), including non-zero `eta_min` and base_lr=1.0).
- Independent recomputation: `_oracle` takes the closed-form route - integer accumulation
  \(S_n=\sum_{k<n}T_0T_{mult}^k\) locates which period step \(t\) falls in and \(T_{cur}=t-S_n\), simulating no
  recursive state at all; tolerance 1e-12 over 60 steps (spanning four periods).
- The example numbers were produced by actually running the reference solution (bit-identical to the native output).
- Degenerate cases / invariants: the restart ledger is exactly \([0,4,12,28]\) with gaps \(\{T_0T_{mult}^i\}\);
  \(\min(lr)>\eta_{min}\) with strict monotone decrease inside each period; and for \(T_{mult}=1\) the restarts are
  multiples of \(T_0\) with the trough value 0.015278640450004212 exactly reproducible.

## Complexity
One cosine evaluation plus a constant state update per step, i.e. \(O(\text{param groups})\); the number of periods
grows logarithmically with the step count (\(n\approx\log_{T_{mult}}(t/T_0)\)) and `T_i` is an integer product, so
nothing expands linearly with the number of steps.
