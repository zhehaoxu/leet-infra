# In Depth: OneCycle LR and Antisymmetric Momentum

## Source
- Smith & Topin, *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates*, arXiv:1708.07120 (the 1cycle policy: a single lr peak, antisymmetric momentum, and a floor far below the starting rate).
- Reference implementation: PyTorch's own `torch/optim/lr_scheduler.py::OneCycleLR` (`_schedule_phases` / `_annealing_cos` / `_annealing_linear` / `get_lr`); fastai's `Learner.one_cycle` is the engineering version of the same policy.
- The arXiv:1708.07123 recorded in todos.md is wrong; the link in PyTorch's own docstring is **1708.07120**, and this problem follows that paper plus the native implementation.

## Core points and pitfalls
1. **Rise then fall, not monotone decay**: `pct_start` fixes where the peak sits (the up-phase is about `pct_start*total` long) and the down-phase then drops all the way to `floor = initial/final_div_factor`. `final_div_factor` is what lets the **floor sit far below the base learning rate** - the essential difference from warmup+cosine.
2. **Momentum is antisymmetric to lr**: while lr climbs from `initial` to `max_lr`, momentum must fall from `max_momentum` to `base_momentum`, and the other way round in the down-phase. Both share the **same pct** and the same anneal function, only with swapped endpoints. The graded invariant asserts exactly that "the lr peak and the momentum trough are at the same step".
3. **The bounds are floats**: `pct_start*total_steps - 1` is usually not an integer; integer division or rounding shifts the whole second phase (PyTorch itself uses `float(pct_start * total_steps) - 1`).
4. **Phase selection is `t <= end_i`**, with the first phase starting at `0.0` and the last phase as a fall-back; in three-phase mode the left ends of phases 2/3 are the bounds of phases 1/2.
5. **Construction is already step 0**: `OneCycleLR` writes the \(t=0\) lr/momentum at construction time (lr equals `initial`, momentum equals `max_momentum`); this problem keeps the same convention, do not advance one extra step.
6. With `cycle_momentum=False` you must **never** touch `group["momentum"]` (the judge checks it stays at `max_momentum`).

## Cross-check convention
- External reference: the **native `torch.optim.lr_scheduler.OneCycleLR`**, started from the same `SGD(lr=max_lr/div_factor, momentum=max_momentum)`, read under the same convention for \(t=0..\text{total}-1\), item-wise absolute error < 1e-12 (eight configurations: cos/linear x two/three phases x cycle_momentum on/off).
- Independent recomputation: `_oracle` uses the algebraically equivalent but differently-rounded `a+(b-a)*(1-cos(pi*pct))/2`, tolerance 1e-9, to catch "copied the native source yet still mis-split the phases".
- Degenerate / invariants: the final lr is below the initial lr, momentum always stays inside `[base_momentum, max_momentum]`, and with `cycle_momentum=False` momentum is untouched.

## Complexity
An \(O(\text{param groups})\) closed-form evaluation per step, no state tensors; locating the phase is a comparison over a constant number of segments.
