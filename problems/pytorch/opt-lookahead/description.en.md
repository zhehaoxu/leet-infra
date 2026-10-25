# Optimizer: Lookahead Fast-Slow Weight Wrapper (k Steps Forward, 1 Step Back)

## Problem

Changing the optimizer is not the only way to change the trajectory. Lookahead (Zhang et al., arXiv:1710.08983) wraps **any** base optimizer in a **fast/slow weight** pair: the base keeps updating the trainable **fast weights** \(\theta\) as usual, and every \(k\) fast steps a **slow weight** copy \(s\) absorbs a fraction \(\alpha\) of the gap \(\theta-s\), after which \(\theta\) is **pulled back onto \(s\)** and walks on — "k steps forward, 1 step back". The fast weights explore in large steps, the slow weights fold up the net displacement of each block of \(k\) steps, while the base optimizer's own momentum / second moments keep accumulating **without ever being wiped by the rollback**.

Implement that wrapper: **subclass `torch.optim.Optimizer`** and hold and drive a real base optimizer (momentum SGD or Adam) inside it. The judge builds tensors, hands them to your instance, drives it with a **fixed gradient sequence** through repeated `step()` calls, and compares the **parameter trajectory**, the **slow-weight trajectory**, and probes of the inner base optimizer's state buffers. CPU, float32, fully deterministic.

## Rules

- The factory takes `params` plus `base`, `lr`, `momentum`, `k`, `alpha`, `betas`, `eps` and returns an instance of a `torch.optim.Optimizer` subclass; `step()` updates the tensors in `params` **in place**. The exact signature, argument names and defaults live only in the starter comments and are the single source of truth.
- `base="sgd"` builds an inner momentum SGD (`weight_decay=0`, `dampening=0`, `nesterov=False`); `base="adam"` builds an inner Adam (`weight_decay=0`, `amsgrad=False`, using `betas` and `eps`); any other value raises. The instance must expose `.base_optimizer` (that inner optimizer object) and `.slow_weights` (a list of slow-weight tensors aligned one-to-one with `params`, same shape and dtype).
- Indexing and ordering convention: at construction the fast-step counter is \(t=0\) and each slow weight \(s \leftarrow\) a clone of the corresponding parameter. Every `step()` performs the three things **in this order**:
  1. **fast step**: let the base take a step, so \(\theta\) is updated in place by it. The base's own state (`momentum_buffer` for SGD; `exp_avg` / `exp_avg_sq` / `step` for Adam) accumulates as usual — a sync **never** zeroes, resets or rebuilds it;
  2. **advance the counter**: \(t \leftarrow t+1\);
  3. **sync only when due**: only when \(t\) is a positive multiple of \(k\), and for **every** parameter, in this order
     \[ s \leftarrow s + \alpha\,(\theta - s),\qquad \theta \leftarrow s \]
     i.e. the slow update comes **first** and the reset **second** — the two lines cannot be swapped, because only then does \(\theta\) land on the slow weights *after* this sync. The very first sync therefore happens **after** the \(k\)-th `step()` call, not at the start of call \(k+1\).
- The sync is triggered purely by \(t\) modulo \(k\), independently of whether a given parameter had a gradient this step: a parameter with `grad is None` is not updated by the base, yet is still pulled back onto its own slow weight when a sync is due.
- Everything is float32 on CPU and deterministic; using a ready-made Lookahead implementation is forbidden (instantiating the SGD/Adam that is being wrapped is fine — that is the point of a wrapper).

## Example

Single parameter \(p_0=[1.0,\,-2.0,\,0.5]\) with `base="sgd"`, `lr=0.1`, `momentum=0.9`, `k=2`, `alpha=0.5`, fed the four gradients
\(g_1=[0.40,\,-0.10,\,0.25]\), \(g_2=[-0.20,\,0.50,\,-0.10]\), \(g_3=[0.30,\,0.20,\,-0.35]\), \(g_4=[-0.50,\,-0.30,\,0.15]\).
\(\theta^-\) is where the base landed on that step, **before** the sync:

```
t=1  θ⁻=[0.960000, -1.990000, 0.475000]   no sync  → θ=[0.960000, -1.990000, 0.475000]   s=[1.000000, -2.000000, 0.500000]
t=2  θ⁻=[0.944000, -2.031000, 0.462500]   sync     → θ=[0.972000, -2.015500, 0.481250]   s equals θ
t=3  θ⁻=[0.927600, -2.072400, 0.505000]   no sync  → θ=[0.927600, -2.072400, 0.505000]   s=[0.972000, -2.015500, 0.481250]
t=4  θ⁻=[0.937640, -2.093610, 0.511375]   sync     → θ=[0.954820, -2.054555, 0.496312]   s equals θ
```

Note that the fast step at \(t=3\) starts from the post-sync \(\theta=[0.972,-2.0155,0.48125]\), not from \([0.944,-2.031,0.4625]\).
After the four steps the inner base momentum buffer is still `[-0.1004, 0.2121, -0.06375]` — neither rollback touched it.

## Constraints

- \(k\) is a positive integer, \(\alpha \in [0,1]\); `params` may mix 1-D (biases, norm gains) and multi-dimensional (weight matrices) tensors
- After every step both the training weights and the slow weights must match an independent recomputation element by element (floats with `atol=rtol=1e-5`; "just synced \(\Rightarrow\) \(\theta\) and \(s\) are bit-identical" with `torch.equal`; see notes.md for the cross-check convention)
- Three degenerations must hold (the judge asserts them directly): \(k\) larger than the total number of steps → the trajectory is point-by-point the bare base optimizer; \(\alpha=1\) → the sync is the identity on \(\theta\) (the slow weights are lifted onto \(\theta\) and written back unchanged); \(\alpha=0\) → the slow weights absorb nothing and each sync hard-rolls \(\theta\) back to the previous sync point; \(k=1\) → sync every step, which is exactly the base optimizer with every step size scaled by \(\alpha\)
