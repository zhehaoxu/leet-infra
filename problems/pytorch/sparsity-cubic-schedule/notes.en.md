# Deep Dive: Cubic Gradual Sparsity Schedule (GSL)

## Origin
Gradual sparsity, proposed by Zhu & Gupta in *To Prune, or Not to Prune* (arXiv:1710.01878): within the pruning window a cubic function \(p_t=p_f+(p_i-p_f)(1-\frac{t-t_0}{t_1-t_0})^3\) smoothly advances the sparsity rate; it is widely adopted by DeepSpeed/NVIDIA and others.

## Core Ideas and Pitfalls
1. **Cubic, not linear**: \((1-\text{frac})^3\) makes the sparsity rise quickly early and stabilize later — its values differ from a linear schedule, which is the most commonly tested point of distinction.
2. **It is `final + (initial-final)*…`**: at \(t_0\) it takes \(p_i\) and at \(t_1\) it takes \(p_f\); swapping the start and end flips the monotone direction.
3. **Clamp outside the window**: \(t\le t_0\) is always \(p_i\) and \(t\ge t_1\) is always \(p_f\); never extrapolate the cubic formula out of bounds.
4. The denominator is \(t_1-t_0\) (the window length), not the total number of training steps.

## Cross-Checking Against the Oracle
The reference solution does branch clamping + cubic interpolation. The judge's `_oracle` independently recomputes the same closed form in pure Python (different structure, numerically equivalent), `abs(got-exp)<1e-5`. Additional assertions cover "\(t=t_0\to p_i\), \(t=t_1\to p_f\)", "constant at that value when \(p_i=p_f\)", and "monotone midway".

## Complexity
\(O(1)\).
