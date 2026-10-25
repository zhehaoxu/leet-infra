# Knowledge Distillation Temperature Loss

## Problem

Knowledge distillation (Hinton et al., arXiv:1503.02531) makes a "student" network imitate the soft labels produced by a "teacher" network. The softened distribution uses a temperature \(T\) to flatten the logits: \(q_i=\mathrm{softmax}(z_i/T)\). The distillation loss is the reverse KL between the teacher's and student's soft distributions, multiplied by \(T^2\) to compensate for the \(1/T\) shrinkage of gradients during backpropagation, so that the relative weight of the soft and hard losses does not drift with \(T\).

\[ \mathcal{L}_{\text{KD}} = T^2 \cdot \mathrm{KL}\!\big(\text{softmax}(z_t/T)\,\big\|\,\text{softmax}(z_s/T)\big) \]

## Rules

Implement `solve(student_logits, teacher_logits, T, reduction="mean")`, returning a scalar (or a per-sample vector). `student_logits` and `teacher_logits` are float32 unnormalized scores of shape \((N, C)\), and `T>0` is the temperature. Compute per sample: divide the logits on both sides by \(T\) and apply log-softmax to obtain \(\log p_s,\ \log p_t\); let \(p_t=\exp(\log p_t)\); the per-sample KL is \(\sum_c p_{t,c}(\log p_{t,c}-\log p_{s,c})\), then multiply by \(T^2\). Reduce according to `reduction` (`"mean"`/`"sum"`/`"none"`). An invalid `reduction` must raise `ValueError`.

## Examples

\(N=1, C=3\), `student_logits=[[2.0,1.0,0.1]]`, `teacher_logits=[[1.0,2.0,0.5]]`:

```
T=2.0 -> 0.39207178
T=1.0 -> 0.40606672   (here it equals the KL(P_t‖P_s) between the softmax distributions)
student==teacher -> 0.0
```

## Constraints

- CPU float32; `T>0` (scalar float).
- The KL direction is \(\mathrm{KL}(\text{teacher}\|\text{student})\) (the student aligns to the teacher).
- The multiplier \(T^2\) is applied after the per-sample KL and before the reduction.
- At `T=1` the result should equal \(\mathrm{KL}(p_t\|p_s)\) computed on the original softmax distributions.
- Support all three `reduction` values; any other string raises `ValueError`.
