# GKD: On-Policy Distillation with a Generalized JSD

## Problem

**GKD** (Agarwal et al. 2024, *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*, arXiv:2306.13649, ICLR 2024) targets the **distribution mismatch** in sequence-level distillation: if the student is trained only on the teacher's offline outputs, it cannot recover once it drifts during inference (exposure bias). GKD instead has the student learn from the teacher on sequences the student itself generates, and measures the gap between the two next-token distributions with a **generalized JSD**:

$$D_\text{JSD}^{(\beta)}(p_t\,\|\,p_s)=\beta\,\mathrm{KL}(p_t\|m)+(1-\beta)\,\mathrm{KL}(p_s\|m),\qquad m=\beta\,p_t+(1-\beta)\,p_s .$$

$\beta$ interpolates continuously between "forward KL" (cover all teacher modes) and "reverse KL" (chase only the teacher's dominant mode), which is more controllable than committing to one KL. The total loss mixes the distillation term with a hard-label cross-entropy by $\lambda$: $L=(1-\lambda)D_\text{JSD}+\lambda L_\text{CE}$.

This problem gives a batch of **per-position student/teacher logits** plus hard labels; compute that loss.

## Rules

Given $N$ positions and vocabulary size $V$:

1. **Temperature-scale then take log-probabilities**: $s=\log\text{softmax}(\text{student}/\tau)$, $t=\log\text{softmax}(\text{teacher}/\tau)$, and $p_s=e^{s}$, $p_t=e^{t}$.
2. **Define the mixture in probability space**: $m=\beta p_t+(1-\beta)p_s$; implement it with the equivalent log form
   $\log m=\mathrm{logsumexp}\big([\,t+\log\beta,\ s+\log(1-\beta)\,]\big)$ to avoid underflow.
3. **Generalized JSD**: $D=\beta\sum_v p_t(t-\log m)+(1-\beta)\sum_v p_s(s-\log m)$.
4. **Cross-entropy uses the unscaled student logits**: $\text{CE}=-\log\text{softmax}(\text{student})[\text{label}]$.
5. Average each term over **valid-label** positions (`label != ignore_index`) separately, then combine
   $L=(1-\lambda)\overline{D}+\lambda\,\overline{\text{CE}}$, with **no $\tau^2$ factor**.
6. $\lambda\in[0,1]$, $\beta$ **strictly** in $(0,1)$, $\tau>0$; return a 0-dimensional float64 scalar.

## Examples

One position, vocabulary 2:

```
student = [2.0, 0.0]   teacher = [1.0, 0.0]   label = 0
λ = 0.5, β = 0.5, τ = 1.0

p_s = [0.880797, 0.119203]      p_t = [0.731059, 0.268941]
m   = 0.5·p_t + 0.5·p_s = [0.805928, 0.194072]
JSD = 0.5·KL(p_t‖m) + 0.5·KL(p_s‖m) = 0.018306
CE  = -log p_s[label=0]             = 0.126928
loss = 0.5 × 0.018306 + 0.5 × 0.126928 = 0.072617
```

Setting $\lambda$ to 0 leaves $0.018306$; setting it to 1 leaves $0.126928$. Raising the sharpness with $\tau=0.5$ gives $0.074601$, showing the temperature only affects the JSD path.

## Constraints

- Return a 0-dimensional float64 scalar; logits may be float32 and are promoted to float64 internally.
- `ignore_index` positions leave the averages of **both** JSD and CE (not just CE); at least one valid label is required.
- At $\beta=0$ or $1$ the formula above degenerates entirely to 0 (the mixture weight also zeroes the other KL arm), hence the open-interval contract; as $\beta\to0$, $JSD/\beta\to\mathrm{KL}(p_t\|p_s)$, and as $\beta\to1$, $JSD/(1-\beta)\to\mathrm{KL}(p_s\|p_t)$.
- Invalid input raises `ValueError`: non-tensor / logits not 2-D or shape mismatch / non-float / NaN·Inf / $V<1$ / labels not integer or shape mismatch / valid label out of range / no valid label / $\lambda$ out of range / $\beta\notin(0,1)$ / $\tau\le0$.
- CPU, deterministic, no random number calls.
