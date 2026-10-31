# Discrete diffusion sampling: the VQ-Diffusion reverse step assembles transition matrices in log space

## Problem Description

VQ-Diffusion (*Vector Quantized Diffusion Model for Text-to-Image Synthesis*, arXiv:2111.14822) never adds Gaussian noise in its forward process. Instead each latent pixel is rewritten as the **mask class** with probability \(\gamma_t\), and the mask is absorbing — once a pixel is covered it never uncovers. So one reverse step is a discrete Bayesian inversion:

\[ p(x_{t-1}\mid x_t)=\sum_{x_0} q(x_t\mid x_{t-1})\,\frac{q(x_{t-1}\mid x_0)\,p(x_0)}{q(x_t\mid x_0)} \]

diffusers' `VQDiffusionScheduler` does all of this in log space, and without a single matrix multiply: it builds a log one-hot from `sample`, then produces the whole column of transition probabilities with one `(onehot + a).logaddexp(b)` — the matching row gets \(\log(a+b)\) and every other row gets \(\log b\) (the source comment calls this "a cheeky trick"). The price is a pile of shape and index machinery: the single-step tables are length \(T\) while the cumulative ones are length \(T+1\), \(t=0\) has a shortcut that bypasses the posterior entirely, the mask row must be "saved before it is dropped", and \(\log 0\) is clamped to \(\log 10^{-30}\) rather than \(-\infty\).

Implement that one operator. Everything the scheduler would generate itself (the \(\alpha/\gamma\) tables, the Gumbel randomness) is passed in as an explicit argument; there is no codebook, no network forward pass, and no random number generation.

## Rules

Write \(C=\)`num_vec_classes`, \(k=C-1\) known classes (the mask class is always index \(k\)), and \(T=\)`len(at)`. The \(b\) tables are derived from \(a,c\), and a \(t-1\) superscript means an **array index**, not a mathematical time:

\[ b_t=\frac{1-a_t-c_t}{k},\qquad \bar b_t=\frac{1-\bar a_t-\bar c_t}{k}\qquad\Longrightarrow\qquad \bar a_t+k\,\bar b_t+\bar c_t=1 \]

The transition matrices (for one latent pixel, observed value \(j=x_t\)):

\[ q(x_t=j\mid x_0=i)=\begin{cases}\bar c_t & j=k\\ \bar a_t+\bar b_t & i=j\\ \bar b_t & \text{otherwise}\end{cases}\qquad q(x_t=j\mid x_{t-1}=i)=\begin{cases}c_t & j=k\\ a_t+b_t & i=j\\ b_t & \text{otherwise}\end{cases} \]

and \(q(x_t=j\mid x_{t-1}=k)=1\) when \(j=k\), otherwise \(0\) (absorbing state). The log one-hot is

\[ E[i]=\begin{cases}0 & x_t=i\\ \log(10^{-30})=-69.07755278982137 & \text{otherwise}\end{cases} \]

\(t=0\) is the shortcut: the logits are `model_output` returned **verbatim**, only \(k\) classes, the whole posterior is skipped. For \(t>0\) follow the five steps below (all in log space; \(\operatorname{LSE}\) is `logsumexp` along the class axis, `logaddexp` is elementwise):

\[ V[i]=\operatorname{logaddexp}(E[i]+\log \bar a_t,\ \log \bar b_t)\ \xrightarrow{\ x_t=k\ \text{: whole column becomes}\ \log \bar c_t\ }\ ,\quad i=0..k-1 \]

\[ U[i]=\operatorname{logaddexp}(E[i]+\log a_t,\ \log b_t)\ \xrightarrow{\ x_t=k\ \text{: whole column becomes}\ \log c_t\ }\ ,\quad U[k]=E[k] \]

\[ q=\text{model\_output}-V,\qquad S=\operatorname{LSE}_{\text{cls}}(q),\qquad q\leftarrow q-S \]

\[ W[i]=\operatorname{logaddexp}(q[i]+\log \bar a_{t-1},\ \log \bar b_{t-1})\ (i<k),\qquad W[k]=\log \bar c_{t-1} \]

\[ \text{logits}=W+U+S \]

Note the three indices: \(V\) uses the **cumulative table at index \(t\)**, \(U\) the **single-step table at index \(t\)**, \(W\) the **cumulative table at index \(t-1\)**.

Sampling follows the upstream Gumbel recipe, but the randomness comes from an argument:

\[ \text{noise}=-\log\!\big(-\log(u+10^{-30})+10^{-30}\big),\qquad \text{prev}=\arg\max_{\text{cls}}(\text{logits}+\text{noise}) \]

Return `(prev, logits)`: `prev` is an int64 \((B,L)\) tensor, `logits` is float64 — \((B,C,L)\) when \(t>0\) and \((B,C-1,L)\) when \(t=0\).

Six places where this goes wrong:

1. **The two table kinds have different lengths.** Single-step tables are length \(T\), cumulative tables length \(T+1\), whose last entry is a sentinel that is never read (upstream writes \(1.0\) and \(0.0\), so \(\bar b_T=0\)). Reading the cumulative table at index \(t+1\) hits that sentinel when \(t=T-1\) and yields \(-\infty\).
2. **At \(t=0\) `model_output` has only \(k\) classes**, `noise` must have shape \((B,k,L)\), and the returned logits have one row less. Writing an extra mask row breaks the whole shape chain.
3. **The mask row must be saved before `[:, :-1, :]` drops it.** \(U[k]=E[k]\) plays the role of \(q(x_t\mid x_{t-1}=\text{mask})\): it is \(0\) (probability 1) when the observation is the mask, and the clamped value otherwise.
4. **\(\log 0\) is the clamp, not \(-\infty\).** It appears not only in the non-matching one-hot entries (where \(\log(a+b)\) overwrites it and you cannot see the difference) but also as \(U[k]\), surviving all the way into the final logits: in the example the mask row of an unmasked pixel is \(-68.755728\), not \(-\infty\).
5. **The denominator of \(b\) is \(k=C-1\)**, not \(C\) and not 1. Dropping it and writing \(b=1-a-c\) turns the matrix row sum into \(\bar a_t+k(1-\bar a_t-\bar c_t)+\bar c_t=1+(k-1)(1-\bar a_t-\bar c_t)\neq1\), which immediately breaks the conservation law in the next item.
6. **Neither the intermediate normalization nor the final \(S\) can be dropped**: \(\operatorname{logaddexp}\) is not linear, so subtracting \(S\), passing through \(W\), and adding \(S\) back is exactly upstream's expression. It has a checkable consequence — per pixel, `exp(logits).sum(cls)` always equals `exp(model_output).sum(cls)`, independent of \(t\).

## Example

```
num_vec_classes = 3          # classes 0,1 are known; 2 is the mask class
at  = [0.80, 0.60, 0.40, 0.20]              # T = 4
ct  = [0.10, 0.20, 0.30, 0.40]
att = [0.80, 0.48, 0.192, 0.0384, 1.0]      # trailing sentinel is never read
ctt = [0.10, 0.28, 0.496, 0.6976, 0.0]
bt  = (1 - at - ct) / 2 = [0.05, 0.10, 0.15, 0.20]
btt = (1 - att - ctt) / 2 = [0.05, 0.12, 0.156, 0.132, 0.0]

model_output = [[[-0.2, 0.3, -1.0, 0.5],
                 [ 0.4, -0.7,  0.1, -0.3]]]     # (B, C-1, L)
sample       = [[0, 2, 1, 2]]                    # pixels 1 and 3 are masked
```

With `gumbel_uniform` entirely `exp(-1)` the noise is exactly 0 and `prev` is the bare argmax (ties take the lowest index):

```
t=0 prev=[[1, 0, 1, 0]]   shape (1, 2, 4)
   [[-0.2, 0.3, -1.0, 0.5],
    [ 0.4, -0.7,  0.1, -0.3]]

t=1 prev=[[0, 0, 1, 0]]   shape (1, 3, 4)
   [[  0.22076,  -0.177582,  -1.310108,   0.027097],
    [  0.061595,  -1.050658,   0.185034,  -0.675965],
    [-68.755728,   -0.416358, -69.78935,   -0.158519]]

t=2 prev=[[0, 2, 1, 2]]
   [[ 0.341843,  -0.642624,  -1.31155,  -0.427565],
    [-0.102015,  -1.279465,   0.185357,  -0.945435],
    [-67.872666,   0.041475, -68.639611,   0.299314]]

t=3 prev=[[0, 2, 1, 2]]
   [[ 0.317762,  -1.159089,  -0.925513,  -0.928227],
    [-0.065598,  -1.514885,   0.073922,  -1.219759],
    [-66.999512,   0.272192, -67.551656,   0.530031]]

exp(logits).sum(cls) = [2.310555, 1.846444, 1.473050, 2.389539]   # same for all four t
```

Hand-computing the first entry at `t=1` (observation \(x_t=0\)) checks all three indices: \(p=(e^{-0.2},e^{0.4})=(0.818731,1.491825)\); \(\bar a_1=0.48,\bar c_1=0.28,\bar b_1=0.12\) give \(V=(0.60,0.12)\), \(r=(1.364551,12.431872)\), \(S=13.796424\), \(w=(0.098906,0.901094)\); cumulative index \(t-1=0\) gives \(\bar a_0=0.8,\bar b_0=0.05\), hence \(W_0=0.05+0.8w_0=0.129125\); the single-step table gives \(a_1+b_1=0.7\) (the observation matches). \(0.7\times0.129125\times13.796424=1.247024\), \(\log=0.22076\) ✓. The mask row is \(10^{-30}\times\bar c_0\times S=1.38\times10^{-30}\), \(\log=-68.755728\) ✓.

With genuine randomness instead:

```
gumbel_uniform = [[[0.50, 0.10, 0.90, 0.20],
                   [0.30, 0.70, 0.40, 0.60],
                   [0.80, 0.05, 0.25, 0.65]]]

t=0 (has no mask row) prev=[[1, 1, 0, 1]]
t=1, 2, 3           prev=[[0, 1, 0, 2]]     # logits are byte-identical to the table above
```

Noise changes only the sample, never the returned distribution; but passing this \((1,3,4)\) tensor at \(t=0\) raises immediately, because that step has only two rows.

## Constraints

- `model_output` must be a 3-D float tensor \((B, C-1, L)\) with all entries finite; `sample` must be an integer tensor (`int32`/`int64` accepted, `bool` and floats rejected) of shape \((B, L)\) with values in \([0, C-1]\).
- `at` and `ct` are 1-D tensors of equal length \(T\ge 2\), entries in \((0,1)\) with \(a_t+c_t<1\); `att` and `ctt` must have length \(T+1\) and the **first \(T\)** entries satisfy the same conditions (the trailing sentinel is not validated because it is unreachable).
- `timestep` is a Python `int` (`bool` rejected) with \(0\le t<T\); `num_vec_classes` is a Python `int` \(\ge 2\).
- `gumbel_uniform` is a float tensor with values in \([0,1)\) and shape \((B,C,L)\) for \(t>0\) or \((B,C-1,L)\) for \(t=0\). Any violated constraint raises `ValueError`.
- Internal arithmetic is float64; `logits` is returned as float64 and `prev` as int64. `import diffusers` is forbidden.
