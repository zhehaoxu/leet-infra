# kl_div Loss: Log-Prob Input and the batchmean Convention

## Problem Description

Knowledge distillation, the KL penalty of RLHF (the divergence from the reference model in PPO/GRPO) and the VAE regulariser all call `F.kl_div`, and its interface is counter-intuitive in two ways: **the first argument is $\log q$ (the student's or policy's log-probability), not $q$**, and **the default `reduction="mean"` is not the mathematical KL**. The divergence itself is

$$\mathrm{KL}(P\,\|\,Q) = \sum_{c} p_c \log\frac{p_c}{q_c}, \qquad \text{elementwise term } \ell_c = p_c(\log p_c - \log q_c)$$

Summing over the class axis per sample and then averaging over samples is what KL means mathematically — that is exactly `"batchmean"` (divide by $B$); `"mean"` divides by $B\times C$ instead, shrinking the KL by a factor $C$, which explains most of the mistuned KL weights seen in distillation code.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- `log_pred` is $\log q$ (e.g. the output of `log_softmax`) and `target` is $p$; with `log_target=True` the target is already $\log p$ and the elementwise term is $e^{\text{target}}(\text{target}-\text{log\_pred})$.
- Four `reduction` modes: `"batchmean"` (default, sum ÷ length of dim 0 of the output, i.e. $B$) / `"mean"` (sum ÷ element count) / `"sum"` / `"none"` (the broadcast-shaped elementwise tensor).
- Elementwise convention: $p(\log p-\log q)$ for $p>0$; $+\infty$ when $p>0$ and $q=0$; at $p=0$ the mathematical convention $0\log 0 := 0$ applies, contributing exactly 0 (this deliberately **does not copy** the `F.kl_div` detail that returns NaN when $p=0$ and $q=0$).
- `target` may broadcast against `log_pred` (one shared target distribution applied to every step of a batch); an illegal `reduction` raises ValueError.

## Example

logits $=[[2,0,-1],[0,1,3]]$ passed through `log_softmax(dim=1)` give $\log q$, with $p=[[0.5,0.3,0.2],[0.1,0.6,0.3]]$ (numbers taken from the EXAMPLE_* constants filled in from test_cases.py):

```
elem(none) = [[-0.26165056228637695, 0.2897619903087616, 0.31208163499832153],
              [ 0.0867260992527008,  0.9954123497009277, -0.3102380335330963]]
batchmean  = 0.556046724319458
sum        = 1.112093448638916
mean       = sum / 6
```

The first entry $0.5(\log 0.5 - (-0.1694)) = -0.2617$ is negative — elementwise terms **may be positive or negative**, only the summed KL is guaranteed non-negative. `batchmean = sum/2 = 0.5560` whereas `mean = sum/6`: they differ by exactly the number of classes, 3.

## Constraints

- Input is float32, computed in float64, returned as float32; on dense inputs (all $p>0$) compared pointwise against `F.kl_div` with atol=1e-6, rtol=1e-5
- The judge covers the one-hot hard-label degeneration (the $p=1$ entry equals $-\log q$, i.e. cross entropy), the $q=0$ case giving $+\infty$, the $p=0$ zero convention, a broadcast target, and four illegal `reduction` values
- Division of labour with `kl-divergence` (scalar KL between two distributions): that problem examines the mathematical definition, this one the PyTorch loss-API conventions.
