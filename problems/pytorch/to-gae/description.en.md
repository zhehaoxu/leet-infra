# Alignment Training: Generalized Advantage Estimation (GAE)

## Problem

Policy gradients need the "advantage" \(A_t\) — how much better an action is than the state value. A one-step TD error has low variance but high bias; Monte-Carlo returns are unbiased but the variance explodes. **GAE** (Generalized Advantage Estimation, Schulman et al., arXiv:1506.02438) uses a mixing parameter \(\lambda\in[0,1]\) to interpolate smoothly between the two.

Define the one-step TD error
\[ \delta_t = r_t + \gamma\,V_{t+1} - V_t \]
Then the GAE advantage is an infinite-horizon accumulation discounted by \((\gamma\lambda)\), with a backward recursion:
\[ \hat A_t = \sum_{l\ge0}(\gamma\lambda)^l\,\delta_{t+l} = \delta_t + \gamma\lambda\,\hat A_{t+1},\qquad \hat A_{T-1}=\delta_{T-1} \]
The value regression target (return) is
\[ R_t = \hat A_t + V_t \]

Everything past the end of the trajectory counts as terminated: \(V_T=0\) (no bootstrap out of bounds).

## Rules

Input `rewards` (\((T,)\)), `values` (\((T,)\), i.e. \(V_0..V_{T-1}\)), and the scalars `gamma`, `lam`. Return the tuple `(advantages, returns)`, both float32 tensors of shape \((T,)\). Ready-made RL libraries are not allowed.

## Examples

`rewards=[1.0,0.0,1.0]`, `values=[0.5,0.7,0.3]`, `gamma=0.99`, `lam=0.95`:

```
delta = [1.0+0.99*0.7-0.5, 0.0+0.99*0.3-0.7, 1.0+0.99*0-0.3] = [1.193, -0.403, 0.7]
A_2 = 0.7
A_1 = -0.403 + 0.99*0.95*0.7 = 0.255350
A_0 = 1.193 + 0.99*0.95*0.255350 = 1.433157
advantages = [1.433157, 0.255350, 0.699999]
returns    = [1.933157, 0.955350, 1.0]
```

## Constraints

- CPU float32; \(T\ge1\); \(\gamma,\lambda\in[0,1]\).
- The recursion runs backward from \(t=T-1\) to \(0\), with \(\hat A\) initialized (at \(t=T\)) to 0, equivalent to \(V_T=0\).
- \(\lambda=0\) degenerates to the one-step TD \(\hat A_t=\delta_t\); \(\lambda=1,\gamma=1\) approaches Monte Carlo \(R_t=\sum_{l\ge t}r_l\) (this implementation still terminates with \(V_T=0\)).
