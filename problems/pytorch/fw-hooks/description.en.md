# Forward/Backward Hooks: Capturing Activations and Rewriting Gradients

## Problem

Hooks are the hands the framework lets us reach into `nn.Module`'s execution flow: a forward hook can capture activations (and even **replace** the module's output), and the new gradients returned by `register_full_backward_hook` propagate upstream verbatim — the exact mechanism behind feature visualization, gradient probing, GradCAM and "gradient surgery". This problem deploys five hooks and asks you to record all of their observable consequences into a ledger.

## Rules

Implement `solve()`: run one forward and one backward pass on the fixed model from the starter docstring (explicit weights, no randomness) with the five-hook deployment, and return the behavior-ledger dict. Key points:

1. The forward hook on tanh returns `out * 2.0` (non-None): it **replaces** the module output, so fc2 receives \(2\tanh(a_1)\).
2. The `register_full_backward_hook` on fc1 returns `tuple(2*g+1 for g in gin)`: it rewrites the gradient sent to the input, so `x.grad` becomes \(2g+1\); parameter gradients are **not polluted by the rewrite** and keep their analytic values.
3. The legacy `register_backward_hook` on fc2 only counts and does not rewrite (returns `None`).
4. Every hook counts its firings; `acts` records the module outputs **before** replacement.
5. The rewritten `x.grad` must equal the analytically recomputed value: hand-compute the raw \(g\) first, then apply \(2g+1\).

## Examples

```
Input: none
Output:
{
 "loss": 0.003973240032792091,
 "acts": {"fc1": [0.30000001192092896, 0.75],
          "tanh": [0.29131263494491577, 0.6351489424705505],
          "fc2": [0.9369663596153259]},
 "fc2_input": [0.5826252698898315, 1.270297884941101],
 "fired": {"fc1_fwd": 1, "tanh_fwd": 1, "fc2_fwd": 1, "fc1_bwd": 1, "fc2_bwd": 1},
 "x_grad": [0.945848822593689, 0.9139541983604431, 0.8820595145225525],
 "fc1_weight_grad": [-0.034610647708177567, 0.06922129541635513, -0.13844259083271027,
                     -0.04512597247958183, 0.09025194495916367, -0.18050388991832733],
 "fc2_weight_grad": [-0.0734499841928482, -0.16014300286769867],
 "fc1_bias_grad": [-0.06922129541635513, -0.09025194495916367]
}
```

Derivation: \(a_1=(0.3,0.75)\), \(h=2\tanh(a_1)=(0.5826,1.2703)\), \(out=0.3h_0+0.6h_1=0.93697\), \(loss=(out-1)^2=0.003973\). Raw \(\partial L/\partial x_0=-0.02708\); after the hook rewrite \(2(-0.02708)+1=0.94585\).

## Constraints

- Numeric comparison uses `atol=1e-5, rtol=1e-5`; firing counts and the dict structure must match exactly.
- A single forward plus a single backward pass; no randomness.
