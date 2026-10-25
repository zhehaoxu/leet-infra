# NovoGrad Layer-wise Gradient Normalization Explained

## Source
Ginsburg, Brattoli, Bühlmann, et al., *Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training Deep Networks* (arXiv:1905.11286). The reference implementation follows `NovoGrad` from kozistr/pytorch-optimizer (`weight_decouple=True`, `grad_averaging` optional, `adam_debias=False`).

## Three structural differences from Adam
NovoGrad replaces Adam's "element-wise second moment + bias correction" with "per-layer scalar normalization + pure momentum". The specific differences:

1. **The second moment is a scalar**: for a given parameter (a whole layer), first compute \(\lVert g\rVert^2=\sum_i g_i^2\), then take its EMA to get \(s\) (one number per layer rather than one per element). The denominator is \(\sqrt{s}+\epsilon\), which scales the whole layer uniformly. This saves memory and naturally performs "layer normalization".
2. **Special first step**: in the library implementation, the first time a parameter is seen it sets \(s=\lVert g\rVert^2\) directly (not \(0+(1-\beta_2)\lVert g\rVert^2\)), and the momentum \(m=\hat g\) (not \(\beta_1\cdot0+(1-\beta_1)\hat g\)). Only afterwards does the normal EMA/momentum recursion apply.
3. **No bias correction**: the update is \(p\leftarrow p-lr\cdot m\), the step size is just `lr`, and \(1-\beta^t\) never appears. The coefficient with which \(\hat g\) accumulates into the momentum is \(1\) by default; only `grad_averaging=True` multiplies by \((1-\beta_1)\) (a switch provided to align with Adam semantics, off by default).

## Order of weight decay
The paper argues that NovoGrad works best with decoupled weight decay: first compute the normalized gradient \(\hat g\), then apply \(p\leftarrow p(1-lr\cdot\text{wd})\) to the parameters, and finally use \(\hat g\) to update the momentum and the parameters. This problem uses the **decoupled** form (consistent with the paper's recommendation); the decay acts on \(p\) and does not pollute the gradient.

## Grading details
The grader uses an independent vectorized recomputation as the reference (`_oracle_novograd` in `test_cases.py`): element-wise, along an implementation path **different** from the reference solution (scalar `pow(2).sum()`, the algebraic form `b2*s+(1-b2)*g2`, etc.), compared under multiple shapes (including 2D weights + 1D biases) and multiple steps of random gradients with `allclose(atol=1e-5, rtol=1e-5)`. Parameters with `grad is None` must be skipped together with their weight decay — dedicated cases cover this.

## Complexity
\(O(\text{numel})\) per parameter per step (plus one extra whole-layer sum); the extra state is only one momentum \(m\) plus one scalar \(s\) per layer, half the memory of Adam.
