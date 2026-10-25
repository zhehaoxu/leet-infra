# In Depth: AdamW Decoupled Weight Decay

## Source
Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019 (arXiv:1711.05101). The reference implementation is PyTorch's native `torch.optim.AdamW`.

## Why "decouple" it
Doing \(L_2\) regularization inside vanilla Adam means adding \(\lambda p\) to the gradient \(g\) and only then running the adaptive update, so that term too gets divided by \(\sqrt{\hat v}\): the intent was a fixed per-step shrinkage of the parameter, but the actual shrinkage is amplified or shrunk by the gradient's second moment, coupling the regularization strength to the gradient-noise scale. AdamW changes the decay from "entering the gradient" to "acting on the parameter directly":

\[ p \leftarrow p - lr\cdot\text{wd}\cdot p \;=\; p\,(1 - lr\cdot\text{wd}) \]

This step happens **before** the moment estimates and the adaptive update and is completely independent of \(m,v\) — hence "decoupled". This is also why Adam's \(L_2\) breaks down when \(\beta_2\) is large and the gradients are sparse, while AdamW is unaffected.

## Step-by-step alignment with torch.optim.AdamW (cross-check convention)
The judge uses `torch.optim.AdamW(..., amsgrad=False, foreach=False, fused=False)` as the external reference. The three easy-to-get-wrong points that must align:

1. **Order**: first `p.mul_(1 - lr*wd)`, then update \(m,v\), and finally `p.addcdiv_`.
2. **How bias correction is written**: PyTorch writes the update as
   \[ p \leftarrow p - \frac{lr}{1-\beta_1^{t}}\cdot\frac{m}{\sqrt{v}/\sqrt{1-\beta_2^{t}}+\epsilon} \]
   Note that the denominator is \(\sqrt{v}/\sqrt{1-\beta_2^t}\) (the division defining \(\hat v\) is placed inside the denominator) and \(eps\) is added after that step — **not** computing \(\hat m,\hat v\) separately and then adding \(eps\) straight onto \(\sqrt{\hat v}\). The two forms are mathematically equivalent but land differently in floating point, so for element-wise agreement you must copy PyTorch's operator order: `m/(v.sqrt()/sqrt(bc2) + eps)`.
3. **Step counter**: each parameter counts independently from 1 (first step \(bc_1=1-\beta_1\)), regardless of sparsity.

Parameters with `grad is None` simply hit `continue` in PyTorch and receive no weight decay either — the judge has a dedicated test case for this.

## Complexity
\(O(\text{numel})\) per parameter per step, two extra states (\(m,v\)), the same as Adam.
