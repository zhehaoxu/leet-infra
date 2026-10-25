# In Depth: Strong/Weak Scaling Speedup, Efficiency and Saturation

## Source

Megatron-LM (Shoeybi et al., arXiv:1909.08053) and MLPerf Training draw their scaling curves with exactly this
pair of conventions: **strong scaling** fixes the total problem size and watches step time fall as GPUs are added;
**weak scaling** fixes the per-GPU load (e.g. a constant micro-batch per GPU) and watches whether step time stays
flat. Superlinear points are documented in Sun et al.'s scaling-of-training study
(arXiv:2201.11854), where sharding small models collapses the communication share and gives $E>1$. The
strong/weak definitions go back to Dowdy & Foster (1982).

## Key Points and Pitfalls

1. **The numerator differs between the modes**: in strong scaling the samples per step are fixed, so aggregate
   throughput is only $1/t_i$; in weak scaling samples per GPU are fixed, so samples per step grow with $w_i$ and
   throughput is $w_i/t_i$. Applying $w/t$ to strong scaling inflates efficiency by $w_i/w_0$ - 16x at 16 GPUs -
   which is the classic mistake.
2. **Efficiency divides by the GPU multiplier only**: $\text{efficiency}=\text{speedup}/(w_i/w_0)$ holds in both
   modes; expanded it is $(t_0/t_i)(w_0/w_i)$ for strong and $t_0/t_i$ for weak. Do not divide the weak-scaling
   efficiency by $w$ a second time.
3. **The baseline need not be one GPU**: every ratio is taken against entry 0, so a curve such as
   `world_sizes=[2,8,32]` must be normalised by $w_0=2$.
4. **Superlinear means strictly greater**: efficiency exactly $1.0$ (the 4-GPU tier above, $100/25/4=1$) is not a
   superlinear point.
5. **Saturation includes "flat" and takes the first hit**: the test is
   $\text{throughput}_i\le\text{throughput}_{i-1}$, so $t_i=t_{i-1}$ ("more GPUs, zero gain") counts; if the
   curve rises strictly the answer is `-1`, not the index of the last tier.
6. **Self-proving identities**: $\text{speedup}_0=\text{efficiency}_0=1$ and pointwise
   $\text{speedup}_i=\text{efficiency}_i\cdot(w_i/w_0)$ - the grader asserts this structure.
7. **Shapes of perfect scaling**: perfect strong $\Rightarrow t_i=t_0 w_0/w_i$, $\text{speedup}=w_i/w_0$, efficiency 1;
   perfect weak $\Rightarrow t_i=t_0$, efficiency 1. One degenerate case pins each curve.

## Cross-check Approach

The reference solution builds the `throughput` array from its definition and takes ratios; the grader's `_oracle`
uses the closed forms (strong: $t_0/t_i$ and $(t_0 w_0)/(t_i w_i)$ in a single division) and replaces both discrete
tests by **integer cross-multiplication** so that no floating-point rounding can flip a tie: superlinear
$\iff t_0 w_0>t_i w_i$ (strong) / $t_0>t_i$ (weak), saturation $\iff t_i\ge t_{i-1}$ (strong) /
$w_i t_{i-1}\le w_{i-1}t_i$ (weak). Additional cases cover perfect strong and weak scaling, "flat step time means
saturated", the single-point degeneration, the identity $\text{speedup}=\text{efficiency}\cdot w/w_0$, the
$w_0/w$ relation between the two modes' efficiencies, and the consistency of the `superlinear` index set with the
returned efficiency tensor.

## Complexity

$O(n)$ with $n$ the number of configurations ($\le 1024$); no extra allocation beyond the three returned tensors.
