# Notes: Why \(p=r\) Is Unbiased, and What the fwd/bwd Biases Are Each Saving

## Origin

Stochastic rounding entered the deep-learning mainstream with Gupta, Agrawal et al., "Training Deep Networks with Finite Precision" (2012); hardware such as the Graphcore IPU and Intel XPU later shipped SR as a selectable rounding mode. The symmetric "forward toward zero, backward away from zero" pairing comes from the OSDP line of work (Groze et al., 2023): activation quantization error is absorbed by the activation distribution, whereas gradient accumulation that is systematically biased low stops weights from ever growing to their full magnitude. In an interview, explaining "why truncation is biased, why SR is unbiased, and when you deliberately want a bias" is rarer and more valuable than reciting formulas.

## Unbiasedness Takes One Line

Let \(lo\) and \(hi=lo+\Delta\) be neighboring grid points and $x = lo + r\Delta$. Taking \(hi\) with probability \(p\) and \(lo\) with probability \(1-p\) gives
$$\mathbb E[\text{out}]=lo+p\Delta=lo+r\Delta=x$$
So \(p=r\) for `sr` is not "empirically best" — it is the **unique** choice that makes the expectation equal the true value. Conversely `fwd` uses \(p=\max(0,2r-1)\), which forces \(lo\) whenever \(r\le0.5\), so the expectation \(lo+p\Delta\le x\); `bwd` uses \(p=\min(1,2r)\), which forces \(hi\) whenever \(r\ge0.5\), so the expectation is \(\ge x\). Both are intentional biases: the first erases small activation residuals entirely to protect dynamic range, the second makes sure the downward bias of gradient accumulation cannot swallow small gradients.

## Why r Can Be Written as (u32 & 0xFFFF)/65536

bf16 is the high 16 bits of a float32, so "between two adjacent bf16 values" corresponds exactly to incrementing the high 16-bit integer by one — the grid is uniform in **integer** space. Dividing the low 16 bits by \(2^{16}\) is therefore immediately "the relative position inside this ulp", with no need to extract the exponent and compute a binade step first. This single observation yields \(lo\) (truncate), \(hi\) (add one then truncate) and \(r\) at once, and it covers subnormals and negative numbers for free (the sign occupies the top bit, while \(r\) still describes the magnitude direction).

## Core Ideas and Pitfalls

- **The test must be strictly less than, `u < p`**: \(u\in[0,1)\) and \(p\) can equal \(u\) exactly. Using `<=` would make `fwd` carry when \(p=0,\,u=0\), breaking the "small residuals always truncate" semantics; the tests deliberately include boundary cases such as \(r=0.25,u=0.25\) and \(r=0.75,u=0.75\).
- **\(r=0\) must return the input unchanged**: values already on the grid (including ±0 and ±inf) cannot be moved to \(hi\). Since \(p=0\) and the comparison is strict, this holds automatically under `sr` and `fwd`, and also under `bwd` because \(p=2r=0\); what really needs protecting is the sign bit of \(\pm0\).
- **Carrying overflows to inf**: when \(|x|\) sits just below the bf16 maximum, the exponent carry in \(hi\) produces ±inf. This problem does not saturate — the exact opposite of `nu-fp8-e5m2-block-scale`, where the scale guarantees overflow never happens.
- **\(u\) must match \(x\)'s shape, no broadcasting**: a real kernel draws noise per element from a Philox counter; broadcasting one noise value would make the errors of a whole group perfectly correlated, so the judge treats it as a ValueError instead of helping you expand it.
- **Do not reach for `torch.uniform`**: that operator does not exist in this version (`torch.rand` does), and the noise has to come from the arguments anyway, otherwise judging would not be reproducible.

## How It Differs from Neighboring Problems

`nu-bf16-rounding` performs deterministic rounding (rne/rz/rna) on the same grid; this problem makes a probabilistic choice on that very same grid. `fw-autocast-scaler` and `fp16-loss-scaling` deal with dynamic scale, not rounding direction. `ag-dropout-backward` is also random, but there the randomness is a mask rather than a number representation. The distinctive material here is the three bias directions plus the structural observation that the bit pattern is uniform in integer space.

## Cross-Check Basis and Complexity

The oracle redoes the same bit manipulation on plain Python scalars with `struct.pack/unpack`, never touching tensor views, which is independent from the reference solution's `view(torch.int32)` path, and results are compared bitwise. The bias direction is checked separately on 40000 uniform samples over \([1,3)\) by asserting the sign and the ordering of the mean error (`rz` is more negative than `fwd`, and `|bias|` of `sr` stays under 2e-4). Complexity is \(O(N)\) with a constant number of bit operations per element.
