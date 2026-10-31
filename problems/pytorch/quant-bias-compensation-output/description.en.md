# Quantization: Bias Compensation via Streaming Per-Channel Output Correction

## Description

A quantized layer always outputs something slightly different from its floating-point counterpart. Most PTQ methods fix the **quantizer** (pick a scale, pick a rounding direction); Bias Compensation (Gong, Dong et al., arXiv:2404.01892) takes another route: it appends a per-channel bias $l$ **after** the quantized layer and estimates it straight from the output difference on calibration data, completely decoupled from the quantizer.

This exercise gives you the floating-point and quantized outputs of $T$ calibration forwards and asks you to update that bias **streamingly**, following the upstream implementation. No gradients, no backpropagation, no randomness: the same input must always produce a bit-identical answer. The difficulty is not the maths but the handful of details in the upstream code that are easy to get wrong — the counter is incremented before it is used, `post_bias` adds back the bias as it stood **before** the update, the reduction only happens over the non-channel axes of a single forward, and negative axes are normalized against the rank of a single forward.

## Rules

Let $d=\texttt{float\_batches.dim()}-1$ be the rank of a single forward (axis 0 counts "which calibration forward", so it never participates in the channel decision) and $C=\prod(\texttt{bias\_shape})$ the bias length. All internal math is float64; results are returned in the input dtype.

1. **Axis normalization**: each `channel_axis` entry $a<0$ becomes $a+d$ (against $d$, not $d+1$). Then collect every **non-channel** axis in ascending order as `mean_axis`, and build `bias_shape` of length $d$ by writing the size at channel positions and 1 elsewhere.
2. **The per-update mean vector**: for update $t$ ($t=0..T-1$) start from $\Delta_t=\texttt{float\_batches}[t]-\texttt{quant\_batches}[t]$, add the **current** bias broadcast to `bias_shape` when `post_bias=True`, and then
   $$\texttt{bias\_updates}[t]=\operatorname{flatten}\bigl(\operatorname{mean}_{\texttt{mean\_axis}}(\Delta_t)\bigr).$$
   When `mean_axis` is empty (every axis was declared a channel axis) the upstream code **skips the mean entirely** and just flattens $\Delta_t$; this exercise copies that behaviour.
3. **Streaming average**: upstream runs `nsamples += 1` first and only then `(bias * (nsamples - 1) + update) / nsamples`, i.e.
   $$b_{t+1}=\frac{b_t\cdot t+\texttt{bias\_updates}[t]}{t+1},\qquad b_0=\texttt{init\_bias}\ \text{or}\ 0.$$
   The first sample carries weight 1 (it simply overwrites the initial value) and every later sample carries weight $1/(t+1)$; **the size of the calibration batch plays no role**, since the batch axis was already averaged away in step 2.
4. **Finish**: `bias` is $b_T$; `corrected_output` applies the **final** bias to every forward as $Q_t+\mathrm{broadcast}(b_T)$; and $\texttt{residual\_mse}=\operatorname{mean}\bigl((F-\texttt{corrected\_output})^2\bigr)$ averages over all elements.

Return a dict with exactly 8 keys:

- `channel_axis_norm` tuple[int]: the normalized channel axes;
- `bias_shape` tuple[int]: length $d$, 1 at every non-channel axis;
- `mean_axis` tuple[int]: the non-channel axes in ascending order;
- `bias_updates` float $(T, C)$: the per-update mean vectors from step 2, before any averaging;
- `bias_after` float $(T, C)$: the bias snapshot after each update, `bias_after[t]` $=b_{t+1}$;
- `bias` float $(C,)$: the final bias;
- `corrected_output` float $(T, *R)$: `quant_batches` plus the final bias;
- `residual_mse` float 0-dim: the output MSE under the final bias.

## Task

- Inputs: `float_batches` and `quant_batches`, float32 or float64 tensors of identical shape and dtype, $(T,*R)$ with single-forward rank $d\ge 2$; `channel_axis` an int or a tuple/list of ints; `init_bias` either `None` or $C$ elements; `post_bias` a bool; `channel_size` either `None` or an int.
- Output: the 8-key dict above; float tolerance `1e-5` for float32 and `1e-9` for float64; the three tuple keys are compared as Python tuples.
- Raise `ValueError` for: $T=0$, single-forward rank $d<2$, differing shapes, differing dtypes, an empty `channel_axis` tuple, non-integer entries, entries out of range after normalization, duplicate entries, `channel_size` not equal to $C$, or `init_bias` whose element count is not $C$.

## Examples

Example 1: `float_batches` has shape $(T,B,C,S)=(2,2,3,2)$ and `channel_axis=1`, so `bias_shape=(1,3,1)` and `mean_axis=(0,2)` — each forward first averages over the batch and sequence axes to produce 3 per-channel updates. Values come from float64 inputs with `init_bias=None` and `post_bias=True`.

```
F[0][:, 0, :] = [[0.10, 0.20], [0.30, 0.40], [0.50, 0.60]]
F[0][:, 1, :] = [[0.70, 0.80], [0.90, 1.00], [0.11, 0.22]]
F[1][:, 0, :] = [[0.12, 0.18], [0.34, 0.36], [0.48, 0.62]]
F[1][:, 1, :] = [[0.68, 0.82], [0.88, 1.02], [0.09, 0.25]]
Q[0][:, 0, :] = [[0.05, 0.15], [0.25, 0.35], [0.45, 0.55]]
Q[0][:, 1, :] = [[0.60, 0.75], [0.85, 0.95], [0.05, 0.20]]
Q[1][:, 0, :] = [[0.06, 0.14], [0.30, 0.30], [0.44, 0.60]]
Q[1][:, 1, :] = [[0.62, 0.78], [0.80, 1.00], [0.10, 0.24]]
channel_axis = 1

channel_axis_norm = (1,)
bias_shape        = (1, 3, 1)
mean_axis         = (0, 2)
bias_updates      = [[0.06250000, 0.05000000, 0.04500000],
                     [0.11250000, 0.10000000, 0.06000000]]
bias_after        = [[0.06250000, 0.05000000, 0.04500000],
                     [0.08750000, 0.07500000, 0.05250000]]
bias              = [0.08750000, 0.07500000, 0.05250000]
corrected_output[0][:, 0, :] = [[0.13750000, 0.23750000], [0.32500000, 0.42500000], [0.50250000, 0.60250000]]
corrected_output[0][:, 1, :] = [[0.68750000, 0.83750000], [0.92500000, 1.02500000], [0.10250000, 0.25250000]]
corrected_output[1][:, 0, :] = [[0.14750000, 0.22750000], [0.37500000, 0.37500000], [0.49250000, 0.65250000]]
corrected_output[1][:, 1, :] = [[0.70750000, 0.86750000], [0.87500000, 1.07500000], [0.15250000, 0.29250000]]
residual_mse      = 0.00106042
```

(For channel 0 the four differences in the first update are $0.05,0.05,0.10,0.05$, mean $0.0625$; $b_1$ equals it outright because the first sample carries weight 1. In the second update $b_1$ is added back first, the four differences become $0.1225,0.1025,0.1225,0.1025$ with mean $0.1125$, hence $b_2=(0.0625\cdot 1+0.1125)/2=0.0875$.)

Example 2: `channel_axis=(0, 1)` declares **both** axes of a single forward as channel axes, so `mean_axis` is empty, no averaging happens, and the bias length is $2\times 3=6$.

```
F[0] = [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]]
F[1] = [[0.20, 0.10, 0.40], [0.50, 0.30, 0.70]]
Q[0] = [[0.00, 0.25, 0.35], [0.45, 0.55, 0.50]]
Q[1] = [[0.15, 0.15, 0.45], [0.55, 0.35, 0.65]]
channel_axis = (0, 1)

channel_axis_norm = (0, 1)
bias_shape        = (2, 3)
mean_axis         = ()
bias_updates      = [[0.10000000, -0.05000000, -0.05000000, -0.05000000, -0.05000000, 0.10000000],
                     [0.15000000, -0.10000000, -0.10000000, -0.10000000, -0.10000000, 0.15000000]]
bias_after        = [[0.10000000, -0.05000000, -0.05000000, -0.05000000, -0.05000000, 0.10000000],
                     [0.12500000, -0.07500000, -0.07500000, -0.07500000, -0.07500000, 0.12500000]]
bias              = [0.12500000, -0.07500000, -0.07500000, -0.07500000, -0.07500000, 0.12500000]
corrected_output[0] = [[0.12500000, 0.17500000, 0.27500000], [0.37500000, 0.47500000, 0.62500000]]
corrected_output[1] = [[0.27500000, 0.07500000, 0.37500000], [0.47500000, 0.27500000, 0.77500000]]
residual_mse      = 0.00145833
```

## Constraints

- `bias_shape`/`mean_axis` only look at the $d$ axes of a **single forward**; axis 0 is always "which update" and is never a channel axis, and negative axes normalize against $d$, not $d+1$.
- The streaming weights are $1/(t+1)$ regardless of how many elements a forward contains: scaling the batch axis of `float_batches` by 10 changes nothing.
- With `post_bias=True` the value added back is the bias as it stood after the **previous** update (not this one), which is equivalent to subtracting the bias from the quantized output before averaging.
- `init_bias` is the starting value $b_0$ of the running average. With `post_bias=False` it never reaches the result at all: $b_1=(b_0\cdot 0+u_0)/1$ gives it weight 0. With `post_bias=True` it is folded into $\Delta_0$ and its coefficient stays exactly 1 for the rest of the stream, so the whole `bias_updates`/`bias_after`/`bias` trajectory shifts by exactly `init_bias`.
- Judging looks only at the returned dict; the implementation strategy is free.
